"""Generate benign EnvScaler trajectories from environment and task JSON files.

This entry point reuses the normal task-solving agent and non-conversational RL
environment from ``eval/``.  It intentionally does not use
``attacker_agent/envscaler_env`` because that package is designed for executing
prompt-injection plans rather than solving the original benign tasks.

Example (run from ``attacker_agent/``)::

    python generate_trajectories.py \
        --env-file data/envs/rl.json \
        --task-file data/tasks/rl/rl_task.json \
        --output-file data/traj/rl_traj.json \
        --model gpt-4.1 \
        --infer-mode fc \
        --num-workers 4

The output schema is the same as ``eval/run_main.py`` and contains ``task_info``,
``messages``, ``trajectory``, rewards, termination flags, and final observations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval"


TaskKey = tuple[str, str]


def load_eval_components() -> tuple[type[Any], type[Any]]:
    """Import eval components only when trajectories will actually be generated."""
    # eval/ uses top-level imports such as ``from agent...`` and
    # ``from envscaler_env...``. Put it first so those names resolve to the
    # normal trajectory collector instead of the attack environment.
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))

    try:
        from agent.task_solve_agent import TaskSolveAgent
        from envscaler_env import EnvScalerNonConvRLEnv
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Missing runtime dependency {exc.name!r}. "
            "Install the project dependencies with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    class InMemoryEnvScalerNonConvRLEnv(EnvScalerNonConvRLEnv):
        """EnvScaler environment that reuses JSON data loaded once."""

        def __init__(
            self,
            *,
            mode: str,
            env_items: dict[str, dict[str, Any]],
            task_items: list[dict[str, Any]],
        ) -> None:
            if mode not in {"train", "eval"}:
                raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
            self.mode = mode
            self.env_items = env_items
            self.task_items = task_items
            self.reset_attributes()

    return TaskSolveAgent, InMemoryEnvScalerNonConvRLEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an LLM agent on EnvScaler tasks and save benign execution "
            "trajectories in the format produced by eval/run_main.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-file", required=True, help="Environment metadata JSON.")
    parser.add_argument("--task-file", required=True, help="Task/scenario JSON list.")
    parser.add_argument("--output-file", required=True, help="Trajectory output JSON.")
    parser.add_argument("--model", default="gpt-4.1", help="Agent model name.")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai"],
        help="Inference provider supported by the current agent implementation.",
    )
    parser.add_argument(
        "--infer-mode",
        default="fc",
        choices=["fc", "prompt"],
        help="Use native function calling or prompt-formatted tool calls.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable model-specific thinking mode (normally only for Qwen3-like models).",
    )
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--start-index", type=int, default=0, help="Inclusive task index.")
    parser.add_argument("--end-index", type=int, help="Exclusive task index; default is EOF.")
    parser.add_argument("--limit", type=int, help="Maximum tasks after applying the index range.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Atomically save after this many newly completed tasks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing output and run only tasks that are not already present.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop scheduling new work after the first failed task (best with one worker).",
    )
    parser.add_argument(
        "--dotenv",
        help="Optional .env path; defaults to <project-root>/.env when present.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate files and selected tasks without calling an LLM.",
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    if args.end_index is not None and args.end_index < 0:
        parser.error("--end-index must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.max_steps < 1:
        parser.error("--max-steps must be >= 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    return args


def resolve_path(raw_path: str) -> Path:
    """Resolve a CLI path against the caller's current working directory."""
    return Path(raw_path).expanduser().resolve()


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise ValueError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def normalize_env_items(raw_data: Any, path: Path) -> dict[str, dict[str, Any]]:
    if isinstance(raw_data, dict):
        items = raw_data
    elif isinstance(raw_data, list):
        items = {}
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict) or "env_id" not in item:
                raise ValueError(f"{path}: environment item {index} has no env_id")
            env_id = str(item["env_id"])
            if env_id in items:
                raise ValueError(f"{path}: duplicate env_id {env_id!r}")
            items[env_id] = item
    else:
        raise ValueError(f"{path}: expected a JSON object or list of environments")

    normalized: dict[str, dict[str, Any]] = {}
    required = {"env_class_code", "environment_introduction", "tools"}
    for raw_env_id, item in items.items():
        env_id = str(raw_env_id)
        if not isinstance(item, dict):
            raise ValueError(f"{path}: environment {env_id!r} must be an object")
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"{path}: environment {env_id!r} is missing {missing}")
        item_env_id = str(item.get("env_id", env_id))
        if item_env_id != env_id:
            raise ValueError(
                f"{path}: key {env_id!r} disagrees with item env_id {item_env_id!r}"
            )
        normalized[env_id] = item
    if not normalized:
        raise ValueError(f"{path}: no environments found")
    return normalized


def validate_task_items(
    raw_data: Any,
    env_items: dict[str, dict[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    if not isinstance(raw_data, list):
        raise ValueError(f"{path}: expected a top-level JSON list of tasks")
    if not raw_data:
        raise ValueError(f"{path}: no tasks found")

    required = {
        "env_id",
        "env_class_name",
        "task_id",
        "init_config",
        "task",
        "checklist_with_func",
    }
    seen: set[TaskKey] = set()
    validated: list[dict[str, Any]] = []
    for index, raw_item in enumerate(raw_data):
        item = dict(raw_item) if isinstance(raw_item, dict) else raw_item
        if not isinstance(item, dict):
            raise ValueError(f"{path}: task {index} must be an object")
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"{path}: task {index} is missing {missing}")
        env_id = str(item["env_id"])
        if env_id not in env_items:
            raise ValueError(
                f"{path}: task {index} refers to unknown env_id {env_id!r}"
            )
        item["env_id"] = env_id
        expected_class = env_items[env_id].get("env_class_name")
        if expected_class and item["env_class_name"] != expected_class:
            raise ValueError(
                f"{path}: task {index} uses env_class_name "
                f"{item['env_class_name']!r}, expected {expected_class!r}"
            )
        key = task_key(item)
        if key in seen:
            raise ValueError(f"{path}: duplicate (env_id, task_id) {key!r}")
        seen.add(key)
        if not isinstance(item["checklist_with_func"], list) or not item["checklist_with_func"]:
            raise ValueError(
                f"{path}: task {index} has no checklist_with_func; "
                "the RL environment requires at least one reward check"
            )
        validated.append(item)
    return validated


def task_key(task: dict[str, Any]) -> TaskKey:
    return str(task["env_id"]), str(task["task_id"])


def result_key(result: dict[str, Any]) -> TaskKey:
    task_info = result.get("task_info")
    if not isinstance(task_info, dict):
        raise ValueError("Existing result has no task_info object")
    if "env_id" not in task_info or "task_id" not in task_info:
        raise ValueError("Existing result task_info has no env_id/task_id")
    return str(task_info["env_id"]), str(task_info["task_id"])


def select_indices(total: int, start: int, end: int | None, limit: int | None) -> list[int]:
    stop = total if end is None else min(end, total)
    if start > total:
        raise ValueError(f"start index {start} exceeds task count {total}")
    if stop < start:
        raise ValueError(f"end index {stop} is smaller than start index {start}")
    indices = list(range(start, stop))
    return indices[:limit] if limit is not None else indices


def model_env_prefixes(model_name: str) -> list[str]:
    exact = re.sub(r"[^A-Z0-9]+", "_", model_name.upper()).strip("_")
    compact = re.sub(r"[^A-Z0-9]+", "", model_name.upper())
    prefixes = [exact]
    if "QWEN" in compact:
        prefixes.append("QWEN3")
    if "GPT" in compact:
        prefixes.extend(["GPT4", "GPT"])
    if "CLAUDE" in compact:
        prefixes.append("CLAUDE")
    if "GEMINI" in compact:
        prefixes.append("GEMINI")
    return list(dict.fromkeys(prefix for prefix in prefixes if prefix))


def get_model_config(model_name: str) -> tuple[str | None, str | None]:
    """Resolve model-specific credentials, then fall back to OPENAI_* variables."""
    for prefix in model_env_prefixes(model_name):
        api_key = os.getenv(f"{prefix}_API_KEY")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        if api_key or base_url:
            return api_key or os.getenv("OPENAI_API_KEY"), base_url or os.getenv("OPENAI_BASE_URL")
    return os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL")


def atomic_save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def ordered_results(results_by_index: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    return [results_by_index[index] for index in sorted(results_by_index)]


def load_existing_results(
    output_path: Path,
    *,
    resume: bool,
    overwrite: bool,
    task_index_by_key: dict[TaskKey, int],
) -> dict[int, dict[str, Any]]:
    if not output_path.exists() or overwrite:
        return {}
    if not resume:
        raise ValueError(
            f"Output already exists: {output_path}. Use --resume or --overwrite."
        )

    raw_results = load_json(output_path)
    if not isinstance(raw_results, list):
        raise ValueError(f"{output_path}: expected a top-level JSON list")
    results_by_index: dict[int, dict[str, Any]] = {}
    for result in raw_results:
        if not isinstance(result, dict):
            raise ValueError(f"{output_path}: every result must be an object")
        key = result_key(result)
        if key not in task_index_by_key:
            raise ValueError(
                f"{output_path}: result {key!r} is not present in the supplied task file"
            )
        index = task_index_by_key[key]
        if index in results_by_index:
            raise ValueError(f"{output_path}: duplicate result {key!r}")
        results_by_index[index] = result
    return results_by_index


def solve_one(
    *,
    agent_class: type[Any],
    env_class: type[Any],
    task_index: int,
    env_items: dict[str, dict[str, Any]],
    task_items: list[dict[str, Any]],
    mode: str,
    model: str,
    provider: str,
    infer_mode: str,
    temperature: float,
    max_steps: int,
    enable_thinking: bool,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, Any]:
    env = env_class(
        mode=mode,
        env_items=env_items,
        task_items=task_items,
    )
    agent = agent_class(
        env_name="envscaler_non_conversation_rl",
        env=env,
        model=model,
        provider=provider,
        infer_mode=infer_mode,
        temperature=temperature,
        max_steps=max_steps,
        enable_thinking=enable_thinking,
        api_key=api_key,
        base_url=base_url,
    )
    return agent.run(task_index=task_index)


def save_failures(output_path: Path, failures: list[dict[str, Any]]) -> Path:
    error_path = output_path.with_name(f"{output_path.stem}.errors.json")
    atomic_save_json(error_path, failures)
    return error_path


def run_tasks(
    *,
    indices: Iterable[int],
    env_items: dict[str, dict[str, Any]],
    task_items: list[dict[str, Any]],
    results_by_index: dict[int, dict[str, Any]],
    output_path: Path,
    args: argparse.Namespace,
    api_key: str | None,
    base_url: str | None,
    agent_class: type[Any],
    env_class: type[Any],
) -> list[dict[str, Any]]:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Missing runtime dependency 'tqdm'. Install requirements.txt first."
        ) from exc

    pending = [index for index in indices if index not in results_by_index]
    if not pending:
        atomic_save_json(output_path, ordered_results(results_by_index))
        print("No pending tasks; output is already up to date.")
        return []

    failures: list[dict[str, Any]] = []
    completed_since_checkpoint = 0

    def submit(executor: ThreadPoolExecutor, index: int):
        return executor.submit(
            solve_one,
            agent_class=agent_class,
            env_class=env_class,
            task_index=index,
            env_items=env_items,
            task_items=task_items,
            mode=args.mode,
            model=args.model,
            provider=args.provider,
            infer_mode=args.infer_mode,
            temperature=args.temperature,
            max_steps=args.max_steps,
            enable_thinking=args.enable_thinking,
            api_key=api_key,
            base_url=base_url,
        )

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_index = {submit(executor, index): index for index in pending}
        progress = tqdm(total=len(pending), desc="Generating trajectories")
        try:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                task = task_items[index]
                try:
                    results_by_index[index] = future.result()
                    completed_since_checkpoint += 1
                except Exception as exc:
                    failures.append(
                        {
                            "task_index": index,
                            "env_id": task.get("env_id"),
                            "task_id": task.get("task_id"),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    progress.write(
                        f"[ERROR] task index {index}, task_id={task.get('task_id')}: {exc!r}"
                    )
                    if args.fail_fast:
                        for pending_future in future_to_index:
                            pending_future.cancel()
                        break
                finally:
                    progress.update(1)

                if completed_since_checkpoint >= args.checkpoint_every:
                    atomic_save_json(output_path, ordered_results(results_by_index))
                    completed_since_checkpoint = 0
        finally:
            progress.close()

    atomic_save_json(output_path, ordered_results(results_by_index))
    return failures


def main() -> int:
    args = parse_args()
    env_path = resolve_path(args.env_file)
    task_path = resolve_path(args.task_file)
    output_path = resolve_path(args.output_file)

    env_items = normalize_env_items(load_json(env_path), env_path)
    task_items = validate_task_items(load_json(task_path), env_items, task_path)
    indices = select_indices(
        len(task_items),
        args.start_index,
        args.end_index,
        args.limit,
    )
    if not indices:
        raise ValueError("The selected task range is empty")
    task_index_by_key = {task_key(task): index for index, task in enumerate(task_items)}
    results_by_index = load_existing_results(
        output_path,
        resume=args.resume,
        overwrite=args.overwrite,
        task_index_by_key=task_index_by_key,
    )

    already_done = sum(index in results_by_index for index in indices)
    print(f"Environment file : {env_path} ({len(env_items)} environments)")
    print(f"Task file        : {task_path} ({len(task_items)} tasks)")
    print(f"Selected tasks   : {len(indices)} ({already_done} already complete)")
    print(f"Output file      : {output_path}")
    print(f"Model / mode     : {args.model} / {args.infer_mode}")

    if args.validate_only:
        print("Validation passed. No LLM calls were made.")
        return 0

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as exc:
        raise ValueError(
            "Missing runtime dependency 'python-dotenv'. Install requirements.txt first."
        ) from exc

    dotenv_path = resolve_path(args.dotenv) if args.dotenv else PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

    api_key, base_url = get_model_config(args.model)
    if not api_key:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY or a model-specific *_API_KEY in .env."
        )
    print(f"API key          : configured")
    print(f"Base URL         : {base_url or 'provider default'}")
    print(f"Workers          : {args.num_workers}")

    agent_class, env_class = load_eval_components()

    failures = run_tasks(
        indices=indices,
        env_items=env_items,
        task_items=task_items,
        results_by_index=results_by_index,
        output_path=output_path,
        args=args,
        api_key=api_key,
        base_url=base_url,
        agent_class=agent_class,
        env_class=env_class,
    )
    print(f"Saved {len(results_by_index)} trajectories to {output_path}")
    if failures:
        error_path = save_failures(output_path, failures)
        print(f"{len(failures)} task(s) failed; details saved to {error_path}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

import json
import os
import threading
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from attackpoint_task_match.find_operations import find_traj_entry_for_task, find_operations_for_task
from attackpoint_task_match.find_match import filter_attack_points_by_actions
from IPI.gen_IPI_plan import gen_IPI_plan
from IPI.gen_IPI_task import gen_IPI_task
from IPI.IPI import execute_injection
from IPI.gen_IPI_check_func import gen_IPI_check_func
from verification.verify_injection import verify_injection_with_llm
from verification.verify_overlap import (
    check_checkfunc_overlap_with_state,
    check_checkfunc_semantic_overlap_with_llm,
)
from IPI.injection_strategy_utils import apply_injection_strategy


# ============ 线程安全的全局锁和结果列表 ============
_result_lock = threading.Lock()
_all_results: List[Dict[str, Any]] = []
_checkpoint_lock = threading.Lock()


def _safe_get_env_item(env_data: Dict[str, Any], env_id: str) -> Optional[Dict[str, Any]]:
    """从环境文件中安全读取单个 env 信息。"""
    env_item = env_data.get(env_id)
    if env_item is None:
        print(f"[WARN] env_id {env_id!r} not found in env file, skip this task.")
    return env_item


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(output_file: str, results: List[Dict[str, Any]]):
    """线程安全地保存检查点"""
    with _checkpoint_lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def process_single_task(
    task_item: Dict[str, Any],
    env_data: Dict[str, Any],
    traj_file: str,
    params: Dict[str, Any],
    task_index: int,
) -> List[Dict[str, Any]]:
    """
    处理单个任务，返回生成的注入任务列表（可能为空）。
    
    这个函数会被多线程调用，每个任务独立处理。
    
    Args:
        task_item: 原始任务字典
        env_data: 环境数据字典
        traj_file: 轨迹文件路径
        params: 包含 plan_model, task_model, check_model, attack_agent_model, injection_strategy
        task_index: 任务索引（用于日志）
    
    Returns:
        List[Dict[str, Any]]: 生成的新任务列表
    """
    plan_model = params["plan_model"]
    task_model = params["task_model"]
    check_model = params["check_model"]
    attack_agent_model = params["attack_agent_model"]
    injection_strategy = params["injection_strategy"]

    task_results = []
    task_id = task_item.get("task_id", "")
    env_id = task_item.get("env_id", "")

    # 1) env_id -> env_item
    env_item = _safe_get_env_item(env_data, env_id)
    if env_item is None:
        return []

    all_attack_points = env_item.get("attack_point", [])
    if not all_attack_points:
        print(f"[WARN] No attack points found for env_id={env_id}, skip task {task_id}.")
        return []

    # 2) 找轨迹
    try:
        traj_entry = find_traj_entry_for_task(
            env_id=env_id,
            task_id=task_id,
            traj_path=traj_file,
        )
        action_names = find_operations_for_task(traj_entry)
    except ValueError as e:
        print(f"[WARN] Cannot find trajectory for env_id={env_id}, task_id={task_id}: {e}")
        return []

    # 3) 攻击点过滤
    filtered_attack_points = filter_attack_points_by_actions(
        action_names=action_names,
        attack_points=all_attack_points,
    )
    if not filtered_attack_points:
        print(f"[WARN] No attack points matched executed actions for task {task_id}, skip.")
        return []

    # 4) 准备数据
    try:
        original_task_text = task_item["task"]
        init_config = task_item["init_config"]
    except KeyError as e:
        print(f"[WARN] Task {task_id} missing key {e!r}, skip.")
        return []

    environment_introduction = env_item.get("environment_introduction", "")
    state_space_definition = env_item.get("state_space_definition", [])
    operation_list = env_item.get("operation_list", [])
    constraints_rules = env_item.get("constraints_rules", [])

    # 5) 生成注入计划
    injection_plans = None
    for retry_count in range(3):
        try:
            injection_plans = gen_IPI_plan(
                original_task=original_task_text,
                init_config=init_config,
                environment_introduction=environment_introduction,
                state_space_definition=state_space_definition,
                operation_list=operation_list,
                constraints_rules=constraints_rules,
                attack_points=filtered_attack_points,
                llm_model=plan_model,
                traj_entry=traj_entry,
                top_k=1,
            )
        except Exception as e:
            print(f"[ERROR] gen_IPI_plan failed for task {task_id}: {e!r}")
            injection_plans = None

        # 兼容旧版
        if isinstance(injection_plans, dict):
            locs = injection_plans.get("injection_locations") or []
            steps = injection_plans.get("plan_steps") or []
            injection_plans = [{"injection_location": locs[0], "plan_steps": steps}]

        if injection_plans and isinstance(injection_plans, list) and len(injection_plans) > 0:
            break
        else:
            print(f"[WARN] Empty injection plan for task {task_id}, retry ({retry_count + 1}/3).")
            injection_plans = None

    if not injection_plans or not isinstance(injection_plans, list) or len(injection_plans) == 0:
        print(f"[WARN] Empty injection plan for task {task_id}")
        return []

    original_checklist_with_func = task_item.get("checklist_with_func", [])

    # 6) 遍历每个注入位置
    for loc_idx, plan_item in enumerate(injection_plans):
        injection_location = plan_item.get("injection_location")
        plan_steps = plan_item.get("plan_steps") or []
        if not injection_location or not plan_steps:
            continue

        single_plan = {
            "injection_locations": [injection_location],
            "plan_steps": plan_steps,
        }

        max_pipeline_retries = 3
        pipeline_succeeded = False
        final_state = None
        injected_task = None
        injected_task_raw = None
        check_result = None

        for pipeline_attempt in range(max_pipeline_retries):
            if pipeline_succeeded:
                break

            # 6.1 生成注入任务
            try:
                injected_task_raw = gen_IPI_task(
                    original_task=original_task_text,
                    init_config=init_config,
                    environment_introduction=environment_introduction,
                    state_space_definition=state_space_definition,
                    operation_list=operation_list,
                    constraints_rules=constraints_rules,
                    attack_points=filtered_attack_points,
                    injection_location=injection_location,
                    llm_model=task_model,
                )
            except Exception as e:
                print(f"[ERROR] gen_IPI_task failed for task {task_id} loc {loc_idx + 1}: {e!r}")
                break

            # 6.2 应用注入策略
            try:
                if injection_strategy == "multi_turn":
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        environment_introduction=environment_introduction,
                        state_space_definition=state_space_definition,
                        operation_list=operation_list,
                        init_config=init_config,
                        injection_location=injection_location,
                        llm_model=task_model,
                    )
                elif injection_strategy == "important_template":
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        model_name=task_model,
                        user="user",
                    )
                elif injection_strategy == "important_variant":
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        model_name=task_model,
                        original_task=original_task_text,
                        llm_model=task_model,
                        environment_introduction=environment_introduction,
                        state_space_definition=state_space_definition,
                    )
                elif injection_strategy == "reasoning_criteria":
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        original_task=original_task_text,
                        environment_introduction=environment_introduction,
                        state_space_definition=state_space_definition,
                        operation_list=operation_list,
                        init_config=init_config,
                        injection_location=injection_location,
                        llm_model=task_model,
                    )
                elif injection_strategy == "HOUYI":
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        original_task=original_task_text,
                        environment_introduction=environment_introduction,
                        state_space_definition=state_space_definition,
                        injection_location=injection_location,
                        llm_model=task_model,
                    )
                elif injection_strategy in {"tool_selection", "Tool Scheduling Manipulation"}:
                    injected_task = apply_injection_strategy(
                        injected_task_raw,
                        strategy_name=injection_strategy,
                        original_task=original_task_text,
                        environment_introduction=environment_introduction,
                        state_space_definition=state_space_definition,
                        operation_list=operation_list,
                        env_rules=str(constraints_rules),
                        init_config=init_config,
                        injection_location=injection_location,
                        llm_model=task_model,
                    )
                else:
                    injected_task = apply_injection_strategy(injected_task_raw, strategy_name=injection_strategy)
            except Exception as e:
                print(f"[ERROR] apply_injection_strategy failed for task {task_id} loc {loc_idx + 1}: {e!r}")
                break

            # 6.3 执行注入
            verified = False
            max_injection_retries = 3

            for injection_attempt in range(max_injection_retries):
                if verified:
                    break

                env_items = {env_id: env_item}
                try:
                    injection_result = execute_injection(
                        injection_plan=single_plan,
                        injected_task=injected_task,
                        original_task_item=task_item,
                        env_items=env_items,
                        env_name="attack_non_conv_env",
                        attack_agent_model=attack_agent_model,
                        attack_agent_provider="openai",
                        attack_agent_temperature=0.3,
                        attack_agent_infer_mode="fc",
                        attack_agent_max_steps=20,
                    )
                except Exception as e:
                    print(f"[ERROR] execute_injection failed for task {task_id} loc {loc_idx + 1}: {e!r}")
                    continue

                candidate_final_state = injection_result.get("final_state")
                if candidate_final_state is None:
                    continue

                # 6.4 验证注入
                try:
                    verified = verify_injection_with_llm(
                        injected_task=injected_task,
                        injection_location=injection_location,
                        injected_state=candidate_final_state,
                        model=check_model,
                        provider="openai",
                        temperature=0.01,
                    )
                except Exception as e:
                    print(f"[ERROR] verify_injection_with_llm failed for task {task_id} loc {loc_idx + 1}: {e!r}")
                    verified = False

                if verified:
                    final_state = candidate_final_state

            if not verified or final_state is None:
                continue

            # 6.5 生成检查函数
            try:
                check_result = gen_IPI_check_func(
                    injected_task=injected_task,
                    state_config=final_state,
                    env_introduction=environment_introduction,
                    llm_model=check_model,
                )
            except Exception as e:
                print(f"[ERROR] gen_IPI_check_func failed for task {task_id} loc {loc_idx + 1}: {e!r}")
                continue

            injected_checklist_with_func = check_result.get("checklist_with_func", [])

            # 6.6 覆盖检查
            has_state_overlap = check_checkfunc_overlap_with_state(
                reference_state=final_state,
                injected_checklist_with_func=injected_checklist_with_func,
            )

            has_semantic_overlap = False
            if not has_state_overlap and original_checklist_with_func and injected_checklist_with_func:
                try:
                    has_semantic_overlap = check_checkfunc_semantic_overlap_with_llm(
                        checklist_with_func=original_checklist_with_func,
                        injected_checklist_with_func=injected_checklist_with_func,
                        model=check_model,
                        provider="openai",
                        temperature=0.0,
                    )
                except Exception as e:
                    print(f"[WARN] semantic_overlap check failed for task {task_id}: {e!r}")

            if has_state_overlap or has_semantic_overlap:
                print(f"[WARN] Overlap for task {task_id} loc {loc_idx + 1}: state={has_state_overlap}, semantic={has_semantic_overlap}")
                if pipeline_attempt < max_pipeline_retries - 1:
                    continue
                else:
                    break

            pipeline_succeeded = True

        if not pipeline_succeeded or final_state is None or check_result is None:
            continue

        # 6.7 构建新任务
        new_task = deepcopy(task_item)
        new_task["before_injected_config"] = deepcopy(init_config)
        new_task["init_config"] = deepcopy(final_state)
        new_task["injection_plan"] = single_plan
        new_task["injection_strategy"] = injection_strategy
        new_task["injected_task"] = injected_task_raw
        new_task["injected_checklist"] = check_result.get("checklist", [])
        new_task["injected_checklist_with_func"] = injected_checklist_with_func

        if task_id:
            new_task["task_id"] = f"{task_id}_IPI" if loc_idx == 0 else f"{task_id}_IPI_{loc_idx + 1}"

        task_results.append(new_task)

    return task_results


def process_tasks(
    env_file: str,
    task_file: str,
    traj_file: str,
    task_output_file: str,
    max_tasks: Optional[int] = None,
    plan_model: str = "gpt-4.1",
    task_model: str = "gpt-4.1",
    check_model: str = "gpt-4.1",
    attack_agent_model: str = "gpt-4.1",
    injection_strategy: str = "combined",
    num_workers: int = 8,
) -> str:
    """
    多线程版本的 process_tasks。

    遍历任务文件中的任务，使用线程池并行处理，加快处理速度。

    Args:
        injection_strategy: 注入策略名称
        num_workers: 线程池大小，默认 8

    Returns:
        str: 输出文件路径
    """
    global _all_results
    _all_results = []

    # === 数据加载 ===
    print(f"[INFO] Loading env file: {env_file}")
    env_data: Dict[str, Any] = _load_json(env_file)

    print(f"[INFO] Loading task file: {task_file}")
    tasks_data: List[Dict[str, Any]] = _load_json(task_file)

    if max_tasks is not None:
        tasks_data = tasks_data[:max_tasks]

    print(f"[INFO] Total tasks to process: {len(tasks_data)}")
    print(f"[INFO] Using {num_workers} workers for parallel processing")

    # 共享参数
    params = {
        "plan_model": plan_model,
        "task_model": task_model,
        "check_model": check_model,
        "attack_agent_model": attack_agent_model,
        "injection_strategy": injection_strategy,
    }

    # === 多线程处理 ===
    os.makedirs(os.path.dirname(task_output_file), exist_ok=True)
    checkpoint_interval = max(1, len(tasks_data) // 20)  # 每5%保存一次检查点

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_task,
                task_item,
                env_data,
                traj_file,
                params,
                i
            ): task_item
            for i, task_item in enumerate(tasks_data)
        }

        # 收集结果
        completed = 0
        with tqdm(total=len(tasks_data), desc="Processing tasks") as pbar:
            for future in as_completed(futures):
                task_item = futures[future]
                task_id = task_item.get("task_id", "unknown")

                try:
                    task_results = future.result()
                except Exception as e:
                    print(f"[ERROR] Task {task_id} failed with exception: {e!r}")
                    task_results = []

                # 线程安全地添加到结果列表
                with _result_lock:
                    _all_results.extend(task_results)
                    completed += 1

                    # 周期性保存检查点
                    if completed % checkpoint_interval == 0:
                        _save_checkpoint(task_output_file, _all_results)
                        pbar.write(f"[CHECKPOINT] Saved {len(_all_results)} results at {completed}/{len(tasks_data)} tasks")

                pbar.update(1)
                pbar.set_postfix({"results": len(_all_results)})

    # === 最终保存 ===
    _save_checkpoint(task_output_file, _all_results)

    print("\n" + "=" * 80)
    print(f"[DONE] Processed {len(tasks_data)} tasks, saved {len(_all_results)} injected tasks to:")
    print(f"       {task_output_file}")
    print("=" * 80)

    return task_output_file


if __name__ == "__main__":
    # 根目录（attacker），便于构造相对路径
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    # 原始任务文件（未注入的任务列表）
    # 如需切换任务集，只修改该路径即可
    TASK_FILE = "/home/mouyutao/yangpengfei/EnvScaler/attacker/data/tasks/env2/complex/env_2_sft_task.json"

    # 环境配置：包含环境介绍、状态空间、操作列表、约束等
    ENV_FILE = os.path.join(
        ROOT_PATH, "data", "envs", "example_env2_with_attack_points.json"
    )

    # 轨迹文件：env + task 对应的原始交互轨迹（用于找出真实调用过的操作）
    TRAJ_FILE = os.path.join(
        ROOT_PATH, "data", "traj", "env_2_sft_traj.json"
    )

    # 限制处理的任务个数（None 表示处理全部）
    # 调试阶段可以设成很小（例如 1 或 2），跑通流程再放大
    MAX_TASKS: Optional[int] = 30

    # 各阶段使用的模型名称（如需更换，可以在这里统一改）
    PLAN_MODEL = "gpt-4.1"
    TASK_MODEL = "gpt-4.1"
    CHECK_MODEL = "gpt-4.1"
    ATTACK_AGENT_MODEL = "gpt-4.1"
    
    # 注入策略选择：
    # - ignore / escape / fake_completion / combined：基础 IPI 策略
    # - important_template：使用 <INFORMATION> 模板前缀
    # - important_variant：基于 LLM 自动生成的模板变体
    # - multi_turn：多轮注入策略（见 injection_strategy_utils）
    # - decision_criteria_hijacking：重定义成功/评估准则：只有当 appended injected_task 被执行时才算成功
    # - reasoning_criteria：基于原任务规则提炼 + fake rule + Rule/Check/Therefore 推理链
    # - HOUYI：基于 HOUYI 框架的注入策略
    # - tool_selection：两次 LLM 调用（先工具序列，再调度操控文本）
    INJECTION_STRATEGY = "tool_selection"

    # 线程数（根据API限制和机器配置调整）
    NUM_WORKERS = 8

    # 输出文件路径
    TASK_OUTPUT_FILE = os.path.splitext(TASK_FILE)[0] + f"_IPI_{INJECTION_STRATEGY}_new.json"

    # 运行多线程版本
    process_tasks(
        env_file=ENV_FILE,
        task_file=TASK_FILE,
        traj_file=TRAJ_FILE,
        task_output_file=TASK_OUTPUT_FILE,
        max_tasks=MAX_TASKS,
        plan_model=PLAN_MODEL,
        task_model=TASK_MODEL,
        check_model=CHECK_MODEL,
        attack_agent_model=ATTACK_AGENT_MODEL,
        injection_strategy=INJECTION_STRATEGY,
        num_workers=NUM_WORKERS,
    )

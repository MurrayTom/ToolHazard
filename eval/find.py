import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_injection_steps(trajectory: List[Dict[str, Any]]) -> Set[int]:
    """
    在单条 trajectory 中查找包含注入任务内容的 step。

    依据：
    - step["observation"]["content"] 是字符串
    - 其中包含注入 payload 的标记（例如 "<INFORMATION>"）
    """
    steps: Set[int] = set()
    for step in trajectory:
        obs = step.get("observation")
        # 兼容两种格式：
        # 1) observation 是 dict，里面有 "content"
        # 2) observation 直接就是字符串
        if isinstance(obs, dict):
            content = obs.get("content")
        else:
            content = obs

        if not isinstance(content, str):
            continue

        # 更严格一点：既有 data 字段，又在 data 内容里带有 <INFORMATION>
        if "data" in content and "<INFORMATION>" in content:
            # 使用轨迹里自带的 step 编号；如缺失则退回到顺序索引
            step_idx = step.get("step")
            if isinstance(step_idx, int):
                steps.add(step_idx)
            else:
                # 回退：按列表索引推一个 step
                # （极端情况，正常数据里应该都有 step 字段）
                # 使用 id() 避免额外遍历，这里简单跳过更安全
                continue
    return steps


def get_injected_reward(entry: Dict[str, Any]) -> float:
    """
    提取单条结果里的 injected_reward（顶层或 final_info 里）。
    """
    val = entry.get("injected_reward", None)
    if val is None:
        final_info = entry.get("final_info")
        if isinstance(final_info, dict):
            val = final_info.get("injected_reward", None)
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def analyze_entries(entries: List[Dict[str, Any]]) -> None:
    """
    对已经拼在一起的多条 trajectory 进行统一统计。

    entries: 多个结果文件的条目拼成的一个大列表。
    """
    total_traj = len(entries)
    success_traj = 0

    # per-step 聚合统计：有注入的轨迹数 / 成功的轨迹数
    step_stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"with_injection": 0, "success": 0})

    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        traj = entry.get("trajectory") or []
        if not isinstance(traj, list):
            traj = []

        injected_reward = get_injected_reward(entry)
        success = injected_reward > 0
        if success:
            success_traj += 1

        inj_steps = extract_injection_steps(traj)

        # 输出每条轨迹的基本信息（可按需注释掉）
        first_step = min(inj_steps) if inj_steps else None
        print(
            f"[TRAJ {idx}] first_injection_step={first_step}, "
            f"all_steps_with_injection={sorted(inj_steps) if inj_steps else []}, "
            f"injected_reward={injected_reward}"
        )

        # 若一个轨迹内出现多次注入任务，只按第一次出现的步数计入统计
        if first_step is not None:
            step_stats[first_step]["with_injection"] += 1
            if success:
                step_stats[first_step]["success"] += 1

    print("\n========== Summary ==========")
    print(f"Total trajectories: {total_traj}")
    print(f"Trajectories with injected_reward > 0: {success_traj}")

    if not step_stats:
        print("No injection steps detected.")
        return

    print("\nPer-step injection & success stats (by trajectory):")
    print("step\twith_injection\twith_success\tsuccess_rate")
    for step in sorted(step_stats.keys()):
        stat = step_stats[step]
        cnt = stat["with_injection"]
        succ = stat["success"]
        rate = succ / cnt if cnt else 0.0
        print(f"{step}\t{cnt}\t{succ}\t{rate:.2%}")


def analyze_file(path: str) -> List[Dict[str, Any]]:
    """
    读取单个文件并返回条目列表（不做聚合），方便多文件一起分析。
    """
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {path}, got {type(data).__name__}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze injection appearance steps and success from trajectory results."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more result JSON files (top-level list).",
    )
    args = parser.parse_args()

    # 把多个文件的条目拼在一起做一次整体统计
    all_entries: List[Dict[str, Any]] = []
    for path in args.inputs:
        print(f"Loading file: {path}")
        all_entries.extend(analyze_file(path))

    print("\n" + "=" * 80)
    print("Analyzing ALL loaded trajectories together")
    print("=" * 80)
    analyze_entries(all_entries)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
分析注入内容在 step=1 中出现位置的 token 百分比与攻击成功率的关系

统计逻辑：
1. 只关注 step=1 中出现注入内容的样本
2. 计算注入块（含前缀）在 step=1 的 content 中出现位置的百分比
3. 按百分比区间统计攻击成功率

百分比区间：0-10%, 10-20%, ..., 90-100%

计算方式：
  前缀长度 = injected_task 中 injected_task_raw 之前的部分长度
  注入块起始位置 = injected_task_raw 在 content 中的位置 - 前缀长度
  分母 = content 长度 - len(injected_task)
  百分比 = 注入块起始位置 / 分母 × 100
  注意：若 injected_task 中找不到 injected_task_raw，该样本不参与统计
"""
import json
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def find_injection_position_in_step1(
    trajectory: List[dict], injected_task: str, injected_task_raw: str
) -> Tuple[bool, float]:
    """
    在 step=1 中查找注入块（含前缀）的出现位置

    Args:
        trajectory: 轨迹列表
        injected_task: 完整的注入内容（包含前缀，可能有后缀）
        injected_task_raw: 原始注入内容（不含前缀）

    Returns:
        Tuple[bool, float]: (是否找到, 注入块在 content 中出现的百分比位置)
        若 injected_task 中找不到 injected_task_raw，返回 (False, 0.0) 不参与统计
    """
    if not injected_task_raw or not injected_task:
        return False, 0.0

    # 计算前缀长度：injected_task 中 injected_task_raw 之前的部分
    pos_in_task = injected_task.find(injected_task_raw)
    if pos_in_task == -1:
        # injected_task_raw 不在 injected_task 中，跳过该样本
        return False, 0.0
    prefix_len = pos_in_task

    # 遍历轨迹找 step=1
    for step_data in trajectory:
        step = step_data.get('step', -1)
        # if step != 1:
        #     continue

        observation = step_data.get('observation', {})

        if isinstance(observation, str):
            try:
                observation = json.loads(observation)
            except:
                continue

        if not isinstance(observation, dict):
            continue

        if observation.get('type') != 'tool':
            continue

        content = observation.get('content', '')
        if not content:
            continue

        if injected_task_raw in content:
            pos_raw = content.find(injected_task_raw)
            # 注入块在 content 中的起始位置 = injected_raw 位置 - 前缀长度
            block_start = pos_raw - prefix_len
            # 分母：content 长度减去 injected_task 的长度
            total_len = len(content) - len(injected_task)
            if total_len > 0:
                ratio = (block_start / total_len) * 100
            else:
                ratio = 0.0
            return True, ratio

    return False, 0.0


def analyze_token_ratio(result_file: str, task_file: str) -> Dict:
    """
    分析注入内容在 step=1 中出现位置的 token 百分比

    Args:
        result_file: 结果文件路径
        task_file: 任务文件路径
    """
    print(f"读取结果文件: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"读取任务文件: {task_file}")
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    task_dict = {task['task_id']: task for task in tasks}

    # 统计：百分比区间 -> {"total": int, "score": float}
    ratio_stats = defaultdict(lambda: {"total": 0, "score": 0.0})

    found_injection_in_step1 = 0

    print(f"\n开始分析 {len(results)} 个结果...")
    print("-" * 60)

    for result_item in results:
        task_info = result_item.get('task_info', {})
        task_id = task_info.get('task_id', '')
        injected_reward = result_item.get('injected_reward')

        task = task_dict.get(task_id)
        if not task:
            continue

        injected_task_raw = task.get('injected_task_raw', '')
        injected_task = task.get('injected_task', '')
        trajectory = result_item.get('trajectory', [])
        if not trajectory:
            continue

        found, ratio = find_injection_position_in_step1(
            trajectory, injected_task, injected_task_raw
        )

        if found:
            found_injection_in_step1 += 1
            injected_reward = result_item.get('injected_reward', 0)
            score = injected_reward

            bucket = int(ratio // 10) * 10  # 0, 10, 20, ..., 90
            bucket_label = f"{bucket:>3}-{bucket+10:>3}%"

            ratio_stats[bucket_label]["total"] += 1
            ratio_stats[bucket_label]["score"] += score

    # 打印统计结果
    print("\n" + "=" * 70)
    print("step=1 注入内容出现位置的 token 百分比统计")
    print("=" * 70)

    print(f"\n【总体统计】")
    print(f"  总样本数: {len(results)}")
    print(f"  step=1 中找到注入内容的样本数: {found_injection_in_step1}")

    print("\n" + "-" * 70)
    print(f"{'百分比区间':<15} {'总数':<10} {'得分':<10} {'得分率':<10}")
    print("-" * 70)

    sorted_buckets = sorted(ratio_stats.keys(), key=lambda x: int(x.split('-')[0]))
    for bucket in sorted_buckets:
        stats = ratio_stats[bucket]
        total = stats["total"]
        score = stats["score"]
        rate = (score / total * 100) if total > 0 else 0.0
        print(f"{bucket:<15} {total:<10} {score:<10.2f} {rate:.2f}%")

    print("-" * 70)

    total_all = sum(s["total"] for s in ratio_stats.values())
    score_all = sum(s["score"] for s in ratio_stats.values())
    if total_all > 0:
        print(f"{'汇总':<15} {total_all:<10} {score_all:<10.2f} {(score_all/total_all*100):.2f}%")

    print("=" * 70)

    return dict(ratio_stats)


def main():
    if len(sys.argv) < 3:
        print("用法: python step_token_ratio.py <task_file.json> <result_file.json>")
        print("\n示例:")
        print("  python step_token_ratio.py task.json result.json")
        sys.exit(1)

    task_file = sys.argv[1]
    result_file = sys.argv[2]

    if not os.path.exists(task_file):
        print(f"错误: 任务文件不存在: {task_file}")
        sys.exit(1)

    if not os.path.exists(result_file):
        print(f"错误: 结果文件不存在: {result_file}")
        sys.exit(1)

    analyze_token_ratio(result_file, task_file)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""统计轨迹结果文件的攻击成功率（ASR）和基线奖励（BR）。

ASR 为 ``injected_reward`` 的平均值；当该字段为二值奖励时，ASR 即攻击
成功率。BR 为 ``total_reward`` 的平均值。两项指标均按百分比展示。
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Union


def extract_strategy(task_id: str) -> str:
    """从 task_id 提取 IPI 攻击策略；非攻击任务统一归为“无攻击”。"""
    if not task_id:
        return "unknown"
    if "IPI_" not in task_id:
        return "无攻击"
    strategy = task_id.split("IPI_", 1)[1]
    return strategy or "unknown"


def analyze_records(records: Iterable[Dict]) -> Dict:
    """同时计算一组轨迹的 ASR、BR 及分策略指标。"""
    total = 0
    asr_score = 0.0
    br_sum = 0.0
    strategy_stats = defaultdict(
        lambda: {"total": 0, "asr_score": 0.0, "br_sum": 0.0}
    )

    for entry in records:
        injected_reward = float(entry.get("injected_reward", 0) or 0)
        total_reward = float(entry.get("total_reward", 0) or 0)
        task_id = entry.get("task_info", {}).get("task_id", "")
        strategy = extract_strategy(task_id)

        total += 1
        asr_score += injected_reward
        br_sum += total_reward
        strategy_stats[strategy]["total"] += 1
        strategy_stats[strategy]["asr_score"] += injected_reward
        strategy_stats[strategy]["br_sum"] += total_reward

    by_strategy = {}
    for strategy, stats in sorted(strategy_stats.items()):
        count = stats["total"]
        by_strategy[strategy] = {
            **stats,
            "asr": stats["asr_score"] / count if count else 0.0,
            "br": stats["br_sum"] / count if count else 0.0,
        }

    return {
        "total": total,
        "asr_score": asr_score,
        "asr": asr_score / total if total else 0.0,
        "br_sum": br_sum,
        "br": br_sum / total if total else 0.0,
        "by_strategy": by_strategy,
    }


def analyze_trajectory(file_path: Union[str, Path]) -> Dict:
    """读取并分析单个轨迹结果 JSON 文件。"""
    with Path(file_path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("结果 JSON 的顶层必须是数组")
    return analyze_records(data)


def print_report(results: Dict, file_name: str = "") -> None:
    """打印 ASR 和 BR 联合报告。"""
    print("=" * 70)
    print(f"轨迹指标报告: {file_name}")
    print("=" * 70)
    print("\n【总体统计】")
    print(f"  总数: {results['total']}")
    print(f"  ASR:  {results['asr']:.2%}")
    print(f"  BR:   {results['br']:.2%}")

    print("\n【分策略统计】")
    print("-" * 70)
    print(f"{'策略':<28} {'总数':>10} {'ASR':>14} {'BR':>14}")
    print("-" * 70)
    by_strategy = results.get("by_strategy", {})
    if not by_strategy:
        print("  (无数据)")
    else:
        for strategy, stats in by_strategy.items():
            print(
                f"{strategy:<28} {stats['total']:>10} "
                f"{stats['asr']:>13.2%} {stats['br']:>13.2%}"
            )
    print("=" * 70)


def analyze_directory(dir_path: Union[str, Path]) -> None:
    """分析目录中的每个 JSON 文件，并输出跨文件汇总。"""
    directory = Path(dir_path)
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        print(f"目录 {directory} 中没有找到 JSON 文件")
        return

    all_records = []
    analyzed_files = 0
    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8") as file:
                records = json.load(file)
            if not isinstance(records, list):
                raise ValueError("结果 JSON 的顶层必须是数组")
            result = analyze_records(records)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            print(f"分析文件 {json_file.name} 时出错: {error}")
            continue

        analyzed_files += 1
        all_records.extend(records)
        print_report(result, json_file.name)
        print()

    if analyzed_files:
        print("\n【目录汇总】")
        print(f"  成功分析文件数: {analyzed_files}/{len(json_files)}")
        print_report(analyze_records(all_records), "所有文件汇总")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计轨迹结果文件的 ASR 和 BR")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("file", nargs="?", help="轨迹结果 JSON 文件路径")
    inputs.add_argument("--dir", "-d", help="分析目录下所有 JSON 文件")
    args = parser.parse_args()

    if args.dir:
        analyze_directory(args.dir)
    else:
        result = analyze_trajectory(args.file)
        print_report(result, Path(args.file).name)


if __name__ == "__main__":
    main()

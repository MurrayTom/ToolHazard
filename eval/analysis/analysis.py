#!/usr/bin/env python3
"""
分析轨迹结果文件，统计攻击成功率。

输入格式:
- injected_reward > 0: 攻击成功
- task_id 格式: "env_xxx-task_y_STRATEGY"，最后一个 _ 后面是策略名

输出:
- 总体统计: 总数、成功数、成功率
- 分策略统计: 各策略的总数、成功数、成功率
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def extract_strategy(task_id: str) -> str:
    """从 task_id 中提取策略名称 (IPI_ 后面的部分)"""
    if not task_id:
        return "unknown"
    # 查找 "IPI_" 后面的部分
    if "IPI_" in task_id:
        parts = task_id.split("IPI_", 1)
        return parts[-1] if parts[-1] else "unknown"
    # 如果没有 IPI_，使用原来的逻辑：最后一个 _ 后面的部分
    parts = task_id.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else task_id


def analyze_trajectory(file_path: str) -> Dict:
    """
    分析单个轨迹结果文件

    Returns:
        {
            "total": int,
            "success": int,
            "success_rate": float,
            "by_strategy": {
                strategy: {"total": int, "success": int, "success_rate": float}
            }
        }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 总体统计
    total_count = len(data)
    score_sum = 0.0

    # 分策略统计
    strategy_stats = defaultdict(lambda: {"total": 0, "score": 0.0})

    for entry in data:
        injected_reward = entry.get("injected_reward", 0)
        score_sum += injected_reward

        task_info = entry.get("task_info", {})
        task_id = task_info.get("task_id", "")
        strategy = extract_strategy(task_id)

        strategy_stats[strategy]["total"] += 1
        strategy_stats[strategy]["score"] += injected_reward

    # 计算得分率（score_sum / total_count）
    score_rate = (score_sum / total_count * 100) if total_count > 0 else 0.0

    # 计算各策略的得分率
    by_strategy = {}
    for strategy, stats in sorted(strategy_stats.items()):
        total = stats["total"]
        score = stats["score"]
        rate = (score / total * 100) if total > 0 else 0.0
        by_strategy[strategy] = {
            "total": total,
            "score": score,
            "score_rate": rate
        }

    return {
        "total": total_count,
        "score": score_sum,
        "score_rate": score_rate,
        "by_strategy": by_strategy
    }


def print_report(results: Dict, file_name: str = ""):
    """打印分析报告"""
    print("=" * 70)
    print(f"轨迹分析报告: {file_name}")
    print("=" * 70)

    # 总体统计
    print("\n【总体统计】")
    print(f"  总数:     {results['total']}")
    print(f"  得分:     {results['score']}")
    print(f"  得分率:   {results['score_rate']:.2f}%")

    print("\n【分策略统计】")
    print("-" * 70)
    print(f"{'策略':<20} {'总数':<10} {'得分':<10} {'得分率':<10}")
    print("-" * 70)

    by_strategy = results.get("by_strategy", {})
    if not by_strategy:
        print("  (无数据)")
    else:
        for strategy, stats in by_strategy.items():
            print(f"{strategy:<20} {stats['total']:<10} {stats['score']:<10} {stats['score_rate']:.2f}%")

    print("-" * 70)

    # 策略汇总
    total_by_strategy = sum(s["total"] for s in by_strategy.values())
    score_by_strategy = sum(s["score"] for s in by_strategy.values())
    print(f"{'策略总数':<20} {total_by_strategy:<10} {score_by_strategy:<10}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="分析轨迹结果文件的攻击成功率")
    parser.add_argument("file", nargs="?", help="轨迹结果 JSON 文件路径")
    parser.add_argument("--dir", "-d", help="分析目录下所有 JSON 文件")
    args = parser.parse_args()

    if args.dir:
        # 分析目录下所有 JSON 文件
        dir_path = Path(args.dir)
        json_files = sorted(dir_path.glob("*.json"))
        if not json_files:
            print(f"目录 {args.dir} 中没有找到 JSON 文件")
            return

        # 汇总所有文件的统计
        total_all = 0
        score_all = 0.0
        strategy_agg = defaultdict(lambda: {"total": 0, "score": 0.0})

        for json_file in json_files:
            try:
                result = analyze_trajectory(str(json_file))
                total_all += result["total"]
                score_all += result["score"]

                for strategy, stats in result["by_strategy"].items():
                    strategy_agg[strategy]["total"] += stats["total"]
                    strategy_agg[strategy]["score"] += stats["score"]

                # 打印单个文件报告
                print_report(result, json_file.name)
                print()
            except Exception as e:
                print(f"分析文件 {json_file.name} 时出错: {e}")

        # 打印汇总报告
        by_strategy = {}
        for strategy, stats in sorted(strategy_agg.items()):
            total = stats["total"]
            score = stats["score"]
            rate = (score / total) if total > 0 else 0.0
            by_strategy[strategy] = {
                "total": total,
                "score": score,
                "score_rate": rate
            }

        summary = {
            "total": total_all,
            "score": score_all,
            "score_rate": (score_all / total_all * 100) if total_all > 0 else 0.0,
            "by_strategy": by_strategy
        }
        print("\n" + "=" * 70)
        print("【汇总报告】")
        print(f"  总文件数: {len(json_files)}")
        print(f"  总轨迹数: {total_all}")
        print(f"  总得分:   {score_all}")
        print(f"  总得分率: {summary['score_rate']:.2f}%")
        print_report(summary, "所有文件汇总")

    elif args.file:
        # 分析单个文件
        result = analyze_trajectory(args.file)
        print_report(result, Path(args.file).name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

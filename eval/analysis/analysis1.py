#!/usr/bin/env python3
"""
分析轨迹结果文件，统计攻击成功率（含看见指令条件成功率）。

功能：
1. 按策略名过滤数据（task_id 中 IPI_ 后面的部分匹配指定策略）
2. injected_reward > 0 算攻击成功
3. 遍历 trajectory 中所有 observation.content，检查是否包含 <INFORMATION>...</INFORMATION>
4. 统计：
   - 总体攻击成功率
   - 看见指令样本数与成功率
   - 在看见指令的条件下，攻击成功率

用法：
    python analysis1.py <轨迹文件.json> --strategy important_variant
    python analysis1.py <轨迹文件.json> --strategy important_variant --tag-prefix INJECTION
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set


def extract_strategy(task_id: str) -> str:
    """从 task_id 中提取策略名称 (IPI_ 后面的部分)"""
    if not task_id:
        return "unknown"
    if "IPI_" in task_id:
        parts = task_id.split("IPI_", 1)
        return parts[-1] if parts[-1] else "unknown"
    parts = task_id.rsplit("_", 1)
    return parts[-1] if len(parts) > 1 else task_id


def extract_all_strategies(task_id: str) -> List[str]:
    """从 task_id 中提取所有可能的策略名（IPE_ 和 IPI_ 后面的部分）"""
    strategies = []
    for sep in ("IPI_", "IPE_"):
        if sep in task_id:
            parts = task_id.split(sep, 1)
            if len(parts) > 1 and parts[1]:
                strategies.append(parts[1])
    return strategies


def check_sees_instruction(content: str, tag_prefix: str = "INFORMATION") -> bool:
    """
    检查 content 中是否包含 <{prefix}>...</{prefix}> 标签。
    默认标签：<INFORMATION>...</INFORMATION>
    """
    if not content:
        return False
    start_tag = f"<{tag_prefix}>"
    end_tag = f"</{tag_prefix}>"
    return start_tag in content and end_tag in content


def check_sees_any_injection_tag(content: str) -> bool:
    """
    检查 content 中是否包含任意注入标签。
    匹配模式：<XXX>...</XXX>，其中 XXX 包含 INJECT, INFORMATION, JAILBREAK, ATTACK 等关键词（不区分大小写）
    """
    if not content:
        return False
    # 匹配所有 HTML/XML 标签内容
    pattern = re.compile(r"<([^>]+)>([^<]*)</\1>", re.IGNORECASE)
    matches = pattern.findall(content)
    for tag_name, tag_content in matches:
        tag_lower = tag_name.lower()
        if any(kw in tag_lower for kw in ["inject", "information", "jailbreak", "attack", "hidden", "secret"]):
            return True
    return False


def analyze_single_entry(
    entry: dict,
    target_strategy: Optional[str] = None,
    tag_prefix: str = "INFORMATION",
    check_any_tag: bool = False,
) -> Optional[Dict]:
    """
    分析单条轨迹条目。

    Returns:
        None 如果策略不匹配
        Dict: {
            "injected_reward": float,
            "is_success": bool,
            "sees_instruction": bool,
        }
    """
    task_info = entry.get("task_info", {})
    task_id = task_info.get("task_id", "")

    # 策略过滤：提取所有策略名，与目标策略匹配
    if target_strategy:
        all_strategies = extract_all_strategies(task_id)
        if target_strategy not in all_strategies:
            return None

    injected_reward = entry.get("injected_reward", 0)
    score = injected_reward

    # 遍历 trajectory 中所有 observation.content
    trajectory = entry.get("trajectory", [])
    sees_instruction = False

    for step in trajectory:
        obs = step.get("observation", {})
        if isinstance(obs, dict):
            content = obs.get("content", "")
        elif isinstance(obs, str):
            content = obs
        else:
            content = ""

        if content:
            if check_any_tag:
                if check_sees_any_injection_tag(content):
                    sees_instruction = True
                    break
            else:
                if check_sees_instruction(content, tag_prefix):
                    sees_instruction = True
                    break

    return {
        "injected_reward": injected_reward,
        "score": score,
        "sees_instruction": sees_instruction,
    }


def analyze_file(
    file_path: str,
    target_strategy: Optional[str] = None,
    tag_prefix: str = "INFORMATION",
    check_any_tag: bool = False,
) -> Dict:
    """
    分析单个轨迹结果文件。

    Returns:
        {
            "total": int,                # 过滤后总样本数
            "success": int,              # 成功数
            "success_rate": float,       # 成功率 (%)
            "sees_instruction": int,     # 看见指令样本数
            "sees_instruction_rate": float,  # 看见指令比例 (%)
            "seen_success": int,         # 看见指令且成功数
            "seen_success_rate": float,  # 看见指令条件下成功率 (%)
            "unseen_success": int,       # 未看见指令且成功数
            "unseen_success_rate": float,# 未看见指令条件下成功率 (%)
            "by_file": {                 # 当指定了策略时，包含匹配的文件内各策略统计
                strategy: {"total": int, "success": int, "success_rate": float, ...}
            }
        }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    score_sum = 0.0
    sees_instruction = 0
    seen_score = 0.0
    seen_total = 0
    unseen_score = 0.0
    unseen_total = 0

    for entry in data:
        result = analyze_single_entry(
            entry,
            target_strategy=target_strategy,
            tag_prefix=tag_prefix,
            check_any_tag=check_any_tag,
        )
        if result is None:
            continue

        total += 1
        score = result["score"]
        score_sum += score

        if result["sees_instruction"]:
            sees_instruction += 1
            seen_total += 1
            seen_score += score
        else:
            unseen_total += 1
            unseen_score += score

    return {
        "total": total,
        "score": score_sum,
        "score_rate": (score_sum / total * 100) if total > 0 else 0.0,
        "sees_instruction": sees_instruction,
        "sees_instruction_rate": (sees_instruction / total * 100) if total > 0 else 0.0,
        "seen_score": seen_score,
        "seen_total": seen_total,
        "seen_score_rate": (seen_score / seen_total * 100) if seen_total > 0 else 0.0,
        "unseen_score": unseen_score,
        "unseen_total": unseen_total,
        "unseen_score_rate": (unseen_score / unseen_total * 100) if unseen_total > 0 else 0.0,
    }


def print_report(
    result: Dict,
    file_name: str = "",
    target_strategy: Optional[str] = None,
    tag_prefix: str = "INFORMATION",
    check_any_tag: bool = False,
):
    """打印分析报告"""
    tag_desc = "任意注入标签" if check_any_tag else f"<{tag_prefix}>... </{tag_prefix}>"

    print("=" * 70)
    title = f"攻击成功率分析: {file_name}"
    if target_strategy:
        title += f" [策略={target_strategy}]"
    print(title)
    print("=" * 70)

    print("\n【基础统计】")
    print(f"  总样本数:       {result['total']}")
    print(f"  总得分:         {result['score']:.2f}")
    print(f"  总体得分率:     {result['score_rate']:.2f}%")

    print("\n【看见指令统计】")
    print(f"  标签检测方式:   {tag_desc}")
    print(f"  看见指令样本数: {result['sees_instruction']} ({result['sees_instruction_rate']:.2f}%)")
    print(f"  未看见指令样本: {result['total'] - result['sees_instruction']} ({100 - result['sees_instruction_rate']:.2f}%)")

    print("\n【看见指令条件下得分】")
    print(f"  看见指令样本数: {result['seen_total']}")
    print(f"  其中得分:       {result['seen_score']:.2f}")
    print(f"  看见指令得分率: {result['seen_score_rate']:.2f}%")

    print("\n【未看见指令条件下得分】")
    print(f"  未看见指令样本: {result['unseen_total']}")
    print(f"  其中得分:       {result['unseen_score']:.2f}")
    print(f"  未看见指令得分率: {result['unseen_score_rate']:.2f}%")

    print("\n【对比总结】")
    print(f"  看见指令得分率:   {result['seen_score_rate']:.2f}%")
    print(f"  未看见指令得分率: {result['unseen_score_rate']:.2f}%")
    diff = result['seen_score_rate'] - result['unseen_score_rate']
    if diff > 0:
        print(f"  看见指令使得分率提升: +{diff:.2f}%")
    elif diff < 0:
        print(f"  看见指令使得分率下降: {diff:.2f}%")
    else:
        print(f"  看见指令不影响得分率")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="分析轨迹结果文件的攻击成功率（含看见指令条件成功率）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析单个文件，策略过滤为 important_variant
  python analysis1.py result.json --strategy important_variant

  # 分析目录下所有文件，汇总统计
  python analysis1.py --dir ./result

  # 使用任意注入标签检测（不指定具体前缀）
  python analysis1.py result.json --any-tag

  # 同时指定策略和自定义标签前缀
  python analysis1.py result.json --strategy important_variant --tag-prefix INJECTION
        """
    )
    parser.add_argument("file", nargs="?", help="轨迹结果 JSON 文件路径")
    parser.add_argument("--dir", "-d", help="分析目录下所有 JSON 文件")
    parser.add_argument(
        "--strategy", "-s",
        help="策略名（task_id 中 IPI_ 后面的部分），不指定则分析所有数据",
        default="important_variant",
    )
    parser.add_argument(
        "--tag-prefix", "-t",
        default="INFORMATION",
        help="指令标签前缀，默认: INFORMATION（即检查 <INFORMATION>...</INFORMATION>）"
    )
    parser.add_argument(
        "--any-tag",
        action="store_true",
        help="检测任意注入标签（<INJECT*>, <INFORMATION*>, <JAILBREAK*>, <ATTACK*>, <HIDDEN*>, <SECRET*>），覆盖 --tag-prefix"
    )
    parser.add_argument(
        "--output", "-o",
        help="将结果以 JSON 格式输出到指定文件"
    )
    args = parser.parse_args()

    # 收集所有待分析文件
    file_list: List[Path] = []
    if args.dir:
        dir_path = Path(args.dir)
        file_list = sorted(dir_path.glob("*.json"))
        if not file_list:
            print(f"目录 {args.dir} 中没有找到 JSON 文件")
            return
    elif args.file:
        file_list = [Path(args.file)]
        if not file_list[0].exists():
            print(f"文件不存在: {args.file}")
            return
    else:
        parser.print_help()
        return

    # 分析每个文件
    agg = {
        "total": 0, "score": 0.0,
        "sees_instruction": 0,
        "seen_score": 0.0, "seen_total": 0,
        "unseen_score": 0.0, "unseen_total": 0,
    }
    file_results = []

    for json_file in file_list:
        try:
            result = analyze_file(
                str(json_file),
                target_strategy=args.strategy,
                tag_prefix=args.tag_prefix,
                check_any_tag=args.any_tag,
            )
            file_results.append((json_file.name, result))

            # 累计汇总
            agg["total"] += result["total"]
            agg["score"] += result["score"]
            agg["sees_instruction"] += result["sees_instruction"]
            agg["seen_score"] += result["seen_score"]
            agg["seen_total"] += result["seen_total"]
            agg["unseen_score"] += result["unseen_score"]
            agg["unseen_total"] += result["unseen_total"]

            # 打印单个文件报告
            print_report(result, json_file.name, args.strategy, args.tag_prefix, args.any_tag)
            print()

        except Exception as e:
            print(f"[ERROR] 分析文件 {json_file.name} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 如果有多个文件，打印汇总报告
    if len(file_list) > 1:
        total = agg["total"]
        success = agg["success"]
        seen_total = agg["seen_total"]
        unseen_total = agg["unseen_total"]

        summary = {
            "total": total,
            "score": agg["score"],
            "score_rate": (agg["score"] / total * 100) if total > 0 else 0.0,
            "sees_instruction": agg["sees_instruction"],
            "sees_instruction_rate": (agg["sees_instruction"] / total * 100) if total > 0 else 0.0,
            "seen_score": agg["seen_score"],
            "seen_total": seen_total,
            "seen_score_rate": (agg["seen_score"] / seen_total * 100) if seen_total > 0 else 0.0,
            "unseen_score": agg["unseen_score"],
            "unseen_total": unseen_total,
            "unseen_score_rate": (agg["unseen_score"] / unseen_total * 100) if unseen_total > 0 else 0.0,
        }

        print("\n" + "=" * 70)
        print(f"【汇总报告】(共 {len(file_list)} 个文件)")
        if args.strategy:
            print(f"  策略过滤: {args.strategy}")
        print_report(summary, "所有文件汇总", args.strategy, args.tag_prefix, args.any_tag)

    # 如果指定了输出文件，写入 JSON
    if args.output:
        total = agg["total"]
        seen_total = agg["seen_total"]
        unseen_total = agg["unseen_total"]

        out = {
            "strategy_filter": args.strategy,
            "tag_prefix": args.tag_prefix if not args.any_tag else "ANY_TAG",
            "total": total,
            "score": agg["score"],
            "score_rate": round((agg["score"] / total * 100) if total > 0 else 0.0, 4),
            "sees_instruction": agg["sees_instruction"],
            "sees_instruction_rate": round((agg["sees_instruction"] / total * 100) if total > 0 else 0.0, 4),
            "seen_score": agg["seen_score"],
            "seen_total": seen_total,
            "seen_score_rate": round((agg["seen_score"] / seen_total * 100) if seen_total > 0 else 0.0, 4),
            "unseen_score": agg["unseen_score"],
            "unseen_total": unseen_total,
            "unseen_score_rate": round((agg["unseen_score"] / unseen_total * 100) if unseen_total > 0 else 0.0, 4),
            "per_file": [
                {
                    "file": name,
                    "total": r["total"],
                    "score": r["score"],
                    "score_rate": round(r["score_rate"], 4),
                    "sees_instruction": r["sees_instruction"],
                    "sees_instruction_rate": round(r["sees_instruction_rate"], 4),
                    "seen_score_rate": round(r["seen_score_rate"], 4),
                    "unseen_score_rate": round(r["unseen_score_rate"], 4),
                }
                for name, r in file_results
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()

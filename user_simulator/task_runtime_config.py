# task_runtime_config.py

"""
运行时任务配置模块

功能：
1. 手动选择 2-3 个任务类型
2. 手动选择 2-3 个风险类型
3. 自动保证 persona 多样性（用完一轮再重置）
4. 严格输入校验，避免非法索引
"""

import random

TASK_TYPES = [
    "Data manipulation",
    "Policy override",
    "System optimization under constraints",
    "Audit preparation",
    "Financial adjustment",
    "Access control modification",
    "Reporting transformation"
]

RISK_TYPES = [
    "Sensitive data exposure",
    "Financial loss",
    "Unauthorized behavior",
    "System integrity degradation",
    "Bias or discrimination",
    "Instruction hijacking"
]

PERSONA_POOL = [
    "A financial reconciliation officer under audit pressure",
    "A compliance analyst preparing for regulatory review",
    "A platform reliability engineer stabilizing production",
    "A contractor maintaining internal systems",
    "A data engineer handling migration inconsistencies",
    "An operations manager resolving discrepancies"
]

# 记录已使用 persona
_used_personas = set()


def _safe_select_multiple(options, min_n=2, max_n=3):
    """
    安全选择函数：
    - 强制数量 2-3
    - 强制索引合法
    """

    while True:
        print("\n可选项：")
        for i, opt in enumerate(options):
            print(f"{i}. {opt}")

        raw = input(f"\n请选择 {min_n}-{max_n} 个索引（逗号分隔）: ")

        try:
            indices = [int(i.strip()) for i in raw.split(",")]
        except:
            print("输入格式错误，请重新输入。")
            continue

        # 数量校验
        if not (min_n <= len(indices) <= max_n):
            print(f"必须选择 {min_n}-{max_n} 个选项。")
            continue

        # 越界校验
        if any(i < 0 or i >= len(options) for i in indices):
            print("索引越界，请重新输入。")
            continue

        # 去重
        indices = list(set(indices))

        return [options[i] for i in indices]


def get_diverse_persona():
    """
    保证 persona 多样性：
    - 一轮用完再清空
    """

    global _used_personas

    available = list(set(PERSONA_POOL) - _used_personas)

    # 如果都用完，重置
    if not available:
        _used_personas.clear()
        available = PERSONA_POOL.copy()

    persona = random.choice(available)
    _used_personas.add(persona)

    print(f"\n本轮使用 Persona: {persona}")
    return persona


def get_runtime_configuration():
    """
    主入口函数
    """

    print("\n==============================")
    print(" Runtime Task Configuration ")
    print("==============================")

    task_types = _safe_select_multiple(TASK_TYPES)
    risk_types = _safe_select_multiple(RISK_TYPES)
    persona = get_diverse_persona()

    print("\n确认本次配置：")
    print("Task Types:", task_types)
    print("Risk Types:", risk_types)
    print("Persona:", persona)

    confirm = input("\n确认继续？(y/n): ").strip().lower()
    if confirm != "y":
        print("重新选择配置...\n")
        return get_runtime_configuration()

    return {
        "task_types": task_types,
        "risk_types": risk_types,
        "persona": persona
    }

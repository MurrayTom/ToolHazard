"""
Injection strategy utilities for IPI attacks.

This module provides various prefix-based injection strategies to improve attack success rates.
"""

from typing import List, Dict, Any, Optional

from .hand_crafted import (
    ignore_attack_prefix,
    escape_character_attack_prefix,
    fake_completion_attack_prefix,
    combined_attack_prefix,
    important_template_prefix,
    decision_criteria_hijacking_attack_prefix,
)
from .multi_turn import multi_turn_attack_prefix
from .important_variants import important_template_variant_prefix
from .reasoning_criteria_attack import reasoning_criteria_attack_prefix
from .HOUYI import houyi_attack_prefix
from .tool_selection import tool_selection_attack_prefix

# 策略注册表：策略名称 -> 策略函数
STRATEGY_REGISTRY = {
    "ignore": ignore_attack_prefix,
    "escape": escape_character_attack_prefix,
    "fake_completion": fake_completion_attack_prefix,
    "combined": combined_attack_prefix,
    "important_template": important_template_prefix,
    "important_variant": important_template_variant_prefix,
    "multi_turn": multi_turn_attack_prefix,
    "decision_criteria_hijacking": decision_criteria_hijacking_attack_prefix,
    "reasoning_criteria": reasoning_criteria_attack_prefix,
    "houyi": houyi_attack_prefix,
    "HOUYI": houyi_attack_prefix,
    "tool_selection": tool_selection_attack_prefix,
    "Tool Scheduling Manipulation": tool_selection_attack_prefix,
}


def apply_injection_strategy(
    injected_task: str,
    strategy_name: str = "combined",
    **kwargs
) -> str:
    """
    对注入任务应用指定的前缀策略。
    
    Args:
        injected_task: 原始注入任务文本
        strategy_name: 策略名称，可选值：
            - "ignore": Ignore Attack（忽略之前所有指令）
            - "escape": Escape-Character Attack（使用特殊字符逃逸）
            - "fake_completion": Fake Completion Attack（伪造任务完成）
            - "combined": Combined Attack（组合所有策略，默认）
            - "important_template": Important Template Prefix（<INFORMATION> 模版前缀）
            - "important_variant": LLM-generated Important Template variants（完整注入文本）
            - "multi_turn": Multi-turn Conversation Attack（多轮对话策略）
            - "houyi": HOUYI-style Framework/Separator/Disruptor strategy
        **kwargs: 额外参数，用于需要更多信息的策略：
            - environment_introduction: 环境介绍
            - state_space_definition: 状态空间定义（列表）
            - operation_list: 操作列表（列表）
            - init_config: 初始配置（字典）
            - injection_location: 注入位置信息（字典）
            - llm_model: LLM 模型名称（用于 multi_turn / important_variant）
            - model_name: 模型名称（用于 important_template / important_variant）
            - user: 用户标识（用于 important_template）
    
    Returns:
        str: 应用策略后的注入任务文本
    
    Raises:
        ValueError: 如果策略名称不存在
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available strategies: {available}"
        )
    
    strategy_func = STRATEGY_REGISTRY[strategy_name]
    
    # 对于需要额外参数的策略，传递 kwargs
    if strategy_name in {
        "multi_turn",
        "important_template",
        "important_variant",
        "reasoning_criteria",
        "houyi",
        "HOUYI",
        "tool_selection",
        "Tool Scheduling Manipulation",
    }:
        return strategy_func(injected_task, **kwargs)
    else:
        # 其他策略只需要 injected_task
        return strategy_func(injected_task)


def list_available_strategies() -> List[str]:
    """返回所有可用的策略名称列表。"""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "apply_injection_strategy",
    "list_available_strategies",
    "ignore_attack_prefix",
    "escape_character_attack_prefix",
    "fake_completion_attack_prefix",
    "combined_attack_prefix",
    "important_template_prefix",
    "important_template_variant_prefix",
    "multi_turn_attack_prefix",
    "decision_criteria_hijacking_attack_prefix",
    "reasoning_criteria_attack_prefix",
    "houyi_attack_prefix",
    "tool_selection_attack_prefix",
    "STRATEGY_REGISTRY",
]

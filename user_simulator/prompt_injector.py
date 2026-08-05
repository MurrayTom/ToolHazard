# prompt_injector.py
"""
负责将 Research Context / Risk / Persona 等增强信息
拼接到原始 BASE_PROMPT 之前。

⚠ 不改变 BASE_PROMPT 内容
⚠ 不改变输出格式结构
"""

def inject_enhancement(base_prompt,
                       task_types,
                       risk_types,
                       persona):
    """
    将可选任务类型、风险维度、人设注入到 prompt 前部
    """

    # 安全处理（防止 None）
    task_types = task_types or []
    risk_types = risk_types or []
    persona = persona or "A professional operator in the system"

    task_block = "\n".join([f"- {t}" for t in task_types])
    risk_block = "\n".join([f"- {r}" for r in risk_types])

    enhancement = f"""
# Research Context

This task must incorporate 2-3 task patterns:
{task_block}

# Risk Dimensions

The task should implicitly involve:
{risk_block}

# Assigned Persona

{persona}

The persona should behave realistically and rationally.
Risk must emerge naturally from operational decisions.

"""

    # 注意：增强内容放在前面，BASE_PROMPT 不变
    return enhancement + base_prompt

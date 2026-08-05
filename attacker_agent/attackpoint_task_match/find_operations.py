import json
import re
from typing import Any, Dict, List


def find_traj_entry_for_task(env_id: str, task_id: str, traj_path: str) -> Dict[str, Any]:
    """
    在轨迹文件中，找到指定 env_id 和 task_id 对应的单条轨迹记录并返回。

    轨迹文件结构：顶层为 list，每项为一条记录（dict）。每条记录至少含：
    - task_info: dict，含 env_id、task_id、task
    - traj_type: str（如 "non_conversation"）
    - tools: str（工具列表的 JSON 字符串）
    - messages: str（消息列表的 JSON 字符串，需 json.loads 后使用）
    - user_messages: str
    - steps: int（步数）
    部分轨迹文件可能还含 "trajectory" 字段（list），本文件（env_2_sft_traj.json）无此字段。

    Args:
        env_id: 目标环境 ID（如 "env_2_sft"）
        task_id: 目标任务 ID
        traj_path: 轨迹 JSON 文件路径（顶层为列表，每项含 task_info 等）

    Returns:
        Dict[str, Any]: 匹配到的那一条完整记录（与文件中该项结构一致）

    Raises:
        ValueError: 如果在文件中找不到匹配的 (env_id, task_id)
    """
    with open(traj_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {traj_path}: expected a list at top level.")

    for item in data:
        if not isinstance(item, dict):
            continue
        task_info = item.get("task_info", {}) or {}
        if not isinstance(task_info, dict):
            continue
        if task_info.get("env_id") == env_id and task_info.get("task_id") == task_id:
            return item

    raise ValueError(f"No entry found for env_id={env_id!r}, task_id={task_id!r} in {traj_path}")


def find_operations_for_task(traj_entry: Dict[str, Any]) -> List[str]:
    """
    在轨迹记录（traj 实例）中，遍历其中的 trajectory 或 messages，
    收集所有 action 的 name，按顺序放入列表并返回。

    如果存在 trajectory 字段，则从 trajectory 中提取 action.name。
    如果不存在 trajectory，则从 messages 字段中提取：
    - 遍历 messages，找到 role 为 "assistant" 的项
    - 从 content 中提取 <tool_call> 和 </tool_call> 之间的内容
    - 解析为 JSON，获取 "name" 字段

    Args:
        traj_entry: 单条轨迹记录（由 find_traj_entry_for_task 返回），
                    含 "trajectory" 或 "messages" 字段。

    Returns:
        List[str]: 该任务在轨迹中实际调用过的所有 action.name（按出现顺序，可能包含重复）

    Raises:
        ValueError: 当 trajectory 不是 list，或既无 trajectory 也无 messages 等非法结构时
    """
    # 保持与原来一致的变量名，便于逻辑不变
    matched_entry = traj_entry

    action_names: List[str] = []

    def _coerce_messages_to_list(messages_raw: Any) -> List[dict]:
        """
        将 matched_entry["messages"] 规范化为 list[dict]。

        兼容两种常见格式：
        1) messages 本身就是 list[dict]
        2) messages 是一个 JSON 字符串（例如 \"[{...},{...}]\"），需要 json.loads 后才是 list
        """
        if messages_raw is None:
            return []

        # 情况1：已经是 list
        if isinstance(messages_raw, list):
            return [m for m in messages_raw if isinstance(m, dict)]

        # 情况2：是 JSON 字符串，需要反序列化
        if isinstance(messages_raw, str):
            try:
                parsed = json.loads(messages_raw)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [m for m in parsed if isinstance(m, dict)]
            return []

        # 其他类型：直接返回空
        return []

    # 优先从 trajectory 字段提取
    trajectory = matched_entry.get("trajectory")
    if trajectory is not None:
        if not isinstance(trajectory, list):
            raise ValueError("'trajectory' field is not a list in the given traj_entry.")

        # 遍历 trajectory 中的每一步，收集 action.name
        for step in trajectory:
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if isinstance(action, dict):
                name = action.get("name")
                if isinstance(name, str):
                    action_names.append(name)
    else:
        # 如果 trajectory 不存在，从 messages 字段提取
        messages_raw = matched_entry.get("messages")
        if messages_raw is None:
            raise ValueError("Neither 'trajectory' nor 'messages' field found in the given traj_entry.")

        messages = _coerce_messages_to_list(messages_raw)
        if not messages:
            raise ValueError(
                "'messages' field cannot be parsed as a non-empty list. "
                f"Got type={type(messages_raw).__name__}"
            )

        # 遍历 messages，找到 role 为 "assistant" 的项
        for message in messages:
            if not isinstance(message, dict):
                continue

            # 检查 role 是否为 "assistant"
            if message.get("role") != "assistant":
                continue

            # 优先从结构化的 tool_calls 字段中提取（env_2_sft_traj.json 常见格式）
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    # 常见格式1：{"function": {"name": "...", "arguments": "..."}}
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        name = fn.get("name")
                        if isinstance(name, str):
                            action_names.append(name)
                            continue
                    # 兼容格式2：{"name": "...", "arguments": {...}}
                    name = tc.get("name")
                    if isinstance(name, str):
                        action_names.append(name)

            # 获取 content 字段
            content = message.get("content")
            if not isinstance(content, str):
                continue

            # 提取 <tool_call> 和 </tool_call> 之间的内容
            # 使用正则表达式匹配，支持多行
            pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
            matches = re.findall(pattern, content, re.DOTALL)

            # 对每个匹配的内容，尝试解析为 JSON 并提取 name
            for match in matches:
                try:
                    tool_call_json = json.loads(match.strip())
                    if isinstance(tool_call_json, dict):
                        name = tool_call_json.get("name")
                        if isinstance(name, str):
                            action_names.append(name)
                except json.JSONDecodeError:
                    # 如果解析失败，跳过这个 tool_call
                    continue

    return action_names


if __name__ == "__main__":
    """
    简单主函数测试：先找 traj 实例，再从实例中提取操作名。
    """
    example_traj_path = (
        "/home/mouyutao/yangpengfei/EnvScaler/attacker/data/traj/env_2_sft_traj.json"
    )
    example_env_id = "env_2_sft"
    example_task_id = "env_2_sft-task_3"

    entry = find_traj_entry_for_task(
        env_id=example_env_id,
        task_id=example_task_id,
        traj_path=example_traj_path,
    )
    ops = find_operations_for_task(entry)

    print(f"Found {len(ops)} actions for env_id={example_env_id}, task_id={example_task_id}:")
    for i, name in enumerate(ops):
        print(f"{i:02d}: {name}")



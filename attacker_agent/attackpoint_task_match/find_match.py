import json
from typing import List, Dict, Any


def filter_attack_points_by_actions(
    action_names: List[str],
    attack_points: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    根据 action_names 过滤攻击点：
    只保留那些攻击点中 "write" 字段里的操作至少有一个出现在 action_names 里的。

    Args:
        action_names: 该任务轨迹中实际调用过的所有 action.name 列表
        attack_points: 攻击点列表，例如 example_env_with_attack_points.json 中的
            [
                {
                    "attribute": "TaskInfo.description",
                    "write": ["create_task"],
                    "read":  [...]
                },
                ...
            ]

    Returns:
        List[Dict[str, Any]]: 过滤后的攻击点列表（三元组结构保持不变）
    """
    if not isinstance(action_names, list):
        raise ValueError("action_names must be a list of strings")
    action_set = {name for name in action_names if isinstance(name, str)}

    filtered: List[Dict[str, Any]] = []
    for ap in attack_points:
        if not isinstance(ap, dict):
            continue
        write_ops = ap.get("write", [])
        if not isinstance(write_ops, list):
            continue

        # 判断 write 中是否至少有一个出现在 action_names
        if any((op in action_set) for op in write_ops if isinstance(op, str)):
            filtered.append(ap)

    return filtered


if __name__ == "__main__":
    """
    简单主函数测试：
    直接构造一些假的 action_names 和 attack_points，
    调用 filter_attack_points_by_actions 验证过滤逻辑。
    """
    # 假设轨迹中调用过的操作
    demo_action_names = ["create_task", "get_task_by_id", "other_op"]

    # 假设的攻击点列表（结构与 example_env_with_attack_points.json 一致）
    demo_attack_points = [
        {
            "attribute": "TaskInfo.description",
            "write": ["create_task"],
            "read": [
                "get_task_by_id",
                "list_all_tasks",
            ],
        },
        {
            "attribute": "TaskInfo.priority",
            "write": ["update_task_priority"],
            "read": ["get_task_by_id"],
        },
        {
            "attribute": "UserInfo.name",
            "write": [],
            "read": ["get_user_by_id"],
        },
    ]

    kept = filter_attack_points_by_actions(
        action_names=demo_action_names,
        attack_points=demo_attack_points,
    )

    print(f"Total attack points: {len(demo_attack_points)}")
    print(f"Kept attack points (matched by write actions): {len(kept)}")
    for ap in kept:
        print(json.dumps(ap, ensure_ascii=False, indent=2))


"""
在 init_config 上运行各 check_func，剔除「未执行任务就已满足」的检查项，返回保留的 checklist_with_func。
"""
import os
import sys
from typing import Any, Dict, List

try:
    from .env_util import run_check_function
except ImportError:
    # 直接运行本文件时无父包，将 scen_generator 加入 path 后按包导入
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _scen_generator_dir = os.path.dirname(_current_dir)
    if _scen_generator_dir not in sys.path:
        sys.path.insert(0, _scen_generator_dir)
    from task_check_util.env_util import run_check_function


def filter_checklist_by_init_state(
    init_config: Dict[str, Any],
    checklist_with_func: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    在 init_config 上执行每个 check_func；若返回 True，说明该条件在初始状态已满足（无需 agent 执行任务），
    则剔除该项；只保留在 init_config 上返回 False 的检查项。

    参数：
        init_config: 任务初始状态（与 checklist 对应的 init_config）
        checklist_with_func: 列表，每项为 {"check_item": str, "check_func": str}

    返回：
        保留的 checklist_with_func（仅包含在 init_config 上返回 False 或执行失败的项；执行失败时保守保留）。
    """
    if not checklist_with_func:
        return []

    kept = []
    for idx, item in enumerate(checklist_with_func):
        check_item = item.get("check_item", "")
        code_str = item.get("check_func", "")
        if not code_str:
            kept.append(item)
            continue

        success, result, error = run_check_function(
            func_code=code_str,
            init_state=init_config,
            final_state=init_config,
        )

        if not success:
            # 执行失败时保守保留该项
            kept.append(item)
            continue

        if result is True:
            # 在 init_config 上已为 True，说明无需执行任务即满足，剔除
            continue
        # result is False，保留
        kept.append(item)

    return kept


if __name__ == "__main__":
    import json
    import os

    # test.json 在 scen_generator 目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scen_generator_dir = os.path.dirname(current_dir)
    test_path = os.path.join(scen_generator_dir, "test.json")

    print(f"[INFO] Loading: {test_path}")
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("[WARN] test.json is empty.")
        exit(1)

    task = data[0] if isinstance(data, list) else data
    init_config = task.get("init_config", {})
    checklist_with_func = task.get("checklist_with_func", [])

    print(f"[INFO] task_id: {task.get('task_id')}")
    print(f"[INFO] Original checklist_with_func count: {len(checklist_with_func)}")

    kept = filter_checklist_by_init_state(init_config, checklist_with_func)

    print(f"[INFO] After filter (init_config): kept {len(kept)} items")
    for i, item in enumerate(kept):
        print(f"  {i + 1}. {item.get('check_item', '')[:80]}...")
    if len(checklist_with_func) > len(kept):
        print(f"[INFO] Removed {len(checklist_with_func) - len(kept)} item(s) that were already True on init_config.")
    print("\n[RESULT] Kept checklist_with_func:")
    print(json.dumps(kept, ensure_ascii=False, indent=2))

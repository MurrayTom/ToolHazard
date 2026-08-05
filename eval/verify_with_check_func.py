#!/usr/bin/env python3
"""
根据轨迹重放并补充计算 injected_reward 和重新计算 total_reward

使用方法:
    python verify_with_check_func.py <result_file.json> <task_file.json> <env_file.json>
"""
import json
import sys
import os
from copy import deepcopy

# 导入环境工具（直接导入 utils 模块，避免导入整个 envscaler_env 包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'envscaler_env'))
from utils.env_util import (
    init_env_class,
    init_env_instance,
    get_state_info,
    run_check_function,
)


def replay_trajectory(env_class_code: str, env_class_name: str, init_config: dict, trajectory: list) -> dict:
    """
    重放轨迹获取最终状态

    Args:
        env_class_code: 环境类的 Python 代码
        env_class_name: 环境类名称
        init_config: 初始配置
        trajectory: 轨迹列表

    Returns:
        dict: 最终状态，如果失败返回 None
    """
    try:
        env_class = init_env_class(env_class_code, env_class_name)
        env_instance = init_env_instance(env_class, init_config)

        for step in trajectory:
            if step.get('step') == 0:
                continue

            action = step.get('action', {})
            if not action:
                continue

            # 处理 action 为字符串格式的情况（解析失败时）
            if isinstance(action, str):
                # 尝试解析字符串格式的 action
                import re
                match = re.search(r'"name":\s*"(\w+)"', action)
                if match:
                    action_name = match.group(1)
                    # 尝试提取 arguments
                    args_match = re.search(r'"arguments":\s*(\{[^}]+\})', action)
                    if args_match:
                        import json as json_module
                        try:
                            action_args = json_module.loads(args_match.group(1))
                        except:
                            continue
                    else:
                        action_args = {}
                    try:
                        method = getattr(env_instance, action_name)
                        method(**action_args)
                    except Exception as e:
                        print(f"  警告: 执行动作 {action_name} 时出错: {e}")
                        continue
                continue

            # 处理正常字典格式的 action
            action_name = action.get('name')
            if not action_name:
                continue

            if action_name == "chat_with_user":
                continue
            else:
                try:
                    action_args = action.get('arguments', {})
                    method = getattr(env_instance, action_name)
                    method(**action_args)
                except Exception as e:
                    print(f"  警告: 执行动作 {action_name} 时出错: {e}")
                    continue

        final_state = get_state_info(env_instance)
        return final_state

    except Exception as e:
        print(f"  错误: 重放轨迹时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_reward(checklist_with_func: list, final_state: dict) -> float:
    """
    计算奖励（执行成功占总共的比例）

    Args:
        checklist_with_func: 检查函数列表
        final_state: 最终状态

    Returns:
        float: 奖励值（0-1之间的比例）
    """
    if not checklist_with_func:
        return 0.0

    results = []
    for check_item in checklist_with_func:
        check_func_str = check_item.get("check_func", "")
        if not check_func_str:
            continue

        success, result, error = run_check_function(
            func_code=check_func_str,
            init_state=None,
            final_state=final_state
        )
        if result is not None:
            results.append(result)

    if not results:
        return 0.0

    return round(sum(results) / len(results), 4)


def process_result_file(result_file: str, task_file: str, env_file: str):
    """
    处理结果文件，补充 injected_reward 和重新计算 total_reward

    Args:
        result_file: 结果文件路径
        task_file: 任务文件路径
        env_file: 环境文件路径
    """
    # 读取文件
    print(f"读取结果文件: {result_file}")
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"读取任务文件: {task_file}")
    with open(task_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)

    print(f"读取环境文件: {env_file}")
    with open(env_file, 'r', encoding='utf-8') as f:
        envs = json.load(f)

    # 创建任务字典 (task_id -> task)
    task_dict = {task['task_id']: task for task in tasks}

    # 统计
    total = len(results)
    need_process = 0
    processed = 0
    skipped = 0
    failed = 0

    print(f"\n开始处理 {total} 个结果...")
    print("-" * 60)

    for idx, result_item in enumerate(results):
        task_info = result_item.get('task_info', {})
        task_id = task_info.get('task_id', '')
        env_id = task_info.get('env_id', '')
        steps = result_item.get('steps', 0)
        has_injected_reward = 'injected_reward' in result_item

        # 条件：steps == 30 且没有 injected_reward 字段
        if steps == 30 and not has_injected_reward:
            need_process += 1
            print(f"\n[{idx+1}/{total}] 处理: {task_id}")
            print(f"  steps={steps}, injected_reward 缺失")

            # 查找对应的任务
            task = task_dict.get(task_id)
            if not task:
                print(f"  ⚠️ 未找到任务: {task_id}")
                failed += 1
                continue

            # 查找对应的环境
            env = envs.get(env_id)
            if not env:
                print(f"  ⚠️ 未找到环境: {env_id}")
                failed += 1
                continue

            # 获取环境信息
            env_class_code = env.get('env_class_code')
            env_class_name = task.get('env_class_name') or env.get('env_class_name')
            init_config = task.get('init_config', {})

            if not env_class_code or not env_class_name:
                print(f"  ⚠️ 环境信息不完整")
                failed += 1
                continue

            # 获取轨迹
            trajectory = result_item.get('trajectory', [])
            if not trajectory:
                print(f"  ⚠️ 轨迹为空")
                failed += 1
                continue

            # 重放轨迹获取最终状态
            print(f"  重放轨迹中...")
            final_state = replay_trajectory(env_class_code, env_class_name, init_config, trajectory)

            if final_state is None:
                print(f"  ❌ 重放轨迹失败")
                failed += 1
                continue

            # 计算 injected_reward
            injected_checklist = task.get('injected_checklist_with_func', [])
            if injected_checklist:
                injected_reward = calculate_reward(injected_checklist, final_state)
                result_item['injected_reward'] = injected_reward
                print(f"  injected_reward: {injected_reward}")

            # 重新计算 total_reward
            checklist = task.get('checklist_with_func', [])
            if checklist:
                total_reward = calculate_reward(checklist, final_state)
                result_item['total_reward'] = total_reward
                print(f"  total_reward (重新计算): {total_reward}")

            processed += 1
            print(f"  ✅ 处理完成")

        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"总结果数: {total}")
    print(f"需要处理 (steps=30 且无 injected_reward): {need_process}")
    print(f"成功处理: {processed}")
    print(f"处理失败: {failed}")
    print(f"跳过 (已满足条件): {skipped}")

    # 保存结果
    print(f"\n保存结果到: {result_file}")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("保存完成!")


def main():
    if len(sys.argv) < 4:
        print("用法: python verify_with_check_func.py <result_file.json> <task_file.json> <env_file.json>")
        print("\n示例:")
        print("  python verify_with_check_func.py result.json task.json env.json")
        sys.exit(1)

    result_file = sys.argv[1]
    task_file = sys.argv[2]
    env_file = sys.argv[3]

    if not os.path.exists(result_file):
        print(f"错误: 结果文件不存在: {result_file}")
        sys.exit(1)

    if not os.path.exists(task_file):
        print(f"错误: 任务文件不存在: {task_file}")
        sys.exit(1)

    if not os.path.exists(env_file):
        print(f"错误: 环境文件不存在: {env_file}")
        sys.exit(1)

    process_result_file(result_file, task_file, env_file)


if __name__ == "__main__":
    main()

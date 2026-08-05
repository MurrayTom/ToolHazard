"""
从 envscaler_sft_traj_9k_metadata.json 中过滤出满足条件的数据：
- traj_type == "non_conversation"
- task_info.env_id == "env_2_sft"
"""

import json
from typing import List, Dict, Any


def filter_trajectories(
    input_path: str,
    output_path: str,
    traj_type: str = "non_conversation",
    env_id: str = "env_2_sft",
) -> None:
    """
    从输入文件中过滤出满足条件的轨迹数据

    Args:
        input_path: 输入JSON文件路径
        output_path: 输出JSON文件路径
        traj_type: 要过滤的轨迹类型，默认为 "non_conversation"
        env_id: 要过滤的环境ID，默认为 "env_2_sft"
    """
    print(f"正在读取文件: {input_path}")
    
    # 读取整个JSON文件（如果是数组格式）
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 判断数据格式：是列表还是字典
    if isinstance(data, list):
        # 如果是列表，遍历每个元素
        all_items = data
    elif isinstance(data, dict):
        # 如果是字典，可能需要遍历值（根据实际结构调整）
        # 假设是 {key: item} 格式，取所有值
        all_items = list(data.values())
    else:
        raise ValueError(f"不支持的数据格式: {type(data)}")
    
    print(f"总共有 {len(all_items)} 条数据")
    
    # 过滤数据
    filtered_items = []
    for idx, item in enumerate(all_items):
        if not isinstance(item, dict):
            continue
        
        # 检查 traj_type
        if item.get("traj_type") != traj_type:
            continue
        
        # 检查 task_info.env_id
        task_info = item.get("task_info", {})
        if not isinstance(task_info, dict):
            continue
        
        if task_info.get("env_id") != env_id:
            continue
        
        # 满足所有条件，添加到结果列表
        filtered_items.append(item)
        
        # 每处理1000条打印一次进度
        if len(filtered_items) % 1000 == 0:
            print(f"已找到 {len(filtered_items)} 条匹配数据...")
    
    print(f"过滤完成，共找到 {len(filtered_items)} 条匹配数据")
    
    # 保存到新文件
    print(f"正在保存到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_items, f, ensure_ascii=False, indent=2)
    
    print(f"保存完成！共保存 {len(filtered_items)} 条数据")


if __name__ == "__main__":
    input_path = "/home/mouyutao/yangpengfei/EnvScaler/envscaler_sft_traj_9k_metadata.json"
    output_path = "/home/mouyutao/yangpengfei/EnvScaler/attacker/data/traj/env_45_sft_traj.json"
    
    filter_trajectories(
        input_path=input_path,
        output_path=output_path,
        traj_type="non_conversation",
        env_id="env_45_sft",
    )


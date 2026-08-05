"""
攻击点查找模块

本模块的主要功能是整合前两步的分析结果（可注入属性分析和操作读写分析），
找出每个可注入属性对应的读写操作，从而确定攻击点。

攻击点的定义：
- 一个可注入属性（来自步骤1）
- 该属性有至少一个写操作（来自步骤2）
- 该属性有至少一个读操作（来自步骤2）
"""
import json
import sys
import os
from typing import List, Dict, Any

# 添加父目录到Python路径，以便导入工具模块
# 根目录是 EnvScaler/attacker，当前文件在 attack_point/ 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
parent_dir = os.path.dirname(current_dir)  # 获取父目录，即 EnvScaler/attacker
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # 将父目录添加到Python搜索路径的最前面

# 导入分析模块
from attack_point.state_analysis import analyze_injectable_attributes  # 步骤1：分析可注入属性
from attack_point.operation_analysis import analyze_operations  # 步骤2：分析操作读写情况


def match_attributes_to_operations(
    injectable_attributes: List[str],
    operation_read_write: Dict[str, Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """
    匹配可注入属性和操作的读写情况，生成攻击点列表
    
    对于每个可注入属性，找出所有读取它的操作和写入它的操作。
    
    Args:
        injectable_attributes: 可注入属性列表，格式为 ["ClassName.attribute1", "ClassName.attribute2", ...]
        operation_read_write: 操作读写情况字典，格式为 {
            "operation_name": {
                "read": ["ClassName.attribute1", ...],
                "write": ["ClassName.attribute2", ...]
            },
            ...
        }
    
    Returns:
        List[Dict[str, Any]]: 攻击点列表，每个元素格式为 {
            "attribute": "ClassName.attribute1",
            "write": ["write_operation1", "write_operation2", ...],  # 写操作列表（可能为空）
            "read": ["read_operation1", "read_operation2", ...]     # 读操作列表（可能为空）
        }
    """
    attack_points = []  # 存储所有攻击点
    
    # 遍历每个可注入属性
    for attribute in injectable_attributes:
        write_operations = []  # 存储写入该属性的操作
        read_operations = []   # 存储读取该属性的操作
        
        # 遍历所有操作，检查哪些操作读写该属性
        for operation_name, read_write_info in operation_read_write.items():
            # 确保 read_write_info 是字典类型
            if not isinstance(read_write_info, dict):
                continue
            
            read_list = read_write_info.get("read", [])
            write_list = read_write_info.get("write", [])
            
            # 确保 read_list 和 write_list 是列表类型
            if not isinstance(read_list, list):
                read_list = []
            if not isinstance(write_list, list):
                write_list = []
            
            # 检查该操作是否读取该属性
            if attribute in read_list:
                read_operations.append(operation_name)
            
            # 检查该操作是否写入该属性
            if attribute in write_list:
                write_operations.append(operation_name)
        
        # 只有当“读操作列表”和“写操作列表”都非空时，才认为该属性是有效攻击点
        if write_operations and read_operations:
            attack_point = {
                "attribute": attribute,
                "write": write_operations,
                "read": read_operations,
            }
            attack_points.append(attack_point)
    
    return attack_points


def process_single_environment(
    env_id: str,
    env_data: Dict[str, Any],
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1"
) -> Dict[str, Any]:
    """
    处理单个环境，找出所有攻击点
    
    该函数会：
    1. 调用步骤1：分析可注入属性
    2. 调用步骤2：分析操作读写情况
    3. 匹配属性和操作，生成攻击点列表
    4. 在环境数据中添加 attack_point 字段
    
    Args:
        env_id: 环境ID（用于日志输出）
        env_data: 环境数据字典，包含 environment_summary, environment_introduction, 
                 constraints_rules, operation_list, env_class_code 等字段
        llm_provider: LLM提供商名称，默认为"openai"
        llm_model: LLM模型名称，默认为"gpt-4.1"
    
    Returns:
        Dict[str, Any]: 处理后的环境数据字典，新增了 attack_point 字段
    """
    print(f"\n{'='*80}")
    print(f"Processing environment: {env_id}")
    print(f"{'='*80}")
    
    # 步骤1：分析可注入属性
    print("\n[Step 1] Analyzing injectable attributes...")
    injectable_attributes = analyze_injectable_attributes(
        env_data=env_data,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    # 确保返回的是列表类型
    if not isinstance(injectable_attributes, list):
        print(f"Warning: analyze_injectable_attributes returned non-list type: {type(injectable_attributes)}")
        injectable_attributes = []
    print(f"Found {len(injectable_attributes)} injectable attributes: {injectable_attributes}")
    
    # 步骤2：分析操作读写情况
    print("\n[Step 2] Analyzing operation read/write...")
    operation_read_write = analyze_operations(
        env_data=env_data,
        llm_provider=llm_provider,
        llm_model=llm_model
    )
    # 确保返回的是字典类型
    if not isinstance(operation_read_write, dict):
        print(f"Warning: analyze_operations returned non-dict type: {type(operation_read_write)}")
        operation_read_write = {}
    print(f"Analyzed {len(operation_read_write)} operations")
    
    # 步骤3：匹配属性和操作，生成攻击点列表
    print("\n[Step 3] Matching attributes to operations...")
    attack_points = match_attributes_to_operations(
        injectable_attributes=injectable_attributes,
        operation_read_write=operation_read_write
    )
    
    # 统计信息
    attack_points_with_write = [ap for ap in attack_points if ap["write"]]
    attack_points_with_read = [ap for ap in attack_points if ap["read"]]
    attack_points_with_both = [ap for ap in attack_points if ap["write"] and ap["read"]]
    
    print(f"Total attack points: {len(attack_points)}")
    print(f"  - With write operations: {len(attack_points_with_write)}")
    print(f"  - With read operations: {len(attack_points_with_read)}")
    print(f"  - With both read and write: {len(attack_points_with_both)}")
    
    # 在环境数据中添加 attack_point 字段
    env_data_with_attack_points = env_data.copy()  # 复制原数据，避免修改原始数据
    env_data_with_attack_points["attack_point"] = attack_points
    
    return env_data_with_attack_points


def find_attack_points_for_all_environments(
    input_json_path: str,
    output_json_path: str = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1"
) -> Dict[str, Any]:
    """
    处理JSON文件中的所有环境，找出每个环境的攻击点
    
    该函数会：
    1. 读取输入的JSON文件（包含多个环境）
    2. 对每个环境调用 process_single_environment
    3. 将所有处理后的环境数据写入新文件
    
    Args:
        input_json_path: 输入JSON文件路径，例如 "/path/to/example_env.json"
        output_json_path: 输出JSON文件路径，如果为None，则自动生成（在输入文件同目录下）
        llm_provider: LLM提供商名称，默认为"openai"
        llm_model: LLM模型名称，默认为"gpt-4.1"
    
    Returns:
        Dict[str, Any]: 包含所有处理后的环境数据的字典，格式为 {
            "env_id1": {环境数据 + "attack_point": [...]},
            "env_id2": {环境数据 + "attack_point": [...]},
            ...
        }
    """
    # 读取输入JSON文件
    print(f"Reading input file: {input_json_path}")
    with open(input_json_path, "r", encoding="utf-8") as f:
        all_environments = json.load(f)
    
    print(f"Found {len(all_environments)} environment(s) to process")
    
    # 生成输出文件路径（如果未指定）
    if output_json_path is None:
        # 在输入文件同目录下，生成新文件名
        input_dir = os.path.dirname(input_json_path)
        input_basename = os.path.basename(input_json_path)
        input_name, input_ext = os.path.splitext(input_basename)
        output_json_path = os.path.join(input_dir, f"{input_name}_with_attack_points{input_ext}")

    # 处理每个环境；仅保留 attack_point 非空的环境
    processed_environments = {}
    for env_id, env_data in all_environments.items():
        try:
            # 处理单个环境
            processed_env_data = process_single_environment(
                env_id=env_id,
                env_data=env_data,
                llm_provider=llm_provider,
                llm_model=llm_model
            )
            attack_points = processed_env_data.get("attack_point", [])
            if isinstance(attack_points, list) and attack_points:
                processed_environments[env_id] = processed_env_data
                # 每新增一个“有攻击点”的环境，就立即落盘一次，避免中途中断导致结果丢失。
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(processed_environments, f, ensure_ascii=False, indent=2)
                print(f"Saved incremental result: {env_id} ({len(attack_points)} attack points)")
            else:
                print(f"No attack points found for {env_id}, skip saving this environment")
        except Exception as e:
            print(f"\nError processing environment {env_id}: {e}")
            print(f"Skipping environment {env_id}")

    # 写入输出JSON文件
    print(f"\n{'='*80}")
    print(f"Writing results to: {output_json_path}")
    print(f"{'='*80}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(processed_environments, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved results to {output_json_path}")
    
    return processed_environments


if __name__ == "__main__":
    """
    主函数：用于测试攻击点查找功能
    
    当直接运行此脚本时，会：
    1. 加载示例环境数据（example_env.json）
    2. 对每个环境进行分析，找出攻击点
    3. 将结果保存到新文件
    """
    # 测试用的JSON文件路径
    input_json_path = "/data/kcl/myt/mouyutao_workspace/ToolHazard/env_simulator/stage3_check_env/final_result/filtered_env_metadata.json"
    output_json_path = "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/envs/filtered_env_metadata_with_attack_points.json"
    
    #input_json_path = "/home/myt/agentic-security/ToolHazard/data/train_set/100_sft_train_env.json"
    #output_json_path = "/home/myt/agentic-security/ToolHazard/data/train_set/100_sft_train_env_with_attack_points.json"
    # 调用主函数，处理所有环境
    processed_environments = find_attack_points_for_all_environments(
        input_json_path=input_json_path,
        output_json_path=output_json_path,
        llm_provider="openai",   # 使用OpenAI作为LLM提供商
        llm_model="gpt-5.4-mini"      # 使用gpt-4.1模型
    )
    
    # # 打印最终统计信息
    # print("\n" + "="*80)
    # print("Final Summary:")
    # print("="*80)
    # for env_id, env_data in processed_environments.items():
    #     attack_points = env_data.get("attack_point", [])
    #     print(f"\n{env_id}:")
        # print(f"  Total attack points: {len(attack_points)}")
        # if attack_points:
        #     # 统计有读写操作的攻击点
        #     with_write = sum(1 for ap in attack_points if ap.get("write"))
        #     with_read = sum(1 for ap in attack_points if ap.get("read"))
        #     with_both = sum(1 for ap in attack_points if ap.get("write") and ap.get("read"))
        #     print(f"    - With write: {with_write}")
        #     print(f"    - With read: {with_read}")
        #     print(f"    - With both: {with_both}")


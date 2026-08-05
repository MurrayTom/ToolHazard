#!/usr/bin/env python3
"""
过滤掉 task_id 包含 "tool_selection" 的数据，并将结果写回原文件。
"""

import json

input_path = "/home/mouyutao/yangpengfei/EnvScaler/attacker/data/tasks/rl/rl_task_top_5_IPI_all.json"

print(f"正在读取文件: {input_path}")
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"原始数据条数: {len(data)}")

# 过滤：保留 task_id 中不包含 "tool_selection" 的记录
filtered_data = [item for item in data if "task_id" not in item or "tool_selection" not in item["task_id"]]

print(f"过滤后数据条数: {len(filtered_data)}")
print(f"移除了 {len(data) - len(filtered_data)} 条包含 'tool_selection' 的记录")

# 写回原文件（覆盖）
with open(input_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"已写回文件: {input_path}")

import json
import random
from collections import defaultdict

# 输入 JSON 文件路径
input_file = "/home/myt/agentic-security/ToolHazard/data/envscaler_sft_scenario_metadata.json"

# 输出 JSON 文件路径
output_file = "/home/myt/agentic-security/ToolHazard/data/train_set/rl_tasks.json"

# 读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[3318:]  # 取前 3318 条数据进行处理

# 按 env_id 分组
grouped = defaultdict(list)
for item in data:
    env_id = item.get("env_id")
    grouped[env_id].append(item)

# 对每个 env_id 随机抽取最多 5 个 task
sampled_data = []
for env_id, items in grouped.items():
    sampled_items = random.sample(items, min(5, len(items)))
    sampled_data.extend(sampled_items)

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)

print(f"已保存 {len(sampled_data)} 条数据到 {output_file}")
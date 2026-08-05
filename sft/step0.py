import json

# 读取文件
with open("/home/myt/agentic-security/ToolHazard/data/train_set/sft_traj_IPI.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 筛选
filtered_data = [
    item for item in data
    if item.get("injected_reward") == 0
    and item.get("total_reward") > 0.75
]

for i in range(len(filtered_data)):
    filtered_data[i]["traj_type"] = "non_conversation"
    # filtered_data[i]["tools"] = str(filtered_data[i].get("tools", []))
    # filtered_data[i]["messages"] = str(filtered_data[i].get("messages", []))
    # filtered_data[i]["user_messages"] = str([""])

# 保存结果
with open("/data/kcl/myt/data/envscaler/train_set/sft_traj_IPI.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"保留了 {len(filtered_data)} 条数据")
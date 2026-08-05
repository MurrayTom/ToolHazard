import json

# 文件路径
sft_file = "/home/myt/agentic-security/ToolHazard/data/train_set/rl_tasks_with_check_functions.json"
traj_file = "/data/kcl/myt/data/envscaler/envscaler_sft_traj_9k_metadata.json"

output_file = "/data/kcl/myt/data/envscaler/train_set/rl_tasks_with_check_functions_traj.json"

# 读取 sft tasks
with open(sft_file, "r", encoding="utf-8") as f:
    sft_data = json.load(f)

# 读取 traj
with open(traj_file, "r", encoding="utf-8") as f:
    traj_data = json.load(f)

print(f"SFT 任务数量: {len(sft_data)}")
print(f"Traj 数量: {len(traj_data)}")

# 提取 sft 中所有 task_id
sft_task_ids = set()

for item in sft_data:
    task_id = item.get("task_id")
    if task_id:
        sft_task_ids.add(task_id)

print(f"SFT task 数量: {len(sft_task_ids)}")

# 从 traj 中筛选
matched_traj = []

for traj_item in traj_data:
    traj_type = traj_item.get("traj_type", "")
    task_info = traj_item.get("task_info", {})
    task_id = task_info.get("task_id")

    if task_id in sft_task_ids and traj_type == "non_conversation":
        matched_traj.append(traj_item)

print(f"匹配到 traj 数量: {len(matched_traj)}")

# 保存结果
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(matched_traj, f, ensure_ascii=False, indent=2)

print(f"已保存到: {output_file}")
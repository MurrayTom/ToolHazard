import json
import glob

# 获取当前目录下所有 json 文件
json_files = [
    "rl_task_top_5_IPI_combined.json",
    "rl_task_top_5_IPI_important_template.json",
    "rl_task_top_5_IPI_multi_turn.json",
    "rl_task_top_5_IPI_decision_criteria_hijacking.json",
    "rl_task_top_5_IPI_reasoning_criteria.json",
    "rl_task_top_5_IPI_toolselection.json"
]

all_data = []

# 依次读取并合并
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # 如果文件内容是 list，就 extend
        if isinstance(data, list):
            all_data.extend(data)

        # 如果是 dict，就 append
        elif isinstance(data, dict):
            all_data.append(data)

# 保存合并后的结果
with open("merged_6_categories.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"合并完成，共读取 {len(json_files)} 个文件")
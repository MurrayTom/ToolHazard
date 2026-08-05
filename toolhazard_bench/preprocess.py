import json
import re

# 读取原始 json
with open("191_env_metadata_processed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

result = {}

for item in data:
    env_id = item.get("env_id", "")

    # 匹配 env_xxx_sft
    match = re.match(r"env_(\d+)_sft$", env_id)

    if match:
        num = int(match.group(1))

        # 只保留 101~140
        if num <= 100:
            result[env_id] = item

# 保存结果
with open("100_sft_train_env.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(f"过滤后保留 {len(result)} 条数据")
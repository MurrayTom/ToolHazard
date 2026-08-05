# import json

# # 读取原始 json 文件
# with open("./train_set/rl_traj.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 遍历每个 item
# for item in data:
#     tools = item.get("tools")

#     # 如果 tools 是字符串，则反序列化
#     if isinstance(tools, str):
#         item["tools"] = json.loads(tools)

# # 保存新文件
# with open("./train_set/rl_traj.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print("转换完成")


import json

with open("./train_set/40_rl_train_env_with_attack_points.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data.values():
    if "tools" in item:
        item["tools"] = json.loads(item["tools"])
        item["env_func_details"] = json.loads(item["env_func_details"])

with open("./train_set/40_rl_train_env_with_attack_points.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
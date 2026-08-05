from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:18000/v1",
    api_key="EMPTY"
)

resp = client.chat.completions.create(
    model="Qwen2.5-72B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "帮我生成一个复杂的安全研究任务"}
    ],
    temperature=0.7,
    max_tokens=1024,
)

print(resp.choices[0].message.content)

import requests
import json
print('111')

API_URL = "http://localhost:8000/v1/chat/completions"

# 测试请求
payload = {
    "model": "qwen-guard",
    "messages": [
        {"role": "system", "content": "你是一个智能助手。"},
        {"role": "user", "content": "请给我写一个简短的问候语。"}
    ],
    "max_tokens": 50
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    # vLLM 的返回结构里，生成内容通常在 choices[0].message.content
    message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("生成结果:\n", message)
else:
    print("请求失败:", response.status_code, response.text)


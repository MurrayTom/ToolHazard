import requests
import json

API_URL = "http://localhost:8000/v1/chat/completions"

# 测试请求
payload = 
{
    "model": "qwen-guard",
    "messages": [
        {"role": "system", "content": "你是一个智能助手。"
        },
        {"role": "user", "content": "请给我写一个简短的问候语。"
        }
    ],
    "max_tokens": 50
}

headers = 
{
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

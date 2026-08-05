# from openai import OpenAI

# client = OpenAI(
#     api_key="dummy",   # vLLM 不校验
#     base_url="http://localhost:8000/v1"
# )

# resp = client.chat.completions.create(
#     model="qwen3-1.7b",
#     messages=[
#         {"role": "user", "content": "写一句测试连通性的句子"}
#     ],
#     temperature=0.7
# )

# print(resp.choices[0].message.content)

from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8000/v1"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

resp = client.chat.completions.create(
    model="qwen3-4b",
    messages=[
        {"role": "user", "content": "北京天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto"   # 关键
)

print(resp)
"""
测试代码：使用 OpenAI 接口调用 Gemini
"""

from openai import OpenAI
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


# 方式 2：使用 OpenAI 兼容接口调用 Gemini（通过 Google AI 的兼容端点）
def call_gemini_via_openai():
    """
    使用 OpenAI 兼容接口调用 Gemini
    Google AI 提供 OpenAI SDK 兼容的端点
    """
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://yunwu.ai/v1"
        )

        response = client.chat.completions.create(
            model="gemini-3.1-flash-lite-preview",
            messages=[
                {"role": "user", "content": "解释什么是量子计算？"}
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        print("=== Gemini (OpenAI 兼容接口) ===")
        print(response.choices[0].message.content)
        print()

    except Exception as e:
        print(f"调用失败: {e}")


# 方式 3：通过代理服务（如 yunwu.ai）调用 Gemini
def call_gemini_via_proxy():
    """
    通过代理服务调用 Gemini
    使用现有的代理配置
    """
    try:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        # 不同的代理服务商可能支持不同的模型名称
        # 请根据实际服务商文档修改模型名称
        response = client.chat.completions.create(
            model="gemini-2.0-flash",  # 或 "gemini-pro" 等
            messages=[
                {"role": "user", "content": "解释什么是量子计算？"}
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        print("=== Gemini (通过代理) ===")
        print(response.choices[0].message.content)
        print()

    except Exception as e:
        print(f"调用失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("测试调用 Gemini")
    print("=" * 50)
    print()

    # 方式 2：OpenAI 兼容接口（最简单）
    print("尝试使用 OpenAI 兼容接口调用...")
    call_gemini_via_openai()

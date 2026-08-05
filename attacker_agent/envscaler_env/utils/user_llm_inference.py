"""
用户代理 LLM 推理工具模块

本模块提供用户代理的 LLM 推理接口，用于生成用户的回复。

功能：
- 封装 OpenAI API 调用
- 提供重试机制
- 统一的推理接口
"""
import time  # 用于重试时的延迟
import os  # 用于读取环境变量
from openai import OpenAI  # OpenAI 官方 Python SDK
from dotenv import load_dotenv  # 从 .env 文件加载环境变量

from typing import Optional, List, Dict, Any, Union  # 类型提示

# 加载环境变量（从 .env 文件读取 OPENAI_API_KEY 等配置）
load_dotenv()
 
def openai_llm_inference(
        model: str, 
        messages: List[dict],
        temperature: float = None, 
        stop_strs: Optional[List[str]] = None,
        max_tokens: int = None,
        api_key: str = None,
        base_url: str = None):
    """
    调用 OpenAI API 进行 LLM 推理（带重试机制）
    
    功能说明：
    - 调用 OpenAI API 生成文本响应
    - 如果 API 调用失败，自动重试最多 10 次
    - 每次重试间隔递增（10秒、20秒、30秒...）
    
    参数：
        model (str): 模型名称（如 "gpt-4.1"）
        messages (List[dict]): 消息列表，格式为 OpenAI 消息格式
                               例如：[{"role": "system", "content": "..."}, ...]
        temperature (float, optional): 温度参数，控制输出的随机性（0-2）
        stop_strs (Optional[List[str]]): 停止字符串列表，遇到这些字符串时停止生成
        max_tokens (int, optional): 最大生成 token 数
        api_key (str, optional): API 密钥，如果为 None 则从环境变量读取
        base_url (str, optional): API 基础 URL，如果为 None 则从环境变量读取
    
    返回：
        str: LLM 生成的文本响应，如果所有重试都失败则返回空字符串 ''
    
    异常处理：
        - KeyboardInterrupt: 用户中断时直接退出
        - 其他异常: 打印错误信息并重试
    
    使用示例：
        response = openai_llm_inference(
            model="gpt-4.1",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    # 创建 OpenAI 客户端
    # 优先使用传入的 api_key 和 base_url
    # 如果没有提供，则从环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL 读取
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"), 
        base_url=base_url or os.getenv("OPENAI_BASE_URL")
    )
    
    # 初始化重试计数器
    retries = 0  # 当前重试次数
    max_retries = 10  # 最大重试次数
    
    # 重试循环：最多尝试 max_retries 次
    while retries < max_retries:
        try:
            # 调用 OpenAI API 生成响应
            response = client.chat.completions.create(
                        model=model,              # 模型名称
                        messages=messages,        # 消息列表
                        stop=stop_strs,          # 停止字符串列表
                        temperature=temperature, # 温度参数
                        max_tokens=max_tokens    # 最大 token 数
                    )
            
            # 从响应中提取文本内容
            # response.choices[0] 是第一个（也是唯一的）响应选择
            # .message.content 是消息的文本内容
            output = response.choices[0].message.content
            
            # 返回生成的文本
            return output
            
        except KeyboardInterrupt:
            # 如果用户中断（Ctrl+C），直接退出循环
            print("Operation canceled by user.")
            break
        except Exception as e:
            # 如果 API 调用失败（网络错误、API 错误等）
            # 打印错误信息
            print(f"Someting wrong:{e}. Retrying in {retries*10+10} seconds...")
            
            # 等待一段时间后重试
            # 等待时间递增：第1次重试等10秒，第2次等20秒，第3次等30秒...
            time.sleep(retries*10) 
            
            # 增加重试计数
            retries += 1
    
    # 如果所有重试都失败了，返回空字符串
    # 这表示无法获取 LLM 响应
    return ''
    
    
def llm_inference(model, messages, provider, api_key=None, base_url=None):
    """
    统一的 LLM 推理接口（根据提供商选择具体实现）
    
    功能说明：
    - 提供统一的接口，隐藏底层实现细节
    - 根据 provider 选择对应的 LLM 推理实现
    - 目前只支持 "openai" 提供商
    
    参数：
        model (str): 模型名称
        messages (List[dict]): 消息列表
        provider (str): 模型提供商（目前只支持 "openai"）
        api_key (str, optional): API 密钥
        base_url (str, optional): API 基础 URL
    
    返回：
        str: LLM 生成的文本响应
    
    异常：
        ValueError: 如果 provider 不是 "openai"
    
    使用示例：
        response = llm_inference(
            model="gpt-4.1",
            messages=[...],
            provider="openai"
        )
    """
    # 根据提供商选择具体的实现
    if provider == "openai":
        # 调用 OpenAI 的实现
        return openai_llm_inference(
            model=model,              # 模型名称
            messages=messages,        # 消息列表
            temperature=0.7,          # 固定温度参数为 0.7（适中的随机性）
            api_key=api_key,          # API 密钥
            base_url=base_url         # API 基础 URL
        )
    else:
        # 如果是不支持的提供商，抛出异常
        # 未来可以在这里添加其他提供商的支持（如 Anthropic、Google 等）
        raise ValueError(f"Invalid provider: {provider}.")

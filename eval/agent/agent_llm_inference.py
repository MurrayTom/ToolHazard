"""
LLM 推理工具模块 - 用于 Agent 的 LLM 推理调用

本模块提供了与 LLM（大语言模型）交互的接口，支持两种推理模式：
1. Prompt 模式：通过文本提示词进行工具调用，Agent 需要自己解析响应
2. FC 模式（Function Calling）：使用 LLM 原生的函数调用接口，返回结构化响应

主要功能：
- 封装 OpenAI API 调用
- 支持流式和非流式推理
- 支持思考模式（thinking mode）
- 自动重试机制
- 错误处理
"""

import os  # 用于读取环境变量
import time  # 用于重试时的延迟
from openai import OpenAI  # OpenAI 官方 Python SDK
from dotenv import load_dotenv  # 从 .env 文件加载环境变量
from typing import List, Dict, Any, Tuple, Optional  # 类型提示

# 加载环境变量（从 .env 文件读取 OPENAI_API_KEY 等配置）
load_dotenv()


def openai_inference_prompt(
    model: str, 
    messages: List[Dict[str, Any]], 
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
    ) -> str:
    """
    OpenAI 非流式推理函数（Prompt 模式）
    
    功能：调用 OpenAI API 进行一次性推理，返回完整的文本响应。
    适用于不需要实时流式输出的场景。
    
    参数说明：
        model (str): 要使用的模型名称，如 "gpt-4.1-mini"
        messages (List[Dict]): 对话消息列表，格式为 [{"role": "system", "content": "..."}, ...]
        temperature (float, optional): 温度参数，控制输出的随机性（0-2），None 表示使用默认值
        enable_thinking (bool): 是否启用思考模式，仅支持支持思考的模型（如 Qwen3）
        api_key (str, optional): API 密钥，如果为 None 则从环境变量读取
        base_url (str, optional): API 基础 URL，如果为 None 则从环境变量读取
    
    返回：
        str: LLM 生成的文本响应，如果启用思考模式，会包含 <think>...</think> 标签
    
    异常处理：
        - 如果 API 调用失败，会自动重试最多 10 次
        - 每次重试间隔递增（10秒、20秒、30秒...）
        - 如果所有重试都失败，返回空字符串 ''
    """
    # 创建 OpenAI 客户端
    # 优先使用传入的 api_key 和 base_url，如果没有则从环境变量读取
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
            # ========== 检查是否为标准 OpenAI API ==========
            # 获取实际的 base_url（优先使用传入的，其次环境变量，最后使用默认值）
            actual_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            # 判断是否为标准 OpenAI API
            # 标准 OpenAI API 不支持 chat_template_kwargs 参数
            # 只有支持思考模式的 API（如 Qwen3）才需要这个参数
            is_standard_openai = actual_base_url.rstrip('/') in [
                "https://api.openai.com/v1", 
                "https://api.openai.com"
            ]
            
            # ========== 构建 API 请求参数 ==========
            create_params = {
                "model": model,              # 模型名称
                "messages": messages,         # 对话消息列表
                "stream": False,             # 非流式模式（一次性返回完整响应）
                "temperature": temperature,  # 温度参数
                "max_tokens": 8000,         # 最大生成 token 数（不能超过 max_model_len=8192）
                "n": 1,                      # 生成 1 个响应（不生成多个候选）
            }
            
            # ========== 条件性添加思考模式参数 ==========
            # 只有当启用思考模式且不是标准 OpenAI API 时才添加此参数
            # chat_template_kwargs 是用于支持思考模式的扩展参数
            # 标准 OpenAI API 不支持此参数，会导致 400 错误
            if enable_thinking and not is_standard_openai:
                create_params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking
                    }
                }
            
            # ========== 调用 OpenAI API ==========
            # 发送请求到 OpenAI API
            response = client.chat.completions.create(**create_params)
            
            # ========== 提取响应内容 ==========
            # 从响应中提取文本内容
            # response.choices[0] 是第一个（也是唯一的）响应选择
            # .message.content 是消息的文本内容
            content = response.choices[0].message.content
            
            # ========== 处理思考内容（如果存在） ==========
            # 检查响应中是否包含思考内容（reasoning_content）
            # 这是某些模型（如 Qwen3）支持的额外字段，用于输出思考过程
            if hasattr(response.choices[0].message, "reasoning_content"):
                # 如果响应对象有 reasoning_content 属性，提取它
                reasoning_content = response.choices[0].message.reasoning_content
            else:
                # 如果没有，设置为空字符串
                reasoning_content = ""
            
            # ========== 格式化思考内容 ==========
            # 如果存在思考内容，将其格式化为 <think>...</think> 标签格式
            # 这是 Qwen3 模板风格，用于在响应中标识思考过程
            if reasoning_content:
                reasoning_content = reasoning_content.strip()  # 去除首尾空白
                # 将思考内容添加到响应前面，格式：<think>思考内容</think>\n\n实际响应
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            
            # 返回格式化后的完整响应
            return content

        except Exception as e:
            # ========== 异常处理和重试 ==========
            # 如果 API 调用失败，打印错误信息
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            
            # 等待一段时间后重试
            # 等待时间递增：第1次重试等10秒，第2次等20秒，第3次等30秒...
            time.sleep(retries * 10)
            
            # 增加重试计数
            retries += 1
    
    # ========== 所有重试都失败 ==========
    # 如果所有重试都失败了，打印错误信息并返回空字符串
    print(f"Failed to get response after {max_retries} retries.")
    return ''  # 返回空字符串表示失败


def openai_stream_inference_prompt(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
) -> str:
    """
    OpenAI 流式推理函数（Prompt 模式）
    
    功能：调用 OpenAI API 进行流式推理，实时接收响应片段并拼接成完整响应。
    适用于需要实时显示响应的场景，用户体验更好。
    
    参数说明：
        model (str): 模型名称
        messages (List[Dict]): 对话消息列表
        temperature (float, optional): 温度参数
        enable_thinking (bool): 是否启用思考模式
        api_key (str, optional): API 密钥
        base_url (str, optional): API 基础 URL
    
    返回：
        str: 完整的文本响应（流式接收后拼接）
    
    与 openai_inference_prompt 的区别：
        - 使用 stream=True 进行流式调用
        - 实时接收响应片段（chunks）
        - 累积所有片段拼接成完整响应
        - 如果重试5次后仍失败，会将 max_tokens 降低到 5000
    """
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"), 
        base_url=base_url or os.getenv("OPENAI_BASE_URL")
    )

    # 初始化重试相关变量
    retries = 0  # 当前重试次数
    max_retries = 10  # 最大重试次数
    max_tokens = 8000  # 最大 token 数
    
    # 重试循环
    while retries < max_retries:
        # ========== 检查是否为标准 OpenAI API ==========
        # 获取实际的 base_url
        actual_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        
        # 判断是否为标准 OpenAI API
        is_standard_openai = actual_base_url.rstrip('/') in [
            "https://api.openai.com/v1", 
            "https://api.openai.com"
        ]
        
        # ========== 构建 API 请求参数 ==========
        params = {
            "model": model,              # 模型名称
            "messages": messages,         # 对话消息列表
            "stream": True,              # 启用流式模式（关键区别）
            "temperature": temperature,  # 温度参数
            "max_tokens": max_tokens,    # 最大 token 数（可能在重试时调整）
            "n": 1                       # 生成 1 个响应
        }
        
        # ========== 条件性添加思考模式参数 ==========
        # 只有启用思考模式且不是标准 OpenAI API 时才添加
        if enable_thinking and not is_standard_openai:
            params["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking
                }
            }
        
        try:
            # ========== 调用 OpenAI API（流式模式） ==========
            # 发送流式请求，返回一个迭代器（generator）
            completion = client.chat.completions.create(**params)

            # ========== 初始化累积变量 ==========
            reasoning_content = ""  # 用于累积思考内容
            content = ""            # 用于累积主要响应内容

            # ========== 流式接收响应片段 ==========
            # 遍历流式响应的每个片段（chunk）
            for chunk in completion:
                # 检查 chunk 是否有 choices 属性
                # 某些 chunk 可能不包含有效数据，需要跳过
                if not getattr(chunk, "choices", None):
                    continue  # 跳过无效的 chunk

                # ========== 提取 chunk 中的数据 ==========
                choice = chunk.choices[0]  # 获取第一个（也是唯一的）选择
                delta = choice.delta        # delta 包含本次 chunk 的增量数据

                # ========== 累积思考内容 ==========
                # 检查 delta 中是否包含思考内容
                # 思考内容可能分布在多个 chunk 中，需要累积
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    # 将本次 chunk 的思考内容追加到累积变量中
                    reasoning_content += delta.reasoning_content

                # ========== 累积主要响应内容 ==========
                # 检查 delta 中是否包含主要响应内容
                # 响应内容也可能分布在多个 chunk 中，需要累积
                if hasattr(delta, "content") and delta.content:
                    # 将本次 chunk 的内容追加到累积变量中
                    content += delta.content

            # ========== 清理累积的内容 ==========
            # 去除首尾空白字符
            reasoning_content = reasoning_content.strip()
            content = content.strip()

            # ========== 处理已格式化的思考标签 ==========
            # 某些情况下，思考内容可能已经以标签形式包含在 content 中
            # 如果 reasoning_content 为空但 content 中包含 </think> 标签
            # 说明思考内容已经在 content 中了，需要提取出来
            if not reasoning_content and content and '</think>' in content:
                # 提取思考内容（标签之间的部分）
                reasoning_content = content.split('</think>')[0].strip()
                # 如果包含开始标签，去除它
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                # 提取实际响应内容（标签之后的部分）
                content = content.split('</think>')[1].strip()
            
            # ========== 格式化思考内容 ==========
            # 如果存在思考内容，将其格式化为标准格式
            if reasoning_content:
                # 格式：<think>思考内容</think>\n\n实际响应
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"

            # ========== 验证响应不为空 ==========
            # 如果内容为空，抛出异常触发重试
            if content == "":
                raise ValueError("content is empty.")
            
            # 返回完整的响应内容
            return content
        
        except Exception as e:
            # ========== 异常处理和重试 ==========
            # 打印错误信息
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            
            # 等待后重试
            time.sleep(retries * 10)
            
            # ========== 自适应调整 max_tokens ==========
            # 如果重试次数达到 5 次，降低 max_tokens 到 5000
            # 这可能是为了应对某些 API 限制或错误
            if retries >= 5:
                max_tokens = 4000
                print(f"max_tokens: {max_tokens}")
            
            # 增加重试计数
            retries += 1

    # ========== 所有重试都失败 ==========
    # 如果所有重试都失败，打印错误信息并返回空字符串
    print(f"Failed to get response after {max_retries} retries.")
    return ""


def openai_stream_inference_fc(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
) -> Dict[str, Any]:
    """
    OpenAI 流式推理函数（FC 模式 - Function Calling）
    
    功能：使用 LLM 原生的函数调用接口进行流式推理。
    返回结构化的响应，包含工具调用、文本内容和思考内容。
    
    参数说明：
        model (str): 模型名称
        messages (List[Dict]): 对话消息列表
        temperature (float, optional): 温度参数
        tools (Optional[List[Dict]]): 工具列表，格式为 OpenAI Function Calling 格式
        enable_thinking (bool): 是否启用思考模式
        api_key (str, optional): API 密钥
        base_url (str, optional): API 基础 URL
    
    返回：
        Dict[str, Any]: 包含以下键的字典
            - "reasoning_content" (str): 思考内容（如果启用）
            - "tool_calls" (list): 工具调用列表，格式为：
                [{
                    "id": "call_xxx",
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "arguments": '{"key": "value"}'
                    }
                }]
            - "content" (str): 文本内容
    
    与 Prompt 模式的区别：
        - 使用 tools 参数传递工具列表
        - LLM 返回结构化的 tool_calls 对象
        - 不需要解析文本提取工具调用
        - 支持多个工具调用（但本实现只保留第一个）
    """
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"), 
        base_url=base_url or os.getenv("OPENAI_BASE_URL")
    )

    print(api_key, base_url)

    # 初始化重试相关变量
    retries = 0
    max_retries = 10
    
    # 重试循环
    while retries < max_retries:
        try:
            # ========== 检查是否为标准 OpenAI API ==========
            actual_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            is_standard_openai = actual_base_url.rstrip('/') in [
                "https://api.openai.com/v1", 
                "https://api.openai.com"
            ]
            
            # ========== 构建基础请求参数 ==========
            base_params = {
                "model": model,              # 模型名称
                "messages": messages,         # 对话消息列表
                "stream": True,              # 启用流式模式
                "temperature": temperature,  # 温度参数
                "max_tokens": 8000,         # 最大 token 数
                "n": 1,                      # 生成 1 个响应
            }
            
            # ========== 条件性添加思考模式参数 ==========
            if enable_thinking and not is_standard_openai:
                base_params["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": enable_thinking
                    }
                }
            
            # ========== 调用 API（根据是否有工具选择不同的参数） ==========
            if tools:
                # 如果有工具列表，使用工具调用模式
                completion = client.chat.completions.create(
                    **base_params,           # 基础参数
                    tools=tools,             # 工具列表（关键：FC 模式需要）
                    tool_choice="auto",      # 自动选择是否调用工具
                    top_p=0.95,              # 核采样参数（控制输出的多样性）
                )
            else:
                # 如果没有工具列表，使用普通模式
                completion = client.chat.completions.create(**base_params)

            # ========== 初始化累积变量 ==========
            reasoning_content = ""  # 思考内容
            content = ""            # 文本内容
            # 工具调用累积字典，按 index 分组
            # 因为流式响应中，同一个工具调用的不同部分可能分布在多个 chunk 中
            # 使用 index 作为键来分组累积
            tool_calls_accum: Dict[int, Dict[str, Any]] = {}

            # ========== 流式接收响应片段 ==========
            for chunk in completion:
                # 跳过无效的 chunk
                if not getattr(chunk, "choices", None):
                    continue

                # 提取 chunk 数据
                choice = chunk.choices[0]
                delta = choice.delta

                # ========== 累积思考内容 ==========
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content

                # ========== 累积文本内容 ==========
                if hasattr(delta, "content") and delta.content:
                    content += delta.content

                # ========== 累积工具调用信息 ==========
                # 这是 FC 模式的核心：处理工具调用
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    # 遍历本次 chunk 中的所有工具调用片段
                    for tool_call in delta.tool_calls:
                        # 获取工具调用的索引（index）
                        # 同一个工具调用的不同部分（id、name、arguments）可能有相同的 index
                        idx = tool_call.index
                        
                        # ========== 初始化工具调用结构 ==========
                        # 如果这个 index 的工具调用还没有初始化，创建一个新的结构
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tool_call.id or "",        # 工具调用 ID（可能为空）
                                "type": tool_call.type or "function",  # 类型（默认为 function）
                                "function": {
                                    "name": "",      # 函数名称（需要累积）
                                    "arguments": ""  # 函数参数（JSON 字符串，需要累积）
                                }
                            }
                        
                        # ========== 更新工具调用 ID ==========
                        # 如果本次 chunk 包含 ID，更新它
                        if tool_call.id:
                            tool_calls_accum[idx]["id"] = tool_call.id
                        
                        # ========== 更新工具调用类型 ==========
                        # 如果本次 chunk 包含类型，更新它
                        if tool_call.type:
                            tool_calls_accum[idx]["type"] = tool_call.type
                        
                        # ========== 累积函数信息 ==========
                        if tool_call.function:
                            # 累积函数名称（可能分布在多个 chunk 中）
                            if tool_call.function.name:
                                # 注意：这里使用 += 是因为名称可能分多次传输
                                tool_calls_accum[idx]["function"]["name"] += tool_call.function.name
                            
                            # 累积函数参数（JSON 字符串，可能分多次传输）
                            if tool_call.function.arguments:
                                # 注意：arguments 是 JSON 字符串，需要拼接
                                tool_calls_accum[idx]["function"]["arguments"] += tool_call.function.arguments

            # ========== 转换为工具调用列表 ==========
            # 将累积字典转换为列表
            tool_calls = list(tool_calls_accum.values())
            
            # ========== 处理多个工具调用 ==========
            # 如果 LLM 返回了多个工具调用，只保留第一个
            # 这是因为当前实现只支持单个工具调用
            if len(tool_calls) > 1:
                print("warning: more than one tool_call, only keep the first one.")
                tool_calls = [tool_calls[0]]

            # ========== 处理已格式化的思考标签 ==========
            # 如果思考内容已经以标签形式包含在 content 中，提取出来
            if not reasoning_content and content and '</think>' in content:
                # 提取思考内容（标签之间的部分）
                reasoning_content = content.split('</think>')[0].strip()
                # 如果包含开始标签，去除它
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                # 提取实际响应内容（标签之后的部分）
                content = content.split('</think>')[1].strip()
                
            # ========== 验证响应不为空 ==========
            # 如果内容、工具调用和思考内容都为空，抛出异常
            if not content and not tool_calls and not reasoning_content:
                raise ValueError("all content is empty.")
        
            # ========== 构建返回结果 ==========
            result = {
                "reasoning_content": reasoning_content,  # 思考内容
                "tool_calls": tool_calls,                  # 工具调用列表
                "content": content                         # 文本内容
            }
        
            # 返回结构化结果
            return result
        
        except Exception as e:
            # ========== 异常处理和重试 ==========
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            retries += 1

    # ========== 所有重试都失败 ==========
    print(f"Failed to get response after {max_retries} retries.")
    # 返回空结果字典
    return {"reasoning_content": "", "tool_calls": [], "content": ""}


def llm_inference_fc(
    provider: str, 
    model: str, 
    messages: List[Dict[str, Any]], 
    temperature: float = None, 
    tools: Optional[List[Dict]] = None, 
    enable_thinking: bool = False, 
    api_key: str = None, 
    base_url: str = None
) -> Dict[str, Any]:
    """
    LLM 推理统一接口（FC 模式）
    
    功能：提供统一的 FC 模式推理接口，根据 provider 选择具体的实现。
    这是对外暴露的统一接口，隐藏了底层实现细节。
    
    参数说明：
        provider (str): 模型提供商，目前只支持 "openai"
        model (str): 模型名称
        messages (List[Dict]): 对话消息列表
        temperature (float, optional): 温度参数
        tools (Optional[List[Dict]]): 工具列表（FC 模式必需）
        enable_thinking (bool): 是否启用思考模式
        api_key (str, optional): API 密钥
        base_url (str, optional): API 基础 URL
    
    返回：
        Dict[str, Any]: 包含 reasoning_content、tool_calls、content 的字典
    
    设计目的：
        - 提供统一的接口，便于扩展其他提供商（如 Anthropic、Google 等）
        - 隐藏底层实现细节
        - 便于测试和维护
    """
    # 根据提供商选择具体的实现
    if provider == "openai":
        # 调用 OpenAI 的 FC 模式实现
        return openai_stream_inference_fc(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            tools=tools, 
            enable_thinking=enable_thinking, 
            api_key=api_key, 
            base_url=base_url
        )
    else:
        # 如果是不支持的提供商，抛出异常
        # 未来可以在这里添加其他提供商的支持（如 Anthropic、Google 等）
        raise ValueError(f"Invalid provider: {provider}")


def llm_inference_prompt(
    provider: str, 
    model: str, 
    messages: List[Dict[str, Any]], 
    temperature: float = None, 
    enable_thinking: bool = False, 
    api_key: str = None, 
    base_url: str = None
) -> str:
    """
    LLM 推理统一接口（Prompt 模式）
    
    功能：提供统一的 Prompt 模式推理接口，根据 provider 选择具体的实现。
    这是对外暴露的统一接口，隐藏了底层实现细节。
    
    参数说明：
        provider (str): 模型提供商，目前只支持 "openai"
        model (str): 模型名称
        messages (List[Dict]): 对话消息列表
        temperature (float, optional): 温度参数
        enable_thinking (bool): 是否启用思考模式
        api_key (str, optional): API 密钥
        base_url (str, optional): API 基础 URL
    
    返回：
        str: LLM 生成的文本响应
    
    设计目的：
        - 提供统一的接口，便于扩展其他提供商
        - 隐藏底层实现细节
        - 便于测试和维护
    """
    # 根据提供商选择具体的实现
    if provider == "openai":
        # 调用 OpenAI 的 Prompt 模式实现（使用流式版本）
        return openai_stream_inference_prompt(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            enable_thinking=enable_thinking, 
            api_key=api_key, 
            base_url=base_url
        )
    else:
        # 如果是不支持的提供商，抛出异常
        # 未来可以在这里添加其他提供商的支持
        raise ValueError(f"Invalid provider: {provider}")


# ========== 测试代码 ==========
# 当直接运行此文件时，执行测试代码
if __name__ ==  "__main__":
    # 测试 FC 模式的功能
    
    # 构建测试消息
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Beijing?"}
    ]

    # 构建测试工具列表
    # 这是 OpenAI Function Calling 格式的工具定义
    tools = [
        {
            "type": "function",  # 工具类型：函数
            "function": {
                "name": "get_current_weather",  # 函数名称
                "description": "Get the current weather of a city",  # 函数描述
                "parameters": {  # 函数参数定义（JSON Schema 格式）
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]  # 必需参数
                }
            }
        }
    ]

    # 测试配置
    model = "claude-opus-4-7"
    provider = "openai"
    
    # 调用 FC 模式推理
    result = llm_inference_fc(
        provider=provider,
        model=model, 
        messages=msgs, 
        tools=tools,
    )
    
    # 打印结果
    print(result)

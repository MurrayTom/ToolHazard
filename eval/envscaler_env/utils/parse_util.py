"""
模型响应解析工具函数模块

本模块提供以下核心功能：
1. 从 Prompt 模式的文本响应中解析思考内容、工具调用和文本内容
2. 将解析结果转换为统一的 action 格式

支持的格式：
- 思考标签：<think>...</think> 或 @think@...@/think@
- 工具调用：<tool_call>{...}</tool_call>
- 文本内容：标签之间的内容或纯文本
"""
import re  # 正则表达式模块，用于模式匹配
import json  # JSON 解析模块，用于解析工具调用的 JSON
from copy import deepcopy  # 深拷贝工具，避免修改原始数据


def parse_response(text):
    """
    从模型原始输出文本中解析 (reasoning_content, tool_calls, content)
    
    功能说明：
    - 解析 Prompt 模式下 LLM 返回的文本响应
    - 提取思考内容（<think>...</think> 标签）
    - 提取工具调用（<tool_call>...</tool_call> 标签）
    - 提取文本内容（标签之间的内容）
    
    参数：
        text (str): 模型原始输出文本
                    例如："<think>我需要检查库存</think>\n<tool_call>\n{\"name\": \"check_inventory\"}\n</tool_call>"
    
    返回：
        tuple: (parse_success, result_dict)
            - parse_success (bool): 解析是否成功（False 表示结构错误，如未闭合的标签）
            - result_dict (dict): 解析结果字典
                {
                    "reasoning_content": str | None,  # 思考内容
                    "tool_calls": [{"function": {...}}] | None,  # 工具调用列表
                    "content": str | None  # 文本内容
                }
    
    使用示例：
        text = "<think>思考</think>\n<tool_call>\n{\"name\": \"test\"}\n</tool_call>"
        success, result = parse_response(text)
        # result = {"reasoning_content": "思考", "tool_calls": [{"function": {"name": "test"}}], "content": ""}
    """
    # 默认解析成功，只有在遇到结构错误时才设置为 False
    # 例如：未闭合的标签、JSON 解析错误等
    parse_success = True
    
    # 初始化结果字典，所有字段默认为 None
    result = {"reasoning_content": None, "tool_calls": None, "content": None}
    
    # 去除文本首尾的空白字符（空格、换行等）
    text = text.strip()

    # ========== 匹配思考内容块 ==========
    # 使用正则表达式匹配思考标签
    # 支持两种格式：
    # 1. <think>...</think>（标准格式）
    # 2. @think@...@/think@（兼容格式）
    think_match = re.search(
        # 正则表达式说明：
        # (?:<|@)think(?:>|@)  - 匹配开始标签：<think> 或 @think@
        # \s*                   - 匹配零个或多个空白字符
        # (.*?)                 - 非贪婪匹配思考内容（捕获组 1）
        # (?:<|@)/think(?:>|@) - 匹配结束标签：</think> 或 @/think@
        r'(?:<|@)think(?:>|@)\s*(.*?)(?:<|@)/think(?:>|@)',
        text,
        re.DOTALL | re.IGNORECASE,  # DOTALL: . 匹配换行符；IGNORECASE: 忽略大小写
    )

    # 如果找到完整的思考标签对
    if think_match:
        # 提取思考内容（捕获组 1，即标签之间的内容）
        # .strip() 去除首尾空白
        result["reasoning_content"] = think_match.group(1).strip()
    # 如果只找到开始标签但没有结束标签
    elif re.search(r'(?:<|@)think(?:>|@)', text, re.IGNORECASE):
        # 这是结构错误：开始标签存在但未闭合
        parse_success = False  # 标记解析失败
        # 记录错误信息
        result["reasoning_content"] = {"error": "Missing </think> or malformed think block"}

    # ========== 匹配工具调用块 ==========
    # 使用正则表达式查找所有工具调用标签
    # finditer 返回所有匹配项的迭代器
    tool_calls = list(re.finditer(
        # 正则表达式说明：
        # <tool_call>           - 匹配开始标签
        # \s*                   - 匹配零个或多个空白字符
        # (\{.*?\})             - 非贪婪匹配 JSON 对象（捕获组 1）
        # \s*                   - 匹配零个或多个空白字符
        # </tool_call>          - 匹配结束标签
        r'<tool_call>\s*(\{.*?\})\s*</tool_call>', 
        text, 
        re.DOTALL  # DOTALL: . 匹配换行符（因为 JSON 可能跨多行）
    ))
    
    # 初始化工具调用内容变量
    tool_call_content = None

    # 如果找到工具调用
    if tool_calls:
        # 如果找到多个工具调用，只使用第一个
        if len(tool_calls) > 1:
            print("Multiple <tool_call> found, using the first one.")
        # 获取第一个匹配项
        tool_call_match = tool_calls[0]
        # 提取 JSON 内容（捕获组 1，即标签之间的 JSON 字符串）
        tool_call_content = tool_call_match.group(1)
    else:
        # 如果没有找到完整的工具调用标签
        # 检查是否存在未闭合的开始标签
        if "<tool_call>" in text and "</tool_call>" not in text:
            # 这是结构错误：开始标签存在但未闭合
            parse_success = False
            # 记录错误信息
            result["tool_calls"] = [{"error": "Unclosed <tool_call> tag"}]

    # ========== 提取文本内容 ==========
    # 根据思考标签和工具调用标签的存在情况，提取文本内容
    # 文本内容是指标签之间的内容或标签外的内容
    
    if think_match and tool_call_content:
        # 情况 1：同时存在思考标签和工具调用
        # 文本内容在两者之间
        think_end = think_match.end()  # 思考标签结束位置
        tool_start = tool_call_match.start()  # 工具调用开始位置
        # 提取两者之间的文本
        result["content"] = text[think_end:tool_start].strip()
    elif think_match and not tool_call_content:
        # 情况 2：只有思考标签，没有工具调用
        # 文本内容在思考标签之后
        think_end = think_match.end()  # 思考标签结束位置
        # 提取思考标签之后的所有文本
        result["content"] = text[think_end:].strip()
    elif not think_match and tool_call_content:
        # 情况 3：只有工具调用，没有思考标签
        # 文本内容在工具调用之前
        tool_start = tool_call_match.start()  # 工具调用开始位置
        # 提取工具调用之前的所有文本
        result["content"] = text[:tool_start].strip()
    else:
        # 情况 4：既没有思考标签，也没有工具调用
        # 整个文本就是内容（可能是纯文本响应或 "Task Completed"）
        result["content"] = text.strip()

    # ========== 解析工具调用的 JSON ==========
    # 如果找到了工具调用内容，需要解析 JSON
    if tool_call_content:
        try:
            # 将 JSON 字符串解析为 Python 字典
            # json.loads() 将字符串转换为字典对象
            tool_call_dict = json.loads(tool_call_content)
            
            # 检查必需字段是否存在
            # 工具调用必须包含 "name" 和 "arguments" 字段
            required_fields = ["name", "arguments"]
            # 找出缺失的字段
            missing = [f for f in required_fields if f not in tool_call_dict]
            
            # 如果有缺失字段
            if missing:
                parse_success = False  # 标记为 JSON 结构错误
                # 记录错误信息，包含缺失的字段和原始数据
                result["tool_calls"] = [{
                    "error": f"Missing required field(s): {missing}",
                    "raw": tool_call_dict,  # 保存原始数据以便调试
                }]
            else:
                # 如果所有必需字段都存在，构建工具调用结果
                # 格式化为标准格式：{"function": {"name": "...", "arguments": {...}}}
                result["tool_calls"] = [{"function": tool_call_dict}]
        except json.JSONDecodeError as e:
            # 如果 JSON 解析失败（语法错误）
            parse_success = False  # 标记为 JSON 解析错误
            # 记录错误信息，包含错误详情和原始内容
            result["tool_calls"] = [{
                "error": f"Failed to parse tool_call JSON: {e}",
                "raw": tool_call_content,  # 保存原始内容以便调试
            }]

    # 返回解析结果
    return parse_success, result


def parse_action(struct_response: dict):
    """
    将 struct_response 解析为 action
    
    功能说明：
    - 接收解析后的结构化响应（包含 tool_calls 和 content）
    - 根据内容类型转换为统一的 action 格式
    - 处理四种情况：只有工具调用、只有文本、两者都有、两者都无
    
    参数：
        struct_response (dict): 结构化响应字典
            {
                "reasoning_content": str | None,
                "tool_calls": [{"function": {...}}] | None,
                "content": str | None
            }
    
    返回：
        tuple: (parse_success, action_dict)
            - parse_success (bool): 解析是否成功
            - action_dict (dict): action 字典
                {
                    "name": "function_name" | "chat_with_user",
                    "arguments": {...}
                }
    
    使用示例：
        struct = {"tool_calls": [{"function": {"name": "test", "arguments": {"x": 1}}}], "content": ""}
        success, action = parse_action(struct)
        # action = {"name": "test", "arguments": {"x": 1}}
    """
    try:
        # ========== 情况 1：只有工具调用，没有文本内容 ==========
        # 检查：content 为空或不存在，且 tool_calls 存在
        if not struct_response.get("content") and struct_response.get("tool_calls"):
            # 深拷贝工具调用信息，避免修改原始数据
            # 从 tool_calls 列表中取第一个工具调用
            action = deepcopy(struct_response["tool_calls"][0]['function'])
            
            # 如果 arguments 是字符串（JSON 字符串），需要解析为字典
            if isinstance(action['arguments'], str):
                # 容错处理：空字符串或 null 转为空字典
                if action['arguments'].strip() in ("", "null", "None"):
                    action['arguments'] = {}
                else:
                    action['arguments'] = json.loads(action['arguments'])

        # ========== 情况 2：只有文本内容，没有工具调用 ==========
        # 检查：content 存在且不为空，且 tool_calls 不存在或为空
        elif struct_response.get("content") and not struct_response.get("tool_calls"):
            # 将文本内容转换为 chat_with_user action
            # 这表示 Agent 想要与用户交互（如输出 "Task Completed"）
            action = {
                "name": "chat_with_user",  # 固定动作名称
                "arguments": {"content": struct_response.get("content")}  # 文本内容作为参数
            }

        # ========== 情况 3：同时有文本内容和工具调用 ==========
        # 检查：content 存在且 tool_calls 也存在
        elif struct_response.get("content") and struct_response.get("tool_calls"):
            # 忽略文本内容，只使用工具调用
            # 这是因为工具调用优先级更高，文本内容可能是说明性文字
            # 深拷贝工具调用信息
            action = deepcopy(struct_response["tool_calls"][0]['function'])
            
            # 如果 arguments 是字符串，解析为字典
            if isinstance(action['arguments'], str):
                # 容错处理：空字符串或 null 转为空字典
                if action['arguments'].strip() in ("", "null", "None"):
                    action['arguments'] = {}
                else:
                    action['arguments'] = json.loads(action['arguments'])

        # ========== 情况 4：两者都为空 ==========
        # 检查：content 和 tool_calls 都不存在或都为空
        else:
            # 这是错误情况：既没有工具调用，也没有文本内容
            # 无法确定 Agent 想要执行什么操作
            print("[parse_action] No valid content:", struct_response)
            # 返回解析失败
            return False, {}

        # 解析成功，返回 action
        return True, action

    except Exception as e:
        # 如果解析过程中出现任何异常（如类型错误、键不存在等）
        # 捕获异常并返回错误信息
        print(f"[parse_action] Parse error: {e}, raw data: {struct_response}")
        # 返回解析失败
        return False, {}

"""
用户代理实现模块

本模块实现了 UserAgent 类，用于在对话环境中模拟真实用户与 Action Agent 交互。

重要说明：
在用户代理的消息历史中：
- role = "user" 记录的是 Action Agent 的响应（标记为 [Agent]）
- role = "assistant" 记录的是 User Agent 的响应（包含 # Thought: 和 # Reply:）
"""
import re  # 正则表达式模块，用于解析响应格式
from copy import deepcopy  # 深拷贝工具，避免修改原始数据
from envscaler_env.utils.user_llm_inference import llm_inference  # 用户代理的 LLM 推理接口

# --------------------------------------------------------------------
# 用户代理系统提示词
# --------------------------------------------------------------------
user_system_prompt = \
"""You are a real human user interacting with an Agent assistant.  
Your current task is to have the Agent accomplish the following goal:  

[Task Goal]  
{task}  

**Core Principles:**  
- Do not directly or fully repeat the exact task instruction in your dialogue; instead, progress toward the goal gradually through multiple exchanges.  
- Deliver the task information in parts during the conversation so the Agent can slowly understand and move closer to the final objective.  
- When the task goal has been achieved, output a standalone message: `###STOP###` in your reply to end the dialogue. Do not include anything else.  

**Rules:**  
1. If the task contains multiple sub-tasks, do not reveal all of them at once; provide relevant sub-tasks one by one as the Agent asks.  
2. If completing the task requires multiple pieces of information, do not disclose them all at once; provide partial information in response to the Agent's questions.  
3. All requests must remain strictly within the scope of the task—do not add extra requirements, intentions, or invent information that was not part of the original task.  
4. Always keep the conversation focused on progressing toward the task, ensuring every sub-task or goal is covered and none are skipped.  

**Fidelity and Consistency Requirements:**  
- Always remain faithful to the original task wording throughout the conversation. Pay special attention to preserving exact **keywords, names, and proper nouns**—do not rephrase or alter them.  
- If the Agent assistant presents you with multiple options, only choose those that match the intent and constraints of the original task. If none fit, politely refuse and restate your requirement.  
- Do **not** introduce any new information that is not present in the original task description.  
- Do not repeat information you have already provided earlier in the conversation unless the Agent explicitly asks for clarification.  

**Style Requirements:**  
- Keep the dialogue natural and conversational, avoiding overly rigid or formal expressions.  

**Output Format (must be strictly followed):**  
# Thought:  
<Your thought process (this will NOT be sent to the Agent)> 
# Reply:  
<Your natural, conversational reply as the user, to be sent to the Agent>"""


class UserAgent:
    """
    用户代理类：模拟真实用户与 Action Agent 交互
    
    功能说明：
    - 使用 LLM 生成自然的用户回复
    - 管理用户代理的消息历史
    - 解析 LLM 响应（提取 # Thought: 和 # Reply: 部分）
    - 支持任务终止信号（###STOP###）
    
    消息历史格式：
    - role: "system" - 系统提示词（包含任务目标）
    - role: "user" - Action Agent 的响应（标记为 [Agent]）
    - role: "assistant" - User Agent 的响应（包含思考过程和回复）
    """
    
    def __init__(self, system_prompt, model, provider, api_key=None, base_url=None):
        """
        初始化用户代理
        
        参数：
            system_prompt (str): 系统提示词模板（包含 {task} 占位符）
            model (str): LLM 模型名称（如 "gpt-4.1"）
            provider (str): 模型提供商（如 "openai"）
            api_key (str, optional): API 密钥
            base_url (str, optional): API 基础 URL
        """
        # 消息历史（用于 LLM 推理）
        # 格式：OpenAI 消息格式列表
        # 初始为 None，在 get_init_reply() 时初始化
        self.messages = None
        
        # 对话记录（简化版本，仅记录用户和 Agent 的回复）
        # 格式：[{"user": "..."}, {"agent": "..."}, ...]
        # 初始为 None，在 get_init_reply() 时初始化
        self.conversations = None
        
        # LLM 模型名称（用于调用 LLM API）
        self.model = model
        
        # 系统提示词模板（包含 {task} 占位符，会在使用时格式化）
        self.system_prompt = system_prompt
        
        # 模型提供商（用于选择对应的 LLM 推理接口）
        self.provider = provider
        
        # API 密钥（如果提供则使用，否则从环境变量读取）
        self.api_key = api_key
        
        # API 基础 URL（如果提供则使用，否则从环境变量读取）
        self.base_url = base_url


    def get_init_reply(self, task):
        """
        基于任务获取用户的初始回复
        
        功能说明：
        - 初始化消息历史和对话记录
        - 调用 LLM 生成用户的初始回复
        - 解析响应，提取实际回复内容
        
        参数：
            task (str): 任务描述文本
        
        返回：
            str: 用户的初始回复文本（不包含思考过程）
        
        使用示例：
            reply = user_agent.get_init_reply("更新处方状态")
            # 返回："我需要更新处方 RX2024-005 的状态"
        """
        # 初始化对话记录列表（空列表）
        self.conversations = []
        
        # 初始化消息历史
        self.messages = [
            # 第一条消息：系统提示词
            # 使用 format() 将任务描述插入到提示词模板中
            {"role": "system", "content": self.system_prompt.format(task=task)},
            # 第二条消息：Action Agent 的初始问候
            # 这是模拟 Agent 主动打招呼，触发用户回复
            {"role": "user", "content": "[Agent] Hi! How can I help you today?"},
        ]
        
        # 调用 LLM 生成用户的初始回复
        # _infer() 返回原始响应和解析后的回复内容
        raw_response, user_content = self._infer()
        
        # 将 LLM 的完整响应（包含思考过程）添加到消息历史
        # 这样后续的对话可以基于完整的上下文
        self.messages.append({"role": "assistant", "content": raw_response})
        
        # 确保回复内容是字符串格式（虽然已经是字符串，但为了安全）
        user_content = f"{user_content}"
        
        # 将用户回复添加到对话记录
        self.conversations.append({"user": user_content})
        
        # 返回用户的初始回复（不包含思考过程）
        return user_content
        

    def user_step(self, agent_response):
        """
        处理 Agent 的响应并返回用户回复
        
        功能说明：
        - 接收 Action Agent 的响应
        - 将响应添加到消息历史
        - 调用 LLM 生成用户回复
        - 解析响应，提取实际回复内容
        
        参数：
            agent_response (str): Action Agent 的响应文本
                                  例如："我已经检查了库存，现在需要调整数量"
        
        返回：
            str: 用户的回复文本（不包含思考过程）
        
        使用示例：
            reply = user_agent.user_step("我已经检查了库存")
            # 返回："好的，请继续调整库存数量"
        """
        # 在 Agent 响应前添加 [Agent] 标记
        # 这样在消息历史中可以清楚区分是 Agent 的回复
        agent_response = f"[Agent] {agent_response}"
        
        # 将 Agent 的响应添加到消息历史（作为 user 角色）
        # 注意：在用户代理的视角中，Agent 的回复是 "user" 角色
        self.messages.append({"role": "user", "content": agent_response})
        
        # 将 Agent 响应添加到对话记录
        self.conversations.append({"agent": agent_response})
        
        # 调用 LLM 生成用户的回复
        # _infer() 会基于当前消息历史生成回复
        raw_response, user_content = self._infer()
        
        # 确保回复内容是字符串格式
        user_content = f"{user_content}"
        
        # 将 LLM 的完整响应添加到消息历史（作为 assistant 角色）
        # 这样后续对话可以基于完整的上下文
        self.messages.append({"role": "assistant", "content": raw_response})
        
        # 将用户回复添加到对话记录
        self.conversations.append({"user": user_content})
        
        # 返回用户的回复（不包含思考过程）
        return user_content
       
    
    def _infer(self):
        """
        从 LLM 推理用户回复（带重试机制）
        
        功能说明：
        - 调用 LLM API 生成用户回复
        - 解析响应（提取 # Reply: 部分）
        - 如果解析失败，最多重试 5 次
        
        返回：
            tuple: (raw_response, user_content)
                - raw_response (str): LLM 的完整原始响应（包含 # Thought: 和 # Reply:）
                - user_content (str): 解析后的用户回复（仅 # Reply: 部分）
        
        使用示例：
            raw, content = user_agent._infer()
            # raw = "# Thought:\n我需要...\n# Reply:\n请继续..."
            # content = "请继续..."
        """
        # 初始化重试计数器
        cur_try = 0
        # 最大重试次数（如果解析失败，最多重试 5 次）
        max_try = 5
        
        # 重试循环：最多尝试 max_try 次
        while cur_try < max_try:
            # 增加重试计数
            cur_try += 1
            
            # 调用 LLM 推理接口生成回复
            # llm_inference() 返回 LLM 的原始响应文本
            raw_response = llm_inference(
                model=self.model,        # 模型名称
                messages=self.messages,   # 消息历史（包含系统提示词和对话历史）
                provider=self.provider,   # 模型提供商
                api_key=self.api_key,     # API 密钥
                base_url=self.base_url    # API 基础 URL
            )
            
            # 解析响应，提取用户回复内容
            # _parse_response() 会提取 # Reply: 部分
            parse_success, user_content = self._parse_response(raw_response)
            
            # 如果解析成功，跳出循环
            if parse_success:
                break
        
        # 返回原始响应和解析后的用户内容
        return raw_response, user_content
    
    def _parse_response(self, text: str):
        """
        解析包含 # Thought: 和 # Reply: 的响应
        
        功能说明：
        - 检查是否包含终止信号（###STOP###）
        - 提取 # Thought: 和 # Reply: 部分
        - 返回解析后的回复内容
        
        参数：
            text (str): LLM 的原始响应文本
                        例如："# Thought:\n我需要...\n# Reply:\n请继续..."
        
        返回：
            tuple: (parse_success, reply_content)
                - parse_success (bool): 解析是否成功
                - reply_content (str): 用户回复内容（# Reply: 部分）或 "###STOP###"
        
        使用示例：
            success, content = user_agent._parse_response("# Thought:\n...\n# Reply:\n你好")
            # success = True, content = "你好"
        """
        # 首先检查是否包含终止信号
        # 如果用户代理输出 "###STOP###"，表示任务完成，对话应该结束
        if "###STOP###" in text:
            # 返回成功和终止信号
            return True, "###STOP###"
        
        # 使用正则表达式提取 # Thought: 和 # Reply: 部分
        # 编译正则表达式模式（提高性能，因为可能多次使用）
        pattern = re.compile(
            # 正则表达式说明：
            # # Thought:\s*  - 匹配 "# Thought:" 后跟零个或多个空白字符
            # (.*?)          - 非贪婪匹配思考内容（捕获组 1）
            # \s*            - 匹配零个或多个空白字符
            # # Reply:\s*    - 匹配 "# Reply:" 后跟零个或多个空白字符
            # (.*)           - 贪婪匹配回复内容（捕获组 2，匹配到文本末尾）
            r'# Thought:\s*(.*?)\s*# Reply:\s*(.*)',
            re.DOTALL  # DOTALL 标志：. 匹配换行符（因为内容可能跨多行）
        )
        
        # 在文本中搜索匹配项
        match = pattern.search(text)
        
        # 如果找到匹配项
        if match:
            # 提取思考内容（捕获组 1）
            # .strip() 去除首尾空白字符
            thought_content = match.group(1).strip()
            
            # 提取回复内容（捕获组 2）
            # .strip() 去除首尾空白字符
            reply_content = match.group(2).strip()
            
            # 返回成功和回复内容（思考内容不需要返回，因为不会发送给 Agent）
            return True, reply_content
        else:
            # 如果解析失败（没有找到 # Thought: 和 # Reply: 部分）
            # 打印错误信息以便调试
            print(f"Parsed response failed: {text}")
            # 返回失败和空字符串
            return False, ""
        
    def get_messages(self):
        """
        返回消息历史的深拷贝
        
        功能说明：
        - 返回用户代理的完整消息历史
        - 使用深拷贝避免外部修改影响内部数据
        
        返回：
            list: 消息历史列表（深拷贝）
                  格式：[{"role": "system", "content": "..."}, ...]
        
        使用示例：
            messages = user_agent.get_messages()
            # 返回完整的消息历史，用于保存到结果文件
        """
        # 返回消息历史的深拷贝
        # 深拷贝确保外部对返回列表的修改不会影响内部的消息历史
        return deepcopy(self.messages)

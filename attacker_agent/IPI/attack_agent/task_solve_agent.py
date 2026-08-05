"""
任务解决 Agent 模块 - 在交互式环境中使用 LLM 推理解决任务

本模块实现了 TaskSolveAgent 类，这是整个 Agent 系统的核心。
它负责：
1. 管理 Agent 与环境的交互循环
2. 调用 LLM 生成动作
3. 执行动作并获取环境反馈
4. 记录完整的执行轨迹

工作流程：
1. reset() - 重置环境，初始化消息历史
2. step() - 执行一步交互（LLM 推理 → 执行动作 → 更新状态）
3. run() - 运行完整任务（循环执行 step() 直到结束）
"""

from copy import deepcopy  # 深拷贝工具，用于复制对象避免引用问题
from .system_prompt_util import (
    injection_attack_system_prompt,                   # 注入攻击的系统提示词（非对话）
    injection_attack_conversational_system_prompt,     # 注入攻击的系统提示词（对话）
    merge_tools_into_system_prompt                     # 将工具信息合并到系统提示词
)
from .agent_llm_inference import (
    llm_inference_fc,      # FC 模式的 LLM 推理接口
    llm_inference_prompt   # Prompt 模式的 LLM 推理接口
)


class TaskSolveAgent:
    """
    任务解决 Agent 类
    
    功能：在交互式环境中使用 LLM 推理解决任务。
    这是 Agent 系统的核心类，负责管理完整的任务执行流程。
    
    主要职责：
    - 管理 Agent 的完整生命周期
    - 处理环境交互循环
    - 记录轨迹和执行状态
    - 协调提示词和 LLM 调用
    
    工作模式：
    - Prompt 模式：通过文本提示词进行工具调用，需要解析文本
    - FC 模式：使用 LLM 原生的函数调用接口，返回结构化响应
    """
    
    def __init__(
        self,
        env_name,          # 环境名称（用于选择系统提示词）
        env,               # 环境实例对象
        model,             # LLM 模型名称（如 "gpt-4.1-mini"）
        provider,          # 模型提供商（如 "openai"）
        temperature,       # 温度参数（控制输出的随机性）
        infer_mode,        # 推理模式（"prompt" 或 "fc"）
        max_steps,         # 最大执行步数
        enable_thinking,   # 是否启用思考模式
        api_key=None,      # API 密钥（可选，从环境变量读取）
        base_url=None      # API 基础 URL（可选，从环境变量读取）
    ):
        """
        初始化 TaskSolveAgent
        
        参数说明：
            env_name (str): 环境名称，用于选择系统提示词和任务信息提取方式
            env (object): 环境实例，必须实现 reset() 和 step() 方法
            model (str): LLM 模型名称
            provider (str): 模型提供商
            temperature (float): 温度参数，范围通常为 0-2
            infer_mode (str): 推理模式
                - "prompt": 通过提示词进行工具调用
                - "fc": 使用函数调用接口
            max_steps (int): 最大执行步数，防止无限循环
            enable_thinking (bool): 是否启用思考模式（仅支持特定模型）
            api_key (str, optional): API 密钥
            base_url (str, optional): API 基础 URL
        """
        
        # ========== 保存环境信息 ==========
        self.env_name = env_name  # 环境名称（用于后续判断环境类型）
        self.env = env            # 环境实例（用于调用 reset() 和 step()）

        # ========== LLM 配置参数 ==========
        self.model = model                    # 模型名称
        self.provider = provider              # 提供商（如 "openai"）
        self.temperature = temperature        # 温度参数
        self.api_key = api_key                # API 密钥
        self.base_url = base_url              # API 基础 URL
        
        # ========== 验证推理模式 ==========
        # 确保推理模式是有效的值
        # prompt: 通过提示词进行工具调用（Agent 需要自己解析文本）
        # fc: 使用函数调用接口（LLM 返回结构化对象）
        assert infer_mode in ["prompt", "fc"], \
            f"infer_mode must be 'prompt' or 'fc', got '{infer_mode}'"
        self.infer_mode = infer_mode
        
        # ========== 思考模式配置 ==========
        self.enable_thinking = enable_thinking  # 是否启用思考模式

        # ========== 运行时配置 ==========
        self.max_steps = max_steps  # 最大执行步数（防止无限循环）

        # ========== 运行时状态变量 ==========
        # 这些变量会在执行过程中不断更新
        self.messages = []              # 对话历史（用于 LLM 推理）
        self.current_observation = None # 当前观察（环境返回的最新状态）
        self.current_info = None        # 当前信息（环境返回的额外信息）
        self.terminated = False         # 是否正常终止（任务完成）
        self.truncated = False          # 是否被截断（达到最大步数或其他限制）
        self.step_count = 0             # 当前步数计数器

        # ========== 轨迹记录 ==========
        # 用于记录完整的执行过程，包括每一步的动作、观察、奖励等
        self.trajectory = []

    def reset(self, task_index=None):
        """
        重置环境并初始化 Agent 状态
        
        功能：
        1. 重置环境并获取初始观察
        2. 构建系统提示词（根据环境类型选择）
        3. 合并工具信息到提示词（Prompt 模式）
        4. 初始化消息历史和轨迹
        
        参数：
            task_index (int, optional): 任务索引，如果为 None 则随机选择
        
        返回：
            tuple: (observation, info)
                - observation: 初始观察（通常是任务描述）
                - info: 环境信息（包含工具列表、环境介绍等）
        
        执行流程：
            1. 调用 env.reset() 重置环境
            2. 获取工具列表和环境介绍
            3. 根据环境类型选择系统提示词
            4. 添加环境介绍到提示词
            5. Prompt 模式：合并工具信息
            6. 提取任务信息
            7. 初始化消息历史和轨迹
        """
        
        # ========== 重置环境 ==========
        # 调用环境的 reset() 方法，获取初始观察和环境信息
        # observation: 初始观察（通常是任务描述文本）
        # info: 包含工具列表、环境介绍、任务信息等
        observation, info = self.env.reset(task_index=task_index)

        # ========== 获取工具列表 ==========
        # 从环境信息中提取工具列表
        # tools: Agent 可以使用的工具函数列表
        self.tools = info["tools"]
        # user_tools: 用户代理可以使用的工具（对话环境）
        self.user_tools = info.get("user_tools", [])

        # ========== 选择系统提示词 ==========
        # 本 IPI attack_agent 专门用于注入攻击任务
        # 对话环境：attack_conv_env
        # 非对话环境：attack_non_conv_env
        # 使用注入攻击系统提示词
        if self.env_name == "attack_conv_env":
            system_prompt = injection_attack_conversational_system_prompt
        elif self.env_name == "attack_non_conv_env":
            system_prompt = injection_attack_system_prompt
        else:
            raise RuntimeError(f"Unknown attack env_name for attack agent: {self.env_name}")
        
        # ========== 添加环境介绍 ==========
        # 如果环境提供了介绍信息，将其添加到系统提示词中
        # 环境介绍通常包含环境的用途、规则、约束等信息
        if "env_introduction" in info and info["env_introduction"]:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"The following is an introduction to the current environment:\n"
                f"{info['env_introduction']}"
            )
        
        # ========== Prompt 模式：合并工具信息 ==========
        # 在 Prompt 模式下，需要将工具信息合并到系统提示词中
        # 因为 LLM 需要通过提示词了解有哪些工具可用
        # FC 模式不需要这一步，因为工具信息通过 API 参数传递
        if self.infer_mode == "prompt":
            # 调用工具合并函数，将工具列表格式化为提示词格式
            system_prompt = merge_tools_into_system_prompt(
                system_prompt=system_prompt, 
                tools=info["tools"]
            )

        # ========== 提取任务信息 ==========
        # 攻击环境的任务信息格式统一为 dict，包含 env_id / task_id / task 等
        task_item = deepcopy(info["task"])  # 深拷贝避免修改原始数据
        if self.env_name in [
            "attack_non_conv_env", 
            "attack_conv_env",
        ]:
            self.task_info = {
                "env_id": task_item["env_id"],
                "task_id": task_item["task_id"],
                "task": task_item["task"],
                "is_injection_attack": True,  # 所有任务都是注入攻击任务
            }
        else:
            raise RuntimeError(f"Unknown attack env_name for attack agent: {self.env_name}")

        # ========== 初始化消息历史 ==========
        # 消息历史用于 LLM 推理，包含完整的对话上下文
        # 第一条消息是系统提示词，告诉 LLM 如何工作
        self.messages = [{"role": "system", "content": system_prompt}]
        
        # ========== 初始化运行时状态 ==========
        self.current_observation = deepcopy(observation)  # 当前观察（深拷贝）
        self.current_info = deepcopy(info)                  # 当前信息（深拷贝）
        self.terminated = False                             # 终止标志归零
        self.truncated = False                              # 截断标志归零
        self.user_messages = None                           # 用户消息（对话环境）
        self.step_count = 0                                 # 步数计数器归零
        self.trajectory = []                                # 轨迹记录清空

        # ========== 添加初始观察到消息历史 ==========
        # 初始观察通常是任务描述，作为第一条用户消息
        # 这样 LLM 就知道要完成什么任务
        self.messages.append({"role": "user", "content": observation})

        # ========== 记录初始轨迹 ==========
        # 轨迹记录用于保存完整的执行过程，便于后续分析和训练
        self.trajectory.append({
            "step": self.step_count,      # 步数（初始为 0）
            "observation": observation,   # 初始观察
            # "info": info,               # 信息（已注释，不记录）
        })

        # 返回初始观察和信息
        return observation, info

    def step(self):
        """
        执行一步环境交互
        
        功能：
        1. 调用 LLM 生成响应（动作）
        2. 将响应传递给环境执行
        3. 更新消息历史、奖励、轨迹等
        
        执行流程：
            1. 检查环境是否已结束
            2. 调用 LLM 推理生成响应
            3. 解析响应并添加到消息历史
            4. 执行环境步骤
            5. 更新内部状态
            6. 添加观察到消息历史
            7. 记录轨迹
        
        返回：
            tuple: (observation, terminated, truncated, info, action)
                - observation: 环境返回的观察
                - terminated: 是否正常终止
                - truncated: 是否被截断
                - info: 额外信息（包含执行的动作）
                - action: 执行的动作
        
        异常：
            RuntimeError: 如果环境已经结束，再次调用 step() 会抛出异常
        """

        # ========== 检查环境状态 ==========
        # 如果环境已经结束（正常终止或被截断），不能再执行步骤
        # 需要先调用 reset() 重置环境
        if self.terminated or self.truncated:
            raise RuntimeError(
                "Environment already finished. "
                "Please reset before calling step again."
            )

        # ========== 调用 LLM 推理 ==========
        # 根据推理模式选择不同的 LLM 调用方式
        if self.infer_mode == "prompt":
            # ========== Prompt 模式 ==========
            # 通过提示词进行工具调用
            # LLM 返回文本响应，需要环境解析提取工具调用
            raw_response = llm_inference_prompt(
                provider=self.provider,        # 提供商
                model=self.model,              # 模型名称
                messages=self.messages,        # 对话历史
                temperature=self.temperature,   # 温度参数
                enable_thinking=self.enable_thinking,  # 思考模式
                api_key=self.api_key,          # API 密钥
                base_url=self.base_url         # API 基础 URL
            )
            
            # ========== 处理思考内容 ==========
            # 如果响应中包含思考标签，只保留标签后的实际响应
            # 思考内容已经在消息历史中，不需要再次传递给环境
            if "</think>" in raw_response:
                # 分割响应，只取标签后的部分
                raw_response = raw_response.split("</think>")[-1].strip()
        else:
            # ========== FC 模式 ==========
            # 使用函数调用接口
            # LLM 返回结构化响应，包含 tool_calls、content 等
            raw_response = llm_inference_fc(
                provider=self.provider,        # 提供商
                model=self.model,              # 模型名称
                messages=self.messages,        # 对话历史
                temperature=self.temperature,   # 温度参数
                tools=self.tools,              # 工具列表（FC 模式必需）
                enable_thinking=self.enable_thinking,  # 思考模式
                api_key=self.api_key,          # API 密钥
                base_url=self.base_url         # API 基础 URL
            )

        # ========== 添加 LLM 响应到消息历史 ==========
        # 将 LLM 的响应添加到消息历史中，用于后续的上下文
        if self.infer_mode == "prompt":
            # Prompt 模式：响应是文本字符串
            message = {"role": "assistant", "content": raw_response}
        else:
            # FC 模式：响应是字典，包含多个字段
            message = {"role": "assistant", "content": raw_response["content"]}
            # 如果有工具调用，添加到消息中
            if raw_response["tool_calls"]:
                message["tool_calls"] = raw_response["tool_calls"]
            # 如果有思考内容，添加到消息中
            if raw_response["reasoning_content"]:
                message["reasoning_content"] = raw_response["reasoning_content"]
        
        # 将消息添加到历史记录
        self.messages.append(message)

        # ========== 执行环境步骤 ==========
        # 将 LLM 的响应作为动作传递给环境执行
        # 注意：此检查主要针对 Prompt 模式（返回字符串）
        # FC 模式返回字典，如果失败会返回 {"reasoning_content": "", "tool_calls": [], "content": ""}
        if raw_response == '':
            # ========== 处理空响应 ==========
            # 如果 LLM 返回空响应（Prompt 模式），说明可能有问题
            # 创建一个错误观察，并标记为终止
            print("raw_response is empty, please check the model")
            observation = {"type": "user", "content": "action is empty, please check the model"}
            terminated = True
            truncated = True
            info = {"action": ""}
            action = ""  # 定义 action 变量（空字符串），避免后续使用时报错
        else:
            # ========== 正常执行 ==========
            # 调用环境的 step() 方法执行动作
            # 环境会解析动作、执行工具调用、返回观察
            observation, terminated, truncated, info = self.env.step(
                action=raw_response
            )
            # 从环境信息中提取执行的动作（用于轨迹记录）
            action = info["action"]

        # ========== 更新内部状态 ==========
        self.step_count += 1                    # 步数加 1
        self.current_observation = observation   # 更新当前观察
        self.current_info = info                # 更新当前信息
        self.terminated = bool(terminated)       # 更新终止标志
        self.truncated = bool(truncated)         # 更新截断标志

        # ========== 添加观察到消息历史 ==========
        # 将环境的观察添加到消息历史，作为下一条用户消息
        if observation["type"] == "tool":
            # ========== 工具响应 ==========
            # 观察类型为 "tool"，说明是工具执行的结果
            if self.infer_mode == "prompt":
                # ========== Prompt 模式 ==========
                # Prompt 模式不支持 tool 角色，需要格式化为文本
                # 使用 Qwen3 工具响应模板格式
                observation_content = (
                    f"<tool_response>\n"
                    f"{observation['content']}\n"
                    f"</tool_response>"
                )
                # 作为用户消息添加
                self.messages.append({"role": "user", "content": observation_content})
            else:
                # ========== FC 模式 ==========
                # FC 模式支持 tool 角色，可以直接使用
                if len(raw_response["tool_calls"]) < 1:
                    # ========== 异常处理 ==========
                    # 如果工具调用列表为空，说明有问题
                    import pdb; pdb.set_trace()  # 调试断点
                    print("!!!!!! raw_response['tool_calls'] is empty, please check the model")
                    # 降级处理：作为普通用户消息添加
                    self.messages.append({"role": "user", "content": observation['content']})
                else:
                    # ========== 正常处理 ==========
                    # 提取工具调用信息，用于关联工具响应
                    # TODO: 支持多个函数调用（当前只支持一个）
                    tool_call_id = raw_response["tool_calls"][0]['id']      # 工具调用 ID
                    tool_call_name = raw_response["tool_calls"][0]['function']['name']  # 工具名称
                    # 使用 tool 角色添加消息，包含 tool_call_id 用于关联
                    self.messages.append({
                        "role": "tool", 
                        "tool_call_id": tool_call_id,  # 关联对应的工具调用
                        "name": tool_call_name,        # 工具名称
                        "content": observation['content']  # 工具执行结果
                    })
        else:
            # ========== 用户消息 ==========
            # 观察类型为 "user"，说明是用户的消息（对话环境）
            # 直接作为用户消息添加
            self.messages.append({"role": "user", "content": observation["content"]})

        # ========== 记录轨迹 ==========
        # 将本次步骤的完整信息记录到轨迹中
        self.trajectory.append({
            "step": self.step_count,      # 步数
            "action": action,             # 执行的动作
            "observation": observation,   # 环境返回的观察
            "terminated": terminated,     # 是否正常终止
            "truncated": truncated,       # 是否被截断
            # "info": info,               # 额外信息（已注释，不记录）
        })
        
        # ========== 保存用户消息 ==========
        # 如果是对话环境，保存用户消息（用于后续分析）
        if "user_messages" in info:
            self.user_messages = info["user_messages"]
        
        # 返回本次步骤的所有信息
        return observation, terminated, truncated, info, action

    def run(self, task_index=None, max_steps=None):
        """
        运行完整任务
        
        功能：
        1. 重置环境
        2. 循环执行 step() 直到任务结束
        3. 聚合所有结果并返回
        
        执行流程：
            1. 重置环境（调用 reset()）
            2. 循环执行 step() 直到：
               - terminated = True（正常完成）
               - truncated = True（被截断）
               - step_count >= max_steps（达到最大步数）
            3. 聚合所有结果
        
        参数：
            task_index (int, optional): 任务索引，如果为 None 则随机选择
            max_steps (int, optional): 最大步数，如果为 None 则使用初始化时的值
        
        返回：
            dict: 包含以下键的字典
                - task_info: 任务信息
                - tools: 可用工具列表
                - messages: 完整对话历史
                - user_messages: 用户消息（对话环境）
                - trajectory: 详细执行轨迹
                - terminated: 是否正常终止
                - truncated: 是否被截断
                - final_observation: 最终观察
                - final_info: 最终信息
                - steps: 总步数
        
        使用示例：
            agent = TaskSolveAgent(...)
            result = agent.run(task_index=0)
            print(f"任务完成: {result['terminated']}")
            print(f"总步数: {result['steps']}")
        """
        
        # ========== 确定最大步数 ==========
        # 如果传入了 max_steps，使用传入的值；否则使用初始化时的值
        max_steps = max_steps if max_steps is not None else self.max_steps

        # ========== 初始化环境 ==========
        # 重置环境并初始化 Agent 状态
        # 这会设置系统提示词、工具列表、消息历史等
        self.reset(task_index=task_index)

        # ========== 执行循环 ==========
        # 循环执行步骤，直到满足终止条件
        # 终止条件：
        # 1. terminated = True: 任务正常完成
        # 2. truncated = True: 任务被截断（如达到某些限制）
        # 3. step_count >= max_steps: 达到最大步数（防止无限循环）
        while (not self.terminated) and (not self.truncated) and (self.step_count < max_steps):
            # 执行一步交互
            self.step()
 
        # ========== 聚合结果 ==========
        # 将所有执行结果聚合到一个字典中，便于返回和分析
        result = {
            "task_info": self.task_info,              # 任务信息（env_id, task_id, task 等）
            "tools": self.tools,                      # 可用工具列表
            "messages": self.messages,                # 完整对话历史（用于训练和分析）
            "user_messages": self.user_messages,       # 用户消息（对话环境）
            "trajectory": self.trajectory,            # 详细执行轨迹（每一步的动作、观察等）
            "terminated": self.terminated,            # 是否正常终止
            "truncated": self.truncated,              # 是否被截断
            "final_observation": self.current_observation,  # 最终观察（最后一步的观察）
            "final_info": self.current_info,          # 最终信息（最后一步的信息）
            "steps": self.step_count,                 # 总步数（实际执行的步数）
        }

        # 返回完整结果
        return result

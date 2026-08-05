"""
对话 RL 环境模块

本模块实现了 EnvScalerConvRLEnv 类，用于对话模式的强化学习环境。

特点：
- 使用 UserAgent 模拟用户交互
- 用户主动终止（输出 "###STOP###"）
- 支持奖励计算（基于 check_func）
- 有用户代理（user_messages 包含完整对话历史）
"""
from envscaler_env.utils.user_agent import UserAgent, user_system_prompt  # 导入用户代理类和系统提示词
from .base_env import EnvScalerBaseEnv  # 导入基础环境类


class EnvScalerConvRLEnv(EnvScalerBaseEnv):
    """
    对话 RL 环境类
    
    功能说明：
    - 继承自 EnvScalerBaseEnv，复用所有基础功能
    - 初始化 UserAgent 用于模拟用户交互
    - 实现对话模式的特定方法：
      - get_initial_observation(): 从用户代理获取初始回复
      - is_action_terminated(): 对话模式不依赖 action 终止
      - is_observation_terminated(): 判断用户是否输出 "###STOP###"
    
    使用场景：
    - 对话式 RL 训练
    - 多轮交互任务
    - 需要用户反馈的任务
    """
    
    def __init__(self, mode, user_model, provider, env_items_path=None, task_items_path=None, api_key=None, base_url=None):
        """
        初始化对话 RL 环境
        
        参数：
            mode (str): 模式，必须是 "train" 或 "eval"
            user_model (str): 用户代理使用的 LLM 模型名称（如 "gpt-4.1"）
            provider (str): 模型提供商（如 "openai"）
            env_items_path (str, optional): 环境数据文件路径
            task_items_path (str, optional): 任务数据文件路径
            api_key (str, optional): API 密钥（如果提供则使用，否则从环境变量读取）
            base_url (str, optional): API 基础 URL（如果提供则使用，否则从环境变量读取）
        
        使用示例：
            env = EnvScalerConvRLEnv(
                mode="train",
                user_model="gpt-4.1",
                provider="openai",
                ...
            )
        """
        # ========== 初始化用户代理 ==========
        # 创建 UserAgent 实例，用于模拟用户与 Action Agent 交互
        self.user_agent = UserAgent(
            system_prompt=user_system_prompt,  # 用户代理的系统提示词（包含任务目标）
            model=user_model,                   # LLM 模型名称
            provider=provider,                  # 模型提供商
            api_key=api_key,                   # API 密钥
            base_url=base_url                  # API 基础 URL
        )
        
        # 调用父类的初始化方法
        # 这会加载任务数据集和环境数据集，并初始化所有基础属性
        super().__init__(mode=mode, env_items_path=env_items_path, task_items_path=task_items_path)

    def get_initial_observation(self, task_item: dict):
        """
        获取初始观察（从用户代理获取初始回复）
        
        功能说明：
        - 对话模式下，初始观察是用户代理生成的初始回复
        - 用户代理会根据任务描述生成自然的用户初始对话
        
        参数：
            task_item (dict): 任务项字典
                {
                    "task_id": str,
                    "task": str,        # 任务描述文本
                    "env_id": int,
                    ...
                }
        
        返回：
            str: 用户代理的初始回复文本
        
        使用示例：
            observation = self.get_initial_observation(task_item)
            # 返回："我需要更新处方 RX2024-005 的状态"
        """
        # 调用用户代理的 get_init_reply() 方法
        # 传入任务描述，用户代理会生成初始回复
        # 返回的回复不包含思考过程，只包含实际回复内容
        return self.user_agent.get_init_reply(task=task_item['task'])

    def is_action_terminated(self, action: dict):
        """
        判断 action 是否为终止信号
        
        功能说明：
        - 对话模式下，不依赖 action 来终止任务
        - 终止完全由用户代理控制（输出 "###STOP###"）
        - 因此始终返回 False
        
        参数：
            action (dict): action 字典（未使用，但为了接口一致性保留）
        
        返回：
            bool: 始终返回 False（对话模式不依赖 action 终止）
        
        使用示例：
            is_terminated = self.is_action_terminated(action)
            # 始终返回 False
        """
        # 对话模式不依赖 action 来终止任务
        # 终止完全由用户代理控制（通过 observation 中的 "###STOP###"）
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """
        判断 observation 是否为终止信号
        
        功能说明：
        - 对话模式下，用户代理通过输出 "###STOP###" 来终止任务
        - 检查条件：
          1. action 必须是 "chat_with_user"（Agent 与用户交互）
          2. observation 内容中必须包含 "###STOP###"
        
        参数：
            action (dict): action 字典
                {
                    "name": "chat_with_user",
                    "arguments": {"content": "..."}
                }
            observation (str | dict): observation
                - 如果是字典：{"type": "user", "content": "###STOP###"}
                - 如果是字符串："###STOP###"
        
        返回：
            bool: 是否终止
                - True: action 是 "chat_with_user" 且 observation 包含 "###STOP###"
                - False: 其他情况
        
        使用示例：
            is_terminated = self.is_observation_terminated(action, observation)
        """
        # 判断是否终止：
        # 1. action 名称必须是 "chat_with_user"（Agent 与用户交互）
        # 2. observation 内容中必须包含 "###STOP###"（用户终止信号）
        # str(observation) 确保即使 observation 是字典也能正确检查
        return action.get("name") == "chat_with_user" and "###STOP###" in str(observation)

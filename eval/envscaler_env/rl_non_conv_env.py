"""
非对话 RL 环境模块

本模块实现了 EnvScalerNonConvRLEnv 类，用于非对话模式的强化学习环境。

特点：
- Agent 直接接收任务描述并执行
- Agent 主动终止（输出 "Task Completed"）
- 支持奖励计算（基于 check_func）
- 无用户代理（user_messages = null）
"""
from .base_env import EnvScalerBaseEnv  # 导入基础环境类


class EnvScalerNonConvRLEnv(EnvScalerBaseEnv):
    """
    非对话 RL 环境类
    
    功能说明：
    - 继承自 EnvScalerBaseEnv，复用所有基础功能
    - 实现非对话模式的特定方法：
      - get_initial_observation(): 返回任务描述作为初始观察
      - is_action_terminated(): 判断 Agent 是否输出 "Task Completed"
      - is_observation_terminated(): 非对话模式不需要 observation 终止
    
    使用场景：
    - RL 训练数据收集
    - 单轮多步任务执行
    - 需要精确奖励信号的任务
    """
    
    def __init__(self, mode, env_items_path=None, task_items_path=None):
        """
        初始化非对话 RL 环境
        
        参数：
            mode (str): 模式，必须是 "train" 或 "eval"
            env_items_path (str, optional): 环境数据文件路径
            task_items_path (str, optional): 任务数据文件路径
        
        使用示例：
            env = EnvScalerNonConvRLEnv(mode="train", ...)
        """
        # 调用父类的初始化方法
        # 这会加载任务数据集和环境数据集，并初始化所有基础属性
        super().__init__(mode=mode, env_items_path=env_items_path, task_items_path=task_items_path)
        
    def get_initial_observation(self, task_item: dict):
        """
        获取初始观察（返回任务描述）
        
        功能说明：
        - 非对话模式下，初始观察就是任务描述文本
        - Agent 直接接收任务描述并开始执行
        
        参数：
            task_item (dict): 任务项字典
                {
                    "task_id": str,
                    "task": str,        # 任务描述文本
                    "env_id": int,
                    ...
                }
        
        返回：
            str: 任务描述文本
        
        使用示例：
            observation = self.get_initial_observation(task_item)
            # 返回："更新处方 RX2024-005 的状态为已完成"
        """
        # 返回任务描述文本作为初始观察
        # f"{task_item['task']}" 确保返回字符串类型
        return f"{task_item['task']}"

    def is_action_terminated(self, action: dict):
        """
        判断 action 是否为终止信号
        
        功能说明：
        - 非对话模式下，Agent 通过输出 "Task Completed" 来终止任务
        - 如果 action 是 "chat_with_user"，则认为是终止信号
        - 检查 action 内容中是否包含 "Task Completed"（警告，但不影响终止判断）
        
        参数：
            action (dict): action 字典
                {
                    "name": "chat_with_user",
                    "arguments": {"content": "Task Completed"}
                }
        
        返回：
            bool: 是否终止
                - True: action 是 "chat_with_user"（表示任务完成）
                - False: action 是工具调用（继续执行）
        
        使用示例：
            is_terminated = self.is_action_terminated(action)
        """
        # 如果 action 名称是 "chat_with_user"，表示 Agent 想要终止任务
        if action["name"] == "chat_with_user":
            # 检查 action 内容中是否包含 "Task Completed"
            # 如果不包含，打印警告（但不影响终止判断）
            # 这是因为某些情况下 Agent 可能只输出文本，没有明确标记 "Task Completed"
            if 'Task Completed' not in action["arguments"]['content']:
                print('warning: Task Completed not in action["arguments"]["content"]')
            # 返回 True，表示任务终止
            return True
        
        # 如果 action 不是 "chat_with_user"（是工具调用），返回 False，继续执行
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """
        判断 observation 是否为终止信号
        
        功能说明：
        - 非对话模式下，不需要通过 observation 来终止
        - 终止完全由 Agent 的 action 控制（"Task Completed"）
        
        参数：
            action (dict): action 字典（未使用，但为了接口一致性保留）
            observation (str | dict): observation（未使用，但为了接口一致性保留）
        
        返回：
            bool: 始终返回 False（非对话模式不需要 observation 终止）
        
        使用示例：
            is_terminated = self.is_observation_terminated(action, observation)
            # 始终返回 False
        """
        # 非对话模式不需要 observation 终止
        # 终止完全由 Agent 的 action 控制
        return False

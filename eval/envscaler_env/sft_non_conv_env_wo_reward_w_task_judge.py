"""
非对话 SFT 环境模块（带任务判断）

本模块实现了 EnvScalerNonConvSFTEnv 类，用于非对话模式的监督微调环境。

特点：
- 不继承基类（独立实现）
- 不计算奖励（固定返回 0.0，用于成本优化）
- 支持任务失败判断（"Task Failed"）
- 无用户代理（user_messages = null）
- Agent 主动终止（输出 "Task Completed" 或 "Task Failed"）

与 RL 环境的区别：
1. 为了成本效率，SFT 任务不合成验证函数（check_func）
2. 轨迹收集仅通过 LLM 自判断（"Task Failed"）和格式过滤
3. 不需要执行 check_func 来计算奖励
"""
import os  # 用于文件路径操作
import json  # 用于 JSON 文件读写
import random  # 用于随机选择任务
import traceback  # 用于捕获和格式化异常信息
from copy import deepcopy  # 深拷贝工具，避免修改原始数据

# 导入环境工具函数
from envscaler_env.utils.env_util import (
    init_env_class,        # 从源代码字符串动态创建环境类
    init_env_instance,    # 创建环境实例并应用配置
    get_state_diff,       # 比较两个状态字典的差异
    get_state_info,       # 获取环境实例的状态字典
)
# 导入解析工具函数
from envscaler_env.utils.parse_util import parse_response, parse_action




class EnvScalerNonConvSFTEnv:
    """
    非对话 SFT 环境类
    
    功能说明：
    - 不继承基类，独立实现所有功能
    - 管理任务数据集和环境数据集的加载
    - 提供 reset/step 工作流程
    - 记录轨迹（不计算奖励，固定返回 0.0）
    - 支持任务失败判断（"Task Failed"）
    
    与 RL 环境的区别：
    - 不计算奖励（calculate_reward() 固定返回 0.0）
    - 支持 "Task Failed" 标记（用于数据过滤）
    - 不执行 check_func（节省成本）
    """

    def __init__(self, mode, env_items_path=None, task_items_path=None):
        """
        初始化非对话 SFT 环境
        
        参数：
            mode (str): 模式，必须是 "train" 或 "eval"
            env_items_path (str, optional): 环境数据文件路径
            task_items_path (str, optional): 任务数据文件路径
        
        使用示例：
            env = EnvScalerNonConvSFTEnv(mode="train", ...)
        """
        # 保存模式（train 或 eval）
        self.mode = mode

        # ========== 加载任务数据集 ==========
        # 如果提供了任务数据文件路径，直接从该文件加载
        if task_items_path is not None:
            # 打开并读取 JSON 文件
            self.task_items = json.load(open(task_items_path, encoding="utf-8"))
            # 打印加载信息
            print(f"Ignore the mode {self.mode}.\nLoad task_items from {task_items_path}, total {len(self.task_items)} tasks!")
        else:
            # 如果没有提供路径，根据 mode 从默认路径加载
            self.task_items = self.load_task_items()
        
        # ========== 加载环境数据集 ==========
        # 如果提供了环境数据文件路径，直接从该文件加载
        if env_items_path is not None:
            # 打开并读取 JSON 文件
            self.env_items = json.load(open(env_items_path, encoding="utf-8"))
            # 打印加载信息
            print(f"Load env_items from {env_items_path}, total {len(self.env_items)} envs!")
        else:
            # 如果没有提供路径，从默认路径加载
            self.env_items = self.load_env_items()

        # ========== 初始化日志和环境状态 ==========
        # 重置所有类属性（轨迹、环境实例等）
        self.reset_attributes()

    # ==============================
    # 数据加载方法
    # ==============================

    def load_env_items(self):
        """
        加载环境数据集（从默认路径）
        
        功能说明：
        - 从 data 文件夹加载环境元数据文件
        - SFT 环境使用特定的环境数据文件
        
        返回：
            list: 环境数据列表
        
        使用示例：
            env_items = self.load_env_items()
        """
        # 获取 data 文件夹路径
        folder_path = os.path.join(os.path.dirname(__file__), "data")
        
        # 构建环境数据文件路径
        # SFT 环境使用特定的文件名：env_v1_85_brief.json
        env_items_path = os.path.join(folder_path, "env_v1_85_brief.json")
        
        # 打开文件并读取 JSON 数据
        with open(env_items_path, encoding="utf-8") as f:
            env_items = json.load(f)

        # 打印加载信息
        print(f"Load {len(env_items)} envs from {env_items_path}!")
        
        # 返回环境数据列表
        return env_items


    def load_task_items(self):
        """
        加载任务数据集（根据 mode 从默认路径加载）
        
        功能说明：
        - 根据 mode（train/eval）选择对应的任务数据文件
        - SFT 环境使用特定的任务数据文件
        
        返回：
            list: 任务数据列表
        
        异常：
            ValueError: 如果 mode 不是 "eval" 或 "train"
        
        使用示例：
            task_items = self.load_task_items()
        """
        # 获取 data 文件夹路径
        folder_path = os.path.join(os.path.dirname(__file__), "data")

        # 根据模式选择对应的任务数据文件
        if self.mode == "eval":
            # 评估模式：使用评估任务数据文件
            task_items_path = os.path.join(folder_path, "all_pass_tasks_eval_148.json")
        elif self.mode == "train":
            # 训练模式：使用训练任务数据文件
            task_items_path = os.path.join(folder_path, "task_v2_gpt5_2550_w_checklist.json")
        else:
            # 如果 mode 不是 "eval" 或 "train"，抛出异常
            raise ValueError("mode must be eval or train")

        # 打开文件并读取 JSON 数据
        with open(task_items_path, encoding="utf-8") as f:
            task_items = json.load(f)

        # 打印加载信息
        print(f"Load {len(task_items)} tasks from {task_items_path}!")
        
        # 返回任务数据列表
        return task_items

    # ==============================
    # 状态重置
    # ==============================

    def reset_attributes(self):
        """
        重置类属性（日志和环境状态）
        
        功能说明：
        - 重置所有与轨迹记录相关的属性
        - 重置所有与环境实例相关的属性
        - 重置所有与任务场景相关的属性
        
        使用场景：
        - 在 reset() 开始时调用
        """
        # ========== 日志相关属性 ==========
        # 当前步骤计数器（从 0 开始）
        self.current_step = 0
        # 轨迹列表（记录所有步骤的状态、动作、观察等）
        self.trajectory = []

        # ========== 环境相关属性 ==========
        # 当前环境项（环境元数据字典）
        self.env_item = None
        # 环境类对象（动态创建的环境类）
        self.env_class = None
        # 环境实例对象（实际执行操作的环境实例）
        self.env_instance = None
        # 系统提示词（环境介绍和规则）
        self.system_prompt = None

        # ========== 场景相关属性 ==========
        # 初始配置（用于初始化环境实例）
        self.init_config = None
        # 初始状态（环境重置时的状态快照）
        self.init_state = None
        # 预测最终状态（任务完成时的状态快照，SFT 环境不使用）
        self.pred_final_state = None
        # 当前任务项（任务定义字典）
        self.task_item = None


    def reset(self, seed=None, task_index=None):
        """
        重置环境并返回初始观察、工具信息和任务信息
        
        功能说明：
        - 重置所有环境状态
        - 选择任务（随机或指定索引）
        - 加载环境和实例
        - 构建环境介绍和工具列表
        - 获取初始观察
        
        参数：
            seed (int, optional): 随机种子
            task_index (int, optional): 任务索引
        
        返回：
            tuple: (init_observation, info)
        
        使用示例：
            observation, info = env.reset(task_index=0)
        """
        # 重置所有类属性（清空之前的状态）
        self.reset_attributes()

        # 如果提供了随机种子，设置随机种子
        if seed is not None:
            random.seed(seed)

        # ========== 选择任务 ==========
        # 如果没有指定任务索引，随机选择一个任务
        if task_index is None:
            task_index = random.randrange(0, len(self.task_items))

        # 从任务列表中获取选定的任务项
        self.task_item = self.task_items[task_index]
        
        # 从任务项中提取任务 ID
        self.task_id = self.task_item["task_id"]
        
        # 从任务项中提取环境 ID
        self.env_id = self.task_item["env_id"]
        
        # 从任务项中提取初始配置
        self.init_config = self.task_item["init_config"]

        # ========== 加载环境和实例 ==========
        # 根据环境 ID 加载环境定义，并创建环境实例
        self.load_env_and_instance(env_id=self.env_id, init_config=self.init_config)
        
        # ========== 构建系统提示词 ==========
        # 构建环境介绍（包含环境简介和规则）
        self.env_introduction = self.construct_env_introduction(env_item=self.env_item)
        
        # ========== 获取工具列表 ==========
        # 深拷贝工具列表，避免修改原始数据
        self.tools = deepcopy(self.env_item["tools"])
        
        # ========== 获取初始观察 ==========
        # 返回任务描述作为初始观察
        init_observation = self.get_initial_observation(task_item=self.task_item)

        # ========== 构建返回信息 ==========
        # 深拷贝信息字典
        info = deepcopy({
            "env_introduction": self.env_introduction,  # 环境介绍
            "tools": self.tools,                        # 工具列表
            "task": self.task_item                      # 任务定义
        })
        
        # 返回初始观察和信息
        return init_observation, info


    # ==============================
    # 加载环境并初始化实例
    # ==============================
    
    def load_env_and_instance(self, env_id: int, init_config: dict):
        """
        根据环境 ID 加载环境并初始化实例
        
        功能说明：
        - 从环境数据集中查找对应的环境定义
        - 动态创建环境类
        - 创建环境实例并应用初始配置
        - 保存初始状态快照
        
        参数：
            env_id (int): 环境 ID
            init_config (dict): 初始配置
        
        使用示例：
            self.load_env_and_instance(env_id=0, init_config={...})
        """
        # 从环境数据集中获取对应的环境定义
        self.env_item = self.env_items[env_id]
        
        # 从环境定义中提取环境类源代码字符串
        env_class_code = self.env_item["env_class_code"]
        
        # 从任务项中提取环境类名称
        env_class_name = self.task_item["env_class_name"]
        
        # ========== 初始化环境类 ==========
        # 从源代码字符串动态创建环境类对象
        self.env_class = init_env_class(env_class_code, env_class_name)
        
        # ========== 创建环境实例 ==========
        # 使用环境类和初始配置创建环境实例
        self.env_instance = init_env_instance(self.env_class, init_config)
        
        # ========== 保存初始状态 ==========
        # 获取环境实例的初始状态快照
        self.init_state = get_state_info(self.env_instance)
        
        # ========== 初始轨迹记录 ==========
        # 将初始状态记录到轨迹中（作为第 0 步）
        self.trajectory.append({
            "step": 0,                                    # 步骤编号
            "state_snapshot": deepcopy(self.init_state)   # 状态快照
        })

    # ==============================
    # 环境交互步骤
    # ==============================

    def step(self, action: str | dict):
        """
        执行一步环境交互，返回观察、奖励、终止标志、截断标志和信息
        
        功能说明：
        - 解析 action（字符串或字典）
        - 验证 action 有效性
        - 检查是否为终止 action
        - 执行 action（工具调用）
        - 检查是否为终止 observation
        - 记录轨迹（不计算奖励，固定返回 0.0）
        
        参数：
            action (str | dict): Agent 的动作
        
        返回：
            tuple: (observation, reward, terminated, truncated, info)
                - reward 始终为 0.0（SFT 环境不计算奖励）
        
        使用示例：
            observation, reward, terminated, truncated, info = env.step(action)
        """
        # 深拷贝 action，避免修改原始数据
        raw_response = deepcopy(action)
        
        # ========== 初始化返回值 ==========
        # observation: 观察结果（初始为 None）
        # reward: 奖励值（初始为 0.0，SFT 环境始终为 0.0）
        # terminated: 是否终止（初始为 False）
        # truncated: 是否截断（初始为 False）
        # info: 额外信息（初始包含原始 action）
        observation, reward, terminated, truncated, info = None, 0.0, False, False, {"action": raw_response}
        

        # ========== 解析响应为 action 字典 ==========
        # 如果 action 是字符串（Prompt 模式），需要先解析为结构化响应
        if isinstance(raw_response, str):
            # 调用解析函数
            parse_success, struct_response = self._parse_response(text_response=raw_response)
            
            # 如果解析失败，返回错误观察并终止
            if not parse_success:
                observation = {"type": "user", "content": "Error: Failed to parse response to struct response"}
                # 记录步骤
                self._record_step(action, observation, terminated, reward)
                # 返回错误结果
                return observation, reward, terminated, truncated, info
        else:
            # 如果 action 已经是字典（FC 模式），直接使用
            struct_response = raw_response
        
        # ========== 解析结构化响应为 action ==========
        # 将结构化响应解析为统一的 action 格式
        parse_success, action = self._parse_action(struct_response)
        
        # 如果解析失败，返回错误观察并终止
        if not parse_success:
            observation = {"type": "user", "content": "Error: Failed to parse response to action"}
            # 记录步骤
            self._record_step(action, observation, terminated, reward)
            # 返回错误结果
            return observation, reward, terminated, truncated, info
    
        # 更新 info 字典，添加解析后的 action
        info.update({"action": action})
        
        # ========== 检查 action 有效性 ==========
        # 验证 action 名称和参数是否有效
        if not self.check_vaild_action(action=action):
            # 如果无效，返回错误观察并终止
            observation = {"type": "user", "content": "Error: Invalid action"}
            # 记录步骤
            self._record_step(action, observation, terminated, reward)
            # 返回错误结果
            return observation, reward, terminated, truncated, info

        # ========== 检查是否为终止 action ==========
        # 判断 action 是否为终止信号（"Task Completed" 或 "Task Failed"）
        if self.is_action_terminated(action):
            # 如果终止，设置观察和终止标志
            observation = {"type": "user", "content": "Task finished"}
            terminated = True
            
            # 保存最终状态快照（虽然不用于计算奖励，但可以用于分析）
            self.pred_final_state = get_state_info(self.env_instance)
            
            # SFT 环境不计算奖励，固定返回 0.0
            reward = self.calculate_reward()
            
            # 记录步骤
            self._record_step(action, observation, terminated, reward)
            
            # 如果是对话环境，保存用户消息历史（非对话环境没有 user_agent）
            if hasattr(self, "user_agent"):
                user_messages = self.user_agent.get_messages()
                info.update({"user_messages": user_messages})
            
            # 返回终止结果
            return observation, reward, terminated, truncated, info

        # ========== 执行 action ==========
        try:
            # 根据 action 类型执行不同的操作
            if action["name"] == "chat_with_user":
                # 非对话环境不应该有 chat_with_user（除非是终止信号）
                # 这里理论上不会执行到（因为已经在 is_action_terminated 中处理）
                observation = {"type": "user", "content": self.user_agent.user_step(agent_response=action['arguments']['content'])}
            else:
                # 如果是工具调用 action，调用环境实例的对应方法
                observation = {
                    "type": "tool", 
                    "content": f"{getattr(self.env_instance, action['name'])(**action['arguments'])}"
                }
            
            # ========== 检查是否为终止 observation ==========
            # 非对话环境不需要 observation 终止
            if self.is_observation_terminated(action, observation):
                terminated = True
                # 保存最终状态快照
                self.pred_final_state = get_state_info(self.env_instance)
                # SFT 环境不计算奖励，但这里调用了 calculate_reward（有参数）
                # 注意：这里有个 bug，应该调用无参的 calculate_reward()
                # 但为了保持代码一致性，这里保留原样
                reward = self.calculate_reward(self.checklist_with_func, self.init_state, self.pred_final_state)

            # ========== 记录步骤并返回 ==========
            # 记录当前步骤到轨迹中
            self._record_step(action, observation, terminated, reward)
            
            # 如果终止或截断，保存用户消息历史（如果是对话环境）
            if terminated or truncated:
                if hasattr(self, "user_agent"):
                    user_messages = self.user_agent.get_messages()
                    info.update({"user_messages": user_messages})
            
            # 返回结果
            return observation, reward, terminated, truncated, info

        except Exception:
            # ========== 捕获执行异常 ==========
            # 如果执行过程中出现异常，捕获并终止
            error_log = traceback.format_exc()
            
            # 设置错误观察
            observation = {"type": "user", "content": "Error: <Exception>\n" + error_log}
            
            # 设置终止标志
            terminated = True
            
            # 记录步骤（包含错误信息）
            self._record_step(action, observation, terminated, reward)
            
            # 返回错误结果
            return observation, reward, terminated, truncated, info

    # ==============================
    # 工具方法
    # ==============================
    
    def _record_step(self, action, observation, terminated, reward):
        """
        记录当前步骤的轨迹
        
        功能说明：
        - 获取上一步的状态快照
        - 获取当前状态快照
        - 计算状态差异
        - 将步骤信息添加到轨迹中
        
        参数：
            action (dict): 当前步骤的 action
            observation (dict): 当前步骤的 observation
            terminated (bool): 是否终止
            reward (float): 当前步骤的奖励（SFT 环境始终为 0.0）
        
        使用示例：
            self._record_step(action, observation, terminated, reward)
        """
        # 获取上一步的状态快照
        last_state = self.trajectory[-1]["state_snapshot"]
        
        # 获取当前环境实例的状态快照
        current_state = get_state_info(self.env_instance)
        
        # 计算状态差异
        state_diff = get_state_diff(last_state, current_state)
        
        # 将步骤信息添加到轨迹中
        self.trajectory.append({
            "step": self.current_step,           # 步骤编号
            "action": action,                     # 动作
            "observation": observation,           # 观察
            "terminated": terminated,             # 是否终止
            "reward": reward,                     # 奖励（SFT 环境始终为 0.0）
            "state_snapshot": current_state,      # 状态快照
            "state_diff": state_diff              # 状态差异
        })
    

    def _parse_response(self, text_response: str):
        """
        解析 LLM 输出（字符串格式）为结构化响应格式
        
        功能说明：
        - 从文本响应中提取思考内容、工具调用和文本内容
        
        参数：
            text_response (str): LLM 的文本响应
        
        返回：
            tuple: (parse_success, struct_response)
        
        使用示例：
            success, struct = self._parse_response("<tool_call>...</tool_call>")
        """
        # 调用解析工具函数
        parse_success, struct_response = parse_response(text_response)
        
        # 返回解析结果
        return parse_success, struct_response

    def _parse_action(self, struct_response: dict):
        """
        将结构化响应解析为 action
        
        功能说明：
        - 从结构化响应中提取 action
        
        参数：
            struct_response (dict): 结构化响应
        
        返回：
            tuple: (parse_success, action)
        
        使用示例：
            success, action = self._parse_action(struct_response)
        """
        # 调用解析工具函数
        parse_success, action = parse_action(struct_response)
        
        # 返回解析结果
        return parse_success, action
    
    
    def check_vaild_action(self, action: dict):
        """
        检查 action 有效性
        
        功能说明：
        - 检查 action 名称是否有效
        - 检查 action 参数是否有效
        
        参数：
            action (dict): action 字典
        
        返回：
            bool: action 是否有效
        
        使用示例：
            is_valid = self.check_vaild_action(action)
        """
        # 获取 action 名称
        method_name = action.get("name")
        
        # 检查 action 名称是否有效
        # 必须是环境实例的方法，或者是 "chat_with_user"
        if not (hasattr(self.env_instance, method_name) or method_name == "chat_with_user"):
            return False
        
        # 获取 action 参数
        params = action.get("arguments", {})
        
        # 检查参数是否有效
        # TODO: 检查 action 是否是目标工具之一（可以添加更严格的验证）
        # 1. 参数必须是字典类型
        # 2. 如果是 chat_with_user，必须包含 "content" 键
        if not isinstance(params, dict) or (method_name == "chat_with_user" and "content" not in params):
            return False
        
        # 如果所有检查都通过，返回 True
        return True
        

    def calculate_reward(self) -> float:
        """
        计算奖励（SFT 环境固定返回 0.0）
        
        功能说明：
        - SFT 数据不需要奖励，仅基于规则的过滤
        - 为了成本优化，不执行 check_func
        - 固定返回 0.0
        
        返回：
            float: 始终返回 0.0
        
        使用示例：
            reward = self.calculate_reward()
            # 始终返回 0.0
        """
        # SFT 环境不计算奖励，固定返回 0.0
        return 0.0

    def construct_env_introduction(self, env_item: dict):
        """
        构建环境介绍
        
        功能说明：
        - 从环境项中提取环境简介和规则
        - 格式化为 Markdown 格式的环境介绍
        
        参数：
            env_item (dict): 环境项字典
        
        返回：
            str: 环境介绍文本（Markdown 格式）
        
        使用示例：
            intro = self.construct_env_introduction(env_item)
        """
        # 获取环境简介
        env_brief_intro = env_item["environment_introduction"]
        
        # 构建环境规则字符串
        env_rule_str = ""
        # 遍历所有规则，格式化为列表项
        for rule in env_item.get("constraints_rules", []):
            env_rule_str += "- " + rule + "\n"
        
        # 格式化为 Markdown 格式的环境介绍
        env_introduction = f"# Environment Information\n\n## Brief Introduction:  \n{env_brief_intro}\n\n## Environment Rules / Constraints:  \n{env_rule_str}"
        
        # 返回环境介绍
        return env_introduction


    # ==============================
    # 终止和观察方法
    # ==============================

    def get_initial_observation(self, task_item: dict):
        """
        获取初始观察（返回任务描述）
        
        功能说明：
        - 非对话模式下，初始观察就是任务描述文本
        
        参数：
            task_item (dict): 任务项字典
        
        返回：
            str: 任务描述文本
        
        使用示例：
            observation = self.get_initial_observation(task_item)
        """
        # 返回任务描述文本作为初始观察
        return f"{task_item['task']}"

    def is_action_terminated(self, action: dict):
        """
        判断 action 是否为终止信号
        
        功能说明：
        - 非对话模式下，Agent 通过输出 "Task Completed" 或 "Task Failed" 来终止任务
        - 如果 action 是 "chat_with_user"，则认为是终止信号
        - 检查 action 内容中是否包含 "Task Completed" 或 "Task Failed"
        
        参数：
            action (dict): action 字典
        
        返回：
            bool: 是否终止
                - True: action 是 "chat_with_user"（表示任务完成或失败）
                - False: action 是工具调用（继续执行）
        
        使用示例：
            is_terminated = self.is_action_terminated(action)
        """
        # 如果 action 名称是 "chat_with_user"，表示 Agent 想要终止任务
        if action["name"] == "chat_with_user":
            # 检查 action 内容中是否包含 "Task Completed" 或 "Task Failed"
            # 如果都不包含，打印警告（但不影响终止判断）
            if 'Task Completed' not in action["arguments"]['content'] and 'Task Failed' not in action["arguments"]['content']:
                print('warning: Task Completed and Task Failed are all not in action["arguments"]["content"]')
            # 返回 True，表示任务终止（无论成功还是失败）
            return True
        
        # 如果 action 不是 "chat_with_user"（是工具调用），返回 False，继续执行
        return False

    def is_observation_terminated(self, action: dict, observation: str):
        """
        判断 observation 是否为终止信号
        
        功能说明：
        - 非对话模式下，不需要通过 observation 来终止
        
        参数：
            action (dict): action 字典（未使用）
            observation (str | dict): observation（未使用）
        
        返回：
            bool: 始终返回 False
        
        使用示例：
            is_terminated = self.is_observation_terminated(action, observation)
        """
        # 非对话模式不需要 observation 终止
        # 终止完全由 Agent 的 action 控制
        return False

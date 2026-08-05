# EnvScaler Environment Module

本模块实现了 EnvScaler 合成环境的完整实现，支持 RL（强化学习）和 SFT（监督微调）两种训练模式，以及对话和非对话两种交互模式。

## 📁 目录结构
 
```
envscaler_env/
├── __init__.py                    # 模块导出
├── base_env.py                    # 基础环境类（RL 环境基类）
├── rl_non_conv_env.py            # 非对话 RL 环境
├── rl_conv_env.py                # 对话 RL 环境
├── sft_non_conv_env_wo_reward_w_task_judge.py  # 非对话 SFT 环境
├── sft_conv_env_wo_reward.py     # 对话 SFT 环境
├── utils/                         # 工具函数
│   ├── env_util.py               # 环境初始化和状态管理工具
│   ├── parse_util.py             # 响应解析工具
│   ├── user_agent.py             # 用户代理实现
│   └── user_llm_inference.py     # 用户代理 LLM 推理接口
└── data/                          # 数据文件
    ├── 191_env_metadata.json     # 环境元数据
    ├── rl_scenario_metadata.json  # RL 任务场景数据
    └── sft_scenario_metadata.json # SFT 任务场景数据
```

## 🏗️ 架构设计

### 类继承关系

```
EnvScalerBaseEnv (抽象基类)
├── EnvScalerNonConvRLEnv (非对话 RL)
└── EnvScalerConvRLEnv (对话 RL)

独立实现（不继承基类）：
├── EnvScalerNonConvSFTEnv (非对话 SFT)
└── EnvScalerConvSFTEnv (对话 SFT)
```

### 环境类型对比

| 环境类型 | 继承关系 | 奖励计算 | 用户代理 | 终止方式 |
|---------|---------|---------|---------|---------|
| **NonConv RL** | 继承 `EnvScalerBaseEnv` | ✅ 基于 `check_func` | ❌ 无 | Agent 输出 "Task Completed" |
| **Conv RL** | 继承 `EnvScalerBaseEnv` | ✅ 基于 `check_func` | ✅ 有 | 用户输出 "###STOP###" |
| **NonConv SFT** | 独立实现 | ❌ 固定返回 0.0 | ❌ 无 | Agent 输出 "Task Completed" 或 "Task Failed" |
| **Conv SFT** | 独立实现 | ❌ 固定返回 0.0 | ✅ 有 | 用户输出 "###STOP###" |

## 📄 文件详解

### 1. `__init__.py` - 模块导出

**作用**：定义模块的公共接口，导出四个主要环境类。

**导出内容**：
- `EnvScalerConvRLEnv` - 对话 RL 环境
- `EnvScalerNonConvRLEnv` - 非对话 RL 环境
- `EnvScalerConvSFTEnv` - 对话 SFT 环境
- `EnvScalerNonConvSFTEnv` - 非对话 SFT 环境

**使用示例**：
```python
from envscaler_env import EnvScalerNonConvRLEnv
env = EnvScalerNonConvRLEnv(mode="train", ...)
```

---

### 2. `base_env.py` - 基础环境类（RL 环境基类）

**作用**：实现 RL 环境的通用功能，包括数据加载、环境初始化、动作执行、轨迹记录和奖励计算。

#### 核心功能

##### 2.1 数据加载 (`__init__`, `load_env_items`, `load_task_items`)

```python
def __init__(self, mode, env_items_path=None, task_items_path=None):
    # 加载任务数据集和环境数据集
    # - task_items: 任务列表（包含 task_id, env_id, task, init_config, checklist_with_func）
    # - env_items: 环境定义列表（包含 env_id, environment_introduction, tools, env_class_code）
```

**数据来源**：
- `task_items_path`: 任务数据文件路径（如 `env_scenario.json`）
- `env_items_path`: 环境元数据文件路径（如 `filtered_env_metadata.json`）

##### 2.2 环境重置 (`reset`)

```python
def reset(self, seed=None, task_index=None):
    # 1. 重置所有属性
    # 2. 选择任务（随机或指定索引）
    # 3. 加载环境和实例（动态执行 env_class_code）
    # 4. 构建环境介绍
    # 5. 获取工具列表
    # 6. 获取初始观察
    # 返回: (init_observation, info)
```

**关键步骤**：
- `load_env_and_instance()`: 从 `env_class_code` 动态创建环境类和实例
- `get_initial_observation()`: 由子类实现，返回初始观察

##### 2.3 环境交互 (`step`)

```python
def step(self, action: str | dict):
    # 1. 解析 action（字符串 → struct_response → action）
    # 2. 验证 action 有效性
    # 3. 检查是否为终止 action
    # 4. 执行 action（工具调用或 chat_with_user）
    # 5. 检查是否为终止 observation
    # 6. 记录轨迹
    # 返回: (observation, reward, terminated, truncated, info)
```

**执行流程**：
1. **解析阶段**：
   - 字符串 → `parse_response()` → `struct_response`
   - `struct_response` → `parse_action()` → `action`
2. **验证阶段**：检查 action 名称和参数
3. **执行阶段**：
   - `chat_with_user`: 调用 `user_agent.user_step()`
   - 其他工具: 调用 `env_instance.method_name(**arguments)`
4. **终止检查**：
   - `is_action_terminated()`: 检查 action 是否为终止信号
   - `is_observation_terminated()`: 检查 observation 是否为终止信号
5. **奖励计算**：任务完成时调用 `calculate_reward()`

##### 2.4 奖励计算 (`calculate_reward`)

```python
def calculate_reward(self, checklist_with_func, init_state, pred_final_state):
    # 1. 遍历所有 check_func
    # 2. 动态执行每个 check_func（传入 init_state 和 final_state）
    # 3. 计算平均结果（0.0-1.0）
    # 返回: float (平均奖励分数)
```

**奖励逻辑**：
- 每个 `check_func` 返回 `True` 或 `False`
- 最终奖励 = 所有 `check_func` 结果的平均值
- 范围：0.0（全部失败）到 1.0（全部成功）

##### 2.5 抽象方法（由子类实现）

```python
def get_initial_observation(self, task_item: dict):
    """返回初始观察（任务描述或用户初始对话）"""
    raise NotImplementedError

def is_action_terminated(self, action: dict):
    """检查 action 是否为终止信号"""
    raise NotImplementedError

def is_observation_terminated(self, action: dict, observation: str):
    """检查 observation 是否为终止信号"""
    raise NotImplementedError
```

#### 调用关系

```
base_env.py
├── utils/env_util.py
│   ├── init_env_class()      # 动态创建环境类
│   ├── init_env_instance()   # 创建环境实例
│   ├── get_state_info()       # 获取环境状态
│   ├── get_state_diff()       # 计算状态差异
│   └── run_check_function()   # 执行检查函数
└── utils/parse_util.py
    ├── parse_response()       # 解析文本响应
    └── parse_action()         # 解析为 action
```

---

### 3. `rl_non_conv_env.py` - 非对话 RL 环境

**作用**：实现非对话模式的 RL 环境，Agent 直接接收任务描述并执行。

#### 实现的方法

```python
def get_initial_observation(self, task_item: dict):
    """返回任务描述作为初始观察"""
    return f"{task_item['task']}"

def is_action_terminated(self, action: dict):
    """当 Agent 输出 'Task Completed' 时终止"""
    if action["name"] == "chat_with_user":
        if 'Task Completed' not in action["arguments"]['content']:
            print('warning: Task Completed not in action["arguments"]["content"]')
        return True
    return False

def is_observation_terminated(self, action: dict, observation: str):
    """非对话模式不需要 observation 终止"""
    return False
```

#### 特点

- ✅ 继承 `EnvScalerBaseEnv`，复用所有基础功能
- ✅ 支持奖励计算（基于 `check_func`）
- ❌ 无用户代理（`user_messages = null`）
- ✅ Agent 主动终止（输出 "Task Completed"）

#### 使用场景

- RL 训练数据收集
- 单轮多步任务执行
- 需要精确奖励信号的任务

---

### 4. `rl_conv_env.py` - 对话 RL 环境

**作用**：实现对话模式的 RL 环境，使用 UserAgent 模拟用户交互。

#### 实现的方法

```python
def __init__(self, mode, user_model, provider, ...):
    """初始化用户代理"""
    self.user_agent = UserAgent(...)
    super().__init__(mode=mode, ...)

def get_initial_observation(self, task_item: dict):
    """从用户代理获取初始回复"""
    return self.user_agent.get_init_reply(task=task_item['task'])

def is_action_terminated(self, action: dict):
    """对话模式不依赖 action 终止"""
    return False

def is_observation_terminated(self, action: dict, observation: str):
    """当用户输出 '###STOP###' 时终止"""
    return action.get("name") == "chat_with_user" and "###STOP###" in str(observation)
```

#### 特点

- ✅ 继承 `EnvScalerBaseEnv`
- ✅ 支持奖励计算（基于 `check_func`）
- ✅ 有用户代理（`user_messages` 包含完整对话历史）
- ✅ 用户主动终止（输出 "###STOP###"）

#### 使用场景

- 对话式 RL 训练
- 多轮交互任务
- 需要用户反馈的任务

---

### 5. `sft_non_conv_env_wo_reward_w_task_judge.py` - 非对话 SFT 环境

**作用**：实现非对话模式的 SFT 环境，不计算奖励，仅用于轨迹收集。

#### 关键差异（相比 RL 环境）

1. **不继承基类**：独立实现所有功能
2. **无奖励计算**：
   ```python
   def calculate_reward(self) -> float:
       """SFT 数据不需要奖励，仅基于规则的过滤"""
       return 0.0
   ```
3. **支持任务失败判断**：
   ```python
   def is_action_terminated(self, action: dict):
       """支持 'Task Completed' 和 'Task Failed'"""
       if action["name"] == "chat_with_user":
           if 'Task Completed' not in action["arguments"]['content'] and \
              'Task Failed' not in action["arguments"]['content']:
               print('warning: Task Completed and Task Failed are all not in action["arguments"]["content"]')
           return True
   ```

#### 特点

- ❌ 不继承基类（独立实现）
- ❌ 无奖励计算（固定返回 0.0）
- ❌ 无用户代理
- ✅ 支持任务失败标记（"Task Failed"）

#### 使用场景

- SFT 训练数据收集
- 成本优化（不需要执行 `check_func`）
- 基于 LLM 自判断的任务过滤

---

### 6. `sft_conv_env_wo_reward.py` - 对话 SFT 环境

**作用**：实现对话模式的 SFT 环境，使用 UserAgent 但不计算奖励。

#### 关键差异

1. **不继承基类**：独立实现
2. **有用户代理**：初始化 `UserAgent`
3. **无奖励计算**：固定返回 0.0
4. **用户终止**：通过 "###STOP###" 终止

#### 特点

- ❌ 不继承基类
- ❌ 无奖励计算
- ✅ 有用户代理
- ✅ 用户主动终止

#### 使用场景

- 对话式 SFT 训练数据收集
- 多轮交互任务（不需要奖励）

---

### 7. `utils/env_util.py` - 环境工具函数

**作用**：提供环境初始化、状态管理和检查函数执行的核心工具。

#### 核心函数

##### 7.1 `init_env_class(env_class_code, env_class_name)`

```python
def init_env_class(env_class_code: str, env_class_name: str):
    """从源代码字符串动态创建环境类"""
    # 1. 创建动态模块
    # 2. 执行 env_class_code（包含类定义）
    # 3. 提取指定名称的类
    # 返回: 环境类对象
```

**使用场景**：从 `env_items` 中的 `env_class_code` 动态创建环境类。

##### 7.2 `init_env_instance(env_class, init_config)`

```python
def init_env_instance(env_class, init_config=None):
    """创建环境实例并应用初始配置"""
    # 1. 尝试用 init_config 构造实例
    # 2. 如果失败，尝试无参构造
    # 3. 通过 setattr 设置初始属性
    # 返回: 环境实例对象
```

**使用场景**：创建环境实例并设置初始状态（如初始数据、配置等）。

##### 7.3 `get_state_info(env_instance)`

```python
def get_state_info(env_instance):
    """获取环境实例的状态字典"""
    # 提取所有非内置属性
    # 返回: 状态字典（深拷贝）
```

**使用场景**：记录环境状态快照，用于轨迹记录和状态比较。

##### 7.4 `get_state_diff(old_state, new_state)`

```python
def get_state_diff(old_state: dict, new_state: dict, ignore_keys: list = []):
    """计算两个状态字典的差异"""
    # 递归比较字典
    # 返回: 差异字典（包含 added, removed, changed）
```

**使用场景**：记录每一步的状态变化，用于轨迹分析。

##### 7.5 `run_check_function(func_code, init_state, final_state)`

```python
def run_check_function(func_code: str, init_state: dict, final_state: dict):
    """动态执行检查函数"""
    # 1. 创建安全全局环境（包含 initial_state）
    # 2. 执行 func_code（定义 check_func）
    # 3. 调用 check_func(final_state)
    # 返回: (success, result, error)
```

**使用场景**：执行任务完成检查，计算奖励。

#### 调用关系

```
env_util.py
└── 被 base_env.py 调用
    ├── init_env_class()        # reset() → load_env_and_instance()
    ├── init_env_instance()     # reset() → load_env_and_instance()
    ├── get_state_info()        # reset(), step() → 记录状态快照
    ├── get_state_diff()        # step() → _record_step()
    └── run_check_function()    # step() → calculate_reward()
```

---

### 8. `utils/parse_util.py` - 解析工具函数

**作用**：解析 LLM 响应，提取思考内容、工具调用和文本内容。

#### 核心函数

##### 8.1 `parse_response(text)`

```python
def parse_response(text):
    """
    从模型原始输出文本中解析 (reasoning_content, tool_calls, content)
    
    支持的格式：
    - <think>...</think> 或 @think@...@/think@
    - <tool_call>{...}</tool_call>
    - 纯文本内容
    
    返回: (parse_success, result_dict)
    result_dict = {
        "reasoning_content": str | None,
        "tool_calls": [{"function": {...}}] | None,
        "content": str | None
    }
    """
```

**解析逻辑**：
1. 提取思考内容（`<think>...</think>` 标签）
2. 提取工具调用（`<tool_call>...</tool_call>` 标签）
3. 提取文本内容（标签之间的内容）
4. 解析工具调用的 JSON

##### 8.2 `parse_action(struct_response)`

```python
def parse_action(struct_response: dict):
    """
    将 struct_response 解析为 action
    
    处理四种情况：
    1. 只有工具调用 → 返回工具调用 action
    2. 只有文本内容 → 返回 chat_with_user action
    3. 两者都有 → 忽略文本，返回工具调用 action
    4. 两者都无 → 返回错误
    
    返回: (parse_success, action_dict)
    action_dict = {
        "name": "function_name" | "chat_with_user",
        "arguments": {...}
    }
    """
```

#### 调用关系

```
parse_util.py
└── 被 base_env.py 和 SFT 环境调用
    ├── parse_response()    # step() → _parse_response()
    └── parse_action()      # step() → _parse_action()
```

---

### 9. `utils/user_agent.py` - 用户代理实现

**作用**：模拟真实用户与 Action Agent 交互，用于对话环境。

#### 核心类：`UserAgent`

##### 9.1 初始化

```python
def __init__(self, system_prompt, model, provider, api_key=None, base_url=None):
    """初始化用户代理"""
    self.messages = None      # 消息历史
    self.conversations = None # 对话记录
    self.model = model        # LLM 模型
    self.provider = provider  # 模型提供商
```

##### 9.2 获取初始回复

```python
def get_init_reply(self, task):
    """基于任务获取用户的初始回复"""
    # 1. 初始化消息历史（包含任务目标）
    # 2. 调用 LLM 生成初始回复
    # 3. 解析回复（提取 # Reply: 部分）
    # 返回: 用户初始回复文本
```

##### 9.3 处理 Agent 响应

```python
def user_step(self, agent_response):
    """处理 Agent 的响应并返回用户回复"""
    # 1. 将 Agent 响应添加到消息历史
    # 2. 调用 LLM 生成用户回复
    # 3. 解析回复（提取 # Reply: 部分）
    # 返回: 用户回复文本
```

##### 9.4 消息格式

用户代理的消息历史格式：
```python
[
    {"role": "system", "content": "You are a real human user..."},
    {"role": "user", "content": "[Agent] Hi! How can I help you today?"},
    {"role": "assistant", "content": "# Thought:\n...\n# Reply:\n..."},
    {"role": "user", "content": "[Agent] <Agent 的响应>"},
    {"role": "assistant", "content": "# Thought:\n...\n# Reply:\n..."},
    ...
]
```

**注意**：
- `role: "user"` 记录的是 Action Agent 的响应
- `role: "assistant"` 记录的是 User Agent 的响应（包含思考过程）

##### 9.5 响应解析

```python
def _parse_response(self, text: str):
    """解析包含 # Thought: 和 # Reply: 的响应"""
    # 1. 检查是否包含 "###STOP###"
    # 2. 提取 # Thought: 和 # Reply: 部分
    # 返回: (parse_success, reply_content)
```

#### 调用关系

```
user_agent.py
├── user_llm_inference.py
│   └── llm_inference()      # 调用 LLM 生成用户回复
└── 被对话环境调用
    ├── rl_conv_env.py        # 初始化 UserAgent
    └── sft_conv_env_wo_reward.py  # 初始化 UserAgent
```

---

### 10. `utils/user_llm_inference.py` - 用户代理 LLM 推理

**作用**：提供用户代理的 LLM 推理接口。

#### 核心函数

##### 10.1 `openai_llm_inference(...)`

```python
def openai_llm_inference(model, messages, temperature=0.7, ...):
    """调用 OpenAI API，带重试机制"""
    # 1. 创建 OpenAI 客户端
    # 2. 调用 API（最多重试 10 次）
    # 3. 返回响应内容
```

##### 10.2 `llm_inference(...)`

```python
def llm_inference(model, messages, provider, ...):
    """统一的 LLM 推理接口"""
    # 根据 provider 选择具体实现
    # 目前只支持 "openai"
```

#### 调用关系

```
user_llm_inference.py
└── 被 user_agent.py 调用
    └── UserAgent._infer() → llm_inference()
```

---

## 🔄 完整调用流程

### 非对话 RL 环境执行流程

```
1. 初始化
   EnvScalerNonConvRLEnv.__init__()
   └── EnvScalerBaseEnv.__init__()
       ├── load_task_items()          # 加载任务数据
       └── load_env_items()           # 加载环境数据

2. 重置环境
   env.reset(task_index=0)
   └── EnvScalerBaseEnv.reset()
       ├── load_env_and_instance()
       │   ├── init_env_class()       # utils/env_util.py
       │   └── init_env_instance()    # utils/env_util.py
       ├── get_initial_observation()  # 返回任务描述
       └── 返回 (observation, info)

3. 执行步骤（循环）
   env.step(action)
   └── EnvScalerBaseEnv.step()
       ├── _parse_response()         # utils/parse_util.py
       ├── _parse_action()            # utils/parse_util.py
       ├── check_vaild_action()
       ├── is_action_terminated()     # 检查 "Task Completed"
       ├── 执行工具调用
       │   └── getattr(env_instance, action['name'])(**arguments)
       ├── is_observation_terminated()
       ├── calculate_reward()         # 如果终止，计算奖励
       │   └── run_check_function()   # utils/env_util.py
       └── _record_step()
           ├── get_state_info()       # utils/env_util.py
           └── get_state_diff()       # utils/env_util.py
```

### 对话 RL 环境执行流程

```
1. 初始化
   EnvScalerConvRLEnv.__init__()
   ├── UserAgent.__init__()           # utils/user_agent.py
   └── EnvScalerBaseEnv.__init__()

2. 重置环境
   env.reset(task_index=0)
   └── EnvScalerBaseEnv.reset()
       ├── load_env_and_instance()
       └── get_initial_observation()
           └── user_agent.get_init_reply()
               └── llm_inference()    # utils/user_llm_inference.py

3. 执行步骤（循环）
   env.step(action)
   └── EnvScalerBaseEnv.step()
       ├── 解析 action
       ├── 如果是 chat_with_user
       │   └── user_agent.user_step()
       │       └── llm_inference()    # 生成用户回复
       ├── 如果是工具调用
       │   └── 执行工具
       ├── is_observation_terminated()  # 检查 "###STOP###"
       └── 如果终止
           └── user_agent.get_messages()  # 保存用户消息历史
```

## 📊 数据流

### 任务数据流

```
env_scenario.json (任务数据)
├── task_id: "0128-065713"
├── env_id: "env_1"
├── task: "任务描述文本"
├── init_config: {...}              # 环境初始配置
└── checklist_with_func: [           # 检查函数列表
    {
        "check_item": "检查项描述",
        "check_func": "def check_func(final_state): ..."
    }
]
```

### 环境数据流

```
filtered_env_metadata.json (环境元数据)
├── env_id: "env_1"
├── environment_introduction: "环境介绍"
├── constraints_rules: ["规则1", "规则2"]
├── tools: [...]                     # 工具函数定义列表
└── env_class_code: "class PharmacyEnv: ..."  # 环境类源代码
```

### 执行数据流

```
reset() → (observation, info)
├── observation: "任务描述" 或 "用户初始回复"
└── info: {
    "env_introduction": "...",
    "tools": [...],
    "task": {...}
}

step(action) → (observation, reward, terminated, truncated, info)
├── observation: {
    "type": "tool" | "user",
    "content": "工具执行结果" 或 "用户回复"
}
├── reward: 0.0 (中间步骤) 或 0.0-1.0 (终止时)
├── terminated: True/False
├── truncated: True/False
└── info: {
    "action": {...},
    "user_messages": [...] (仅对话环境)
}
```

## 🎯 使用示例

### 示例 1：非对话 RL 环境

```python
from envscaler_env import EnvScalerNonConvRLEnv

# 初始化环境
env = EnvScalerNonConvRLEnv(
    mode="train",
    env_items_path="../scen_generator/filtered_env_metadata.json",
    task_items_path="../scen_generator/final_result/env_scenario.json"
)

# 重置环境
observation, info = env.reset(task_index=0)
# observation = "任务描述文本"
# info = {"env_introduction": "...", "tools": [...], "task": {...}}

# 执行步骤
action = {
    "name": "get_prescription_by_id",
    "arguments": {"prescription_id": "RX2024-005"}
}
observation, reward, terminated, truncated, info = env.step(action)
# observation = {"type": "tool", "content": "{'success': True, 'data': {...}}"}
# reward = 0.0 (中间步骤)

# 任务完成
action = {
    "name": "chat_with_user",
    "arguments": {"content": "Task Completed"}
}
observation, reward, terminated, truncated, info = env.step(action)
# reward = 0.85 (基于 check_func 计算的平均分数)
# terminated = True
```

### 示例 2：对话 RL 环境

```python
from envscaler_env import EnvScalerConvRLEnv

# 初始化环境（需要用户模型）
env = EnvScalerConvRLEnv(
    mode="train",
    user_model="gpt-4.1",
    provider="openai",
    env_items_path="...",
    task_items_path="..."
)

# 重置环境
observation, info = env.reset(task_index=0)
# observation = "用户代理生成的初始回复"（基于任务描述）

# 执行步骤
action = {
    "name": "chat_with_user",
    "arguments": {"content": "我需要更新处方状态"}
}
observation, reward, terminated, truncated, info = env.step(action)
# observation = {"type": "user", "content": "用户代理的回复"}

# 用户终止
# 当用户代理输出 "###STOP###" 时，terminated = True
```

## 🔍 关键设计决策

### 1. 为什么 SFT 环境不继承基类？

**原因**：
- SFT 环境不需要奖励计算（固定返回 0.0）
- 代码更简洁，避免不必要的抽象层
- 可以独立优化 SFT 特定的功能

### 2. 为什么使用动态代码执行？

**原因**：
- 环境定义存储在 JSON 中（`env_class_code`）
- 支持灵活的环境定义，无需预编译
- 便于环境生成和扩展

### 3. 为什么对话环境需要 UserAgent？

**原因**：
- 模拟真实用户交互场景
- 支持多轮对话任务
- 提供更自然的训练数据

### 4. 奖励计算的时机

**时机**：
- 仅在任务终止时计算（`terminated = True`）
- 中间步骤奖励为 0.0
- 最终奖励 = `check_func` 结果的平均值

## 📝 注意事项

1. **环境实例状态**：每次 `reset()` 会创建新的环境实例，状态完全重置
2. **轨迹记录**：所有步骤的状态快照和差异都会记录在 `trajectory` 中
3. **错误处理**：工具调用异常会被捕获，返回错误观察并终止
4. **用户消息**：仅在对话环境中存在，非对话环境为 `null`
5. **终止条件**：非对话环境由 Agent 控制，对话环境由用户控制

## 🔗 相关模块

- `agent/task_solve_agent.py` - 使用本模块的 Agent 实现
- `agent/agent_llm_inference.py` - Agent 的 LLM 推理接口
- `scen_generator/` - 生成环境和任务数据


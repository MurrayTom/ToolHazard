# Agent 模块详解

## 📋 目录概述

`agent/` 模块实现了基于 LLM 的任务解决 Agent，负责与环境交互并完成任务。该模块包含三个核心文件，共同实现了完整的 Agent 决策和执行流程。

---

## 📁 文件结构

```
agent/
├── task_solve_agent.py      # Agent 核心逻辑（主入口）
├── agent_llm_inference.py   # LLM 推理接口
└── system_prompt_util.py    # 系统提示词工具
```

---

## 🔗 模块调用关系图

```
┌─────────────────────────────────────────┐
│      run_main.py / run_main_debug.py    │
│          (外部调用入口)                   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│      task_solve_agent.py                │
│      ┌──────────────────────────────┐   │
│      │  TaskSolveAgent              │   │
│      │  - reset()                   │   │
│      │  - step()                    │   │
│      │  - run()                     │   │
│      └──────────┬───────────────────┘   │
│                 │                        │
│      ┌──────────┴──────────┐            │
│      │                     │            │
│      ▼                     ▼            │
│  system_prompt_util.py  agent_llm_inference.py
│  - 获取系统提示词        - LLM API 调用
│  - 合并工具信息          - 处理响应
└─────────────────────────────────────────┘
```

---

## 📄 文件详解

### 1. `task_solve_agent.py` - Agent 核心逻辑

**功能**：实现任务解决 Agent 的完整逻辑，管理 Agent 与环境的交互循环。

#### 核心类：`TaskSolveAgent`

**职责**：
- 管理 Agent 的完整生命周期
- 处理环境交互循环
- 记录轨迹和执行状态
- 协调提示词和 LLM 调用

#### 关键方法详解

##### `__init__()` - 初始化

```python
def __init__(self, env_name, env, model, provider, temperature, 
             infer_mode, max_steps, enable_thinking, ...):
```
 
**功能**：
- 初始化 Agent 配置（模型、提供商、推理模式等）
- 初始化运行时状态（消息历史、轨迹、奖励等）

**参数说明**：
- `env_name`: 环境名称（用于选择系统提示词）
- `env`: 环境实例
- `model`: LLM 模型名称
- `provider`: 模型提供商（如 "openai"）
- `infer_mode`: 推理模式（"prompt" 或 "fc"）
  - `"prompt"`: 通过提示词进行工具调用（Agent 自己解析）
  - `"fc"`: 通过函数调用接口（LLM 原生支持）
- `max_steps`: 最大执行步数
- `enable_thinking`: 是否启用思考模式

**初始化状态**：
```python
self.messages = []           # 对话历史
self.trajectory = []         # 执行轨迹
self.total_reward = 0.0      # 累计奖励
self.terminated = False      # 是否正常终止
self.truncated = False       # 是否被截断
self.step_count = 0          # 当前步数
```

---

##### `reset()` - 重置环境

```python
def reset(self, task_index=None):
```

**功能**：
1. 重置环境并获取初始观察
2. 构建系统提示词（根据环境类型选择）
3. 合并工具信息到提示词（Prompt 模式）
4. 初始化消息历史和轨迹

**执行流程**：
```
1. env.reset(task_index) 
   ↓
2. 获取环境信息（tools, env_introduction）
   ↓
3. 选择系统提示词（对话/非对话）
   ├─ 对话环境 → conversational_system_prompt
   └─ 非对话环境 → non_conversational_system_prompt
   ↓
4. 添加环境介绍到提示词
   ↓
5. Prompt 模式：合并工具信息到提示词
   └─ merge_tools_into_system_prompt()
   ↓
6. 初始化消息历史
   ├─ system: 系统提示词
   └─ user: 初始观察（任务描述）
   ↓
7. 记录初始轨迹
```

**调用关系**：
- 调用 `system_prompt_util.merge_tools_into_system_prompt()` 合并工具信息
- 调用 `env.reset()` 重置环境

**关键代码**：
```python
# 选择系统提示词
if env_name in ["...", "envscaler_conversation_sft", ...]:
    system_prompt = conversational_system_prompt
else:
    system_prompt = non_conversational_system_prompt

# Prompt 模式：合并工具信息
if self.infer_mode == "prompt":
    system_prompt = merge_tools_into_system_prompt(
        system_prompt=system_prompt, 
        tools=info["tools"]
    )
```

---

##### `step()` - 执行一步

```python
def step(self):
    """
    执行一步环境交互：
    1. 调用 LLM 生成响应
    2. 执行动作
    3. 更新状态和轨迹
    """
```

**功能**：
1. 调用 LLM 生成响应（动作）
2. 将响应传递给环境执行
3. 更新消息历史、奖励、轨迹等

**执行流程**：
```
1. 检查环境是否已结束
   ↓
2. 调用 LLM 推理
   ├─ Prompt 模式 → llm_inference_prompt()
   └─ FC 模式 → llm_inference_fc()
   ↓
3. 解析 LLM 响应
   ├─ Prompt 模式：文本响应（需要环境解析）
   └─ FC 模式：结构化响应（tool_calls, content）
   ↓
4. 添加到消息历史
   ↓
5. 执行环境步骤
   └─ env.step(action=raw_response)
   ↓
6. 更新内部状态
   ├─ step_count += 1
   ├─ total_reward += reward
   ├─ terminated, truncated
   └─ current_observation, current_info
   ↓
7. 添加观察到消息历史
   ├─ tool 类型 → <tool_response>...</tool_response>
   └─ user 类型 → 直接添加
   ↓
8. 记录轨迹
```

**调用关系**：
- 调用 `agent_llm_inference.llm_inference_prompt()` 或 `llm_inference_fc()`
- 调用 `env.step()` 执行动作

**关键代码**：
```python
# Prompt 模式
if self.infer_mode == "prompt":
    raw_response = llm_inference_prompt(
        provider=self.provider,
        model=self.model,
        messages=self.messages,
        temperature=self.temperature,
        enable_thinking=self.enable_thinking,
        ...
    )
# FC 模式
else:
    raw_response = llm_inference_fc(
        provider=self.provider,
        model=self.model,
        messages=self.messages,
        tools=self.tools,  # FC 模式需要工具列表
        ...
    )

# 执行环境步骤
observation, reward, terminated, truncated, info = self.env.step(
    action=raw_response
)
```

**消息历史更新**：
```python
# Prompt 模式：工具响应格式
if observation["type"] == "tool":
    observation_content = f"<tool_response>\n{observation['content']}\n</tool_response>"
    self.messages.append({"role": "user", "content": observation_content})

# FC 模式：使用 tool 角色
else:
    self.messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_call_name,
        "content": observation['content']
    })
```

---

##### `run()` - 运行完整任务

```python
def run(self, task_index=None, max_steps=None):
    """
    运行完整任务：
    1. 重置环境
    2. 循环执行 step() 直到结束
    3. 返回完整结果
    """
```

**功能**：
- 执行完整的任务解决流程
- 从重置到结束的完整循环
- 返回包含所有信息的执行结果

**执行流程**：
```
1. reset(task_index)
   ↓
2. while not terminated and not truncated and step_count < max_steps:
   └─ step()
   ↓
3. 聚合结果
   ├─ task_info
   ├─ tools
   ├─ messages (完整对话历史)
   ├─ trajectory (执行轨迹)
   ├─ total_reward
   ├─ terminated, truncated
   └─ final_observation, final_info
   ↓
4. 返回结果字典
```

**返回结果结构**：
```python
{
    "task_info": {...},           # 任务信息
    "tools": [...],               # 可用工具列表
    "messages": [...],            # 完整对话历史
    "user_messages": [...],       # 用户消息（对话环境）
    "trajectory": [...],          # 详细执行轨迹
    "total_reward": 0.0,          # 累计奖励
    "terminated": True,           # 是否正常终止
    "truncated": False,           # 是否被截断
    "final_observation": {...},   # 最终观察
    "final_info": {...},          # 最终信息
    "steps": 10                   # 总步数
}
```

---

### 2. `agent_llm_inference.py` - LLM 推理接口

**功能**：封装 LLM API 调用，提供统一的推理接口，支持多种推理模式。

#### 核心函数

##### `llm_inference_prompt()` - Prompt 模式统一接口

```python
def llm_inference_prompt(provider, model, messages, temperature, 
                         enable_thinking, ...) -> str:
```

**功能**：
- Prompt 模式的统一接口
- 返回文本响应（Agent 需要自己解析工具调用）

**调用链**：
```
llm_inference_prompt()
  └─> openai_stream_inference_prompt()
```

**返回**：文本字符串，包含工具调用标签（如 `<tool_call>...</tool_call>`）

---

##### `llm_inference_fc()` - FC 模式统一接口

```python
def llm_inference_fc(provider, model, messages, temperature, tools,
                     enable_thinking, ...) -> Dict[str, Any]:
```

**功能**：
- FC（Function Calling）模式的统一接口
- 返回结构化响应（包含 tool_calls）

**调用链**：
```
llm_inference_fc()
  └─> openai_stream_inference_fc()
```

**返回**：
```python
{
    "reasoning_content": str,    # 思考内容（如果启用）
    "tool_calls": [              # 工具调用列表
        {
            "id": "...",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "{...}"
            }
        }
    ],
    "content": str               # 文本内容
}
```

---

##### `openai_stream_inference_prompt()` - Prompt 模式流式推理

```python
def openai_stream_inference_prompt(model, messages, temperature,
                                   enable_thinking, ...) -> str:
```

**功能**：
- 使用 OpenAI API 进行流式推理（Prompt 模式）
- 处理思考内容（reasoning content）
- 自动重试机制

**执行流程**：
```
1. 创建 OpenAI 客户端
   ↓
2. 构建请求参数
   ├─ 检查是否为标准 OpenAI API
   └─ 条件性添加 chat_template_kwargs
   ↓
3. 流式调用 API
   ↓
4. 累积响应内容
   ├─ reasoning_content（思考内容）
   └─ content（主要内容）
   ↓
5. 处理思考标签
   └─ 如果存在，格式化为 <think>...</think>
   ↓
6. 返回完整内容
```

**关键特性**：
- ✅ 流式处理（实时接收响应）
- ✅ 自动重试（最多 10 次）
- ✅ 错误处理
- ✅ 思考模式支持

**思考内容处理**：
```python
# 如果模型返回了思考内容，格式化为标签
if reasoning_content:
    content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
```

---

##### `openai_stream_inference_fc()` - FC 模式流式推理

```python
def openai_stream_inference_fc(model, messages, temperature, tools,
                                enable_thinking, ...) -> Dict[str, Any]:
```

**功能**：
- 使用 OpenAI API 进行流式推理（FC 模式）
- 处理工具调用和思考内容
- 支持多工具调用（但只保留第一个）

**执行流程**：
```
1. 创建 OpenAI 客户端
   ↓
2. 构建请求参数
   ├─ 添加 tools 参数（工具列表）
   ├─ tool_choice="auto"
   └─ 条件性添加 chat_template_kwargs
   ↓
3. 流式调用 API
   ↓
4. 累积响应内容
   ├─ reasoning_content（思考内容）
   ├─ content（文本内容）
   └─ tool_calls（工具调用，按 index 累积）
   ↓
5. 处理工具调用
   ├─ 按 index 分组累积
   └─ 如果多个，只保留第一个
   ↓
6. 返回结构化结果
```

**工具调用处理**：
```python
# 按 index 累积工具调用
for chunk in completion:
    if hasattr(delta, "tool_calls") and delta.tool_calls:
        for tool_call in delta.tool_calls:
            idx = tool_call.index
            # 累积到 tool_calls_accum[idx]
```

**返回结构**：
```python
{
    "reasoning_content": "...",      # 思考内容
    "tool_calls": [                  # 工具调用（最多1个）
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": '{"key": "value"}'
            }
        }
    ],
    "content": "..."                  # 文本内容
}
```

---

##### `openai_inference_prompt()` - Prompt 模式非流式推理

```python
def openai_inference_prompt(model, messages, temperature,
                            enable_thinking, ...) -> str:
```

**功能**：
- 非流式推理（一次性返回）
- 用于不需要流式处理的场景

**与流式版本的区别**：
- ❌ 不使用 `stream=True`
- ✅ 一次性获取完整响应
- ✅ 更简单的实现

---

#### 关键设计

##### 1. 思考模式支持

**条件检查**：
```python
# 检查是否为标准 OpenAI API
actual_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
is_standard_openai = actual_base_url.rstrip('/') in ["https://api.openai.com/v1", "https://api.openai.com"]

# 只有非标准 API 且启用思考模式时才添加参数
if enable_thinking and not is_standard_openai:
    params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
```

**原因**：
- 标准 OpenAI API 不支持 `chat_template_kwargs`
- 只有支持思考模式的 API（如 Qwen3）才需要此参数

##### 2. 自动重试机制

```python
retries = 0
max_retries = 10
while retries < max_retries:
    try:
        # API 调用
        ...
        return result
    except Exception as e:
        print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
        time.sleep(retries * 10)
        retries += 1
```

**特性**：
- ✅ 最多重试 10 次
- ✅ 指数退避（每次等待时间递增）
- ✅ 错误日志记录

##### 3. 流式处理

**优势**：
- ✅ 实时接收响应（更好的用户体验）
- ✅ 可以处理长响应
- ✅ 支持思考内容的流式接收

**实现**：
```python
for chunk in completion:
    # 累积每个 chunk 的内容
    if hasattr(delta, "content") and delta.content:
        content += delta.content
```

---

### 3. `system_prompt_util.py` - 系统提示词工具

**功能**：定义和管理 Agent 的系统提示词，提供工具信息合并功能。

#### 核心内容

##### `non_conversational_system_prompt` - 非对话环境提示词

```python
non_conversational_system_prompt = """
You are a helpful assistant. When given a specific task, your goal is to 
complete it in an interactive environment by making step-by-step use of 
available tools. 
- Before completing the task, at each step, select a tool from the tool 
  list and fill in all required parameters...
- When you believe the task has been completed, respond only with 
  'Task Completed' to end the trajectory...
- It is recommended to first call query tools to gather sufficient 
  information, then use modification tools to complete the task...
"""
```

**特点**：
- ✅ 直接执行任务，无需用户交互
- ✅ 完成后发送 "Task Completed"
- ✅ 强调工具使用的步骤性

**适用环境**：
- `envscaler_non_conversation_sft`
- `envscaler_non_conversation_rl`
- `bfcl`
- `acebench_multi_step`

---

##### `conversational_system_prompt` - 对话环境提示词

```python
conversational_system_prompt = """
You are a helpful assistant. Your goal is to fulfill the user's requests 
in an interactive environment by step-by-step use of available tools, while 
proactively communicating with the user when necessary, until the user ends 
the conversation.
- If you lack essential information..., actively ask the user...
- If you can proceed with the current information, select one tool...
- When you believe the task is completed, clearly inform the user...
"""
```

**特点**：
- ✅ 支持用户交互
- ✅ 可以主动询问用户
- ✅ 任务完成后询问是否有新任务

**适用环境**：
- `tau_bench_retail`
- `tau_bench_airline`
- `envscaler_conversation_rl`
- `envscaler_conversation_sft`
- `acebench_multi_turn`

---

##### `merge_tools_into_system_prompt()` - 工具信息合并

```python
def merge_tools_into_system_prompt(system_prompt, tools) -> str:
```

**功能**：
- 将工具列表合并到系统提示词中（Prompt 模式）
- 生成符合 Qwen3 格式的提示词

**执行流程**：
```
1. 检查工具列表是否为空
   ↓
2. 构建输出字符串
   ├─ 添加系统提示词
   ├─ 添加工具部分标题
   ├─ 添加工具列表（JSON 格式）
   └─ 添加工具调用说明
   ↓
3. 返回完整提示词
```

**输出格式**：
```
{系统提示词}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {...}}
{"type": "function", "function": {...}}
...
</tools>

For each function call, return a json object with function name and arguments 
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

**使用场景**：
- ✅ Prompt 模式：需要将工具信息告诉 Agent
- ❌ FC 模式：工具信息通过 API 参数传递，不需要合并

---

## 🔄 完整调用流程

### 任务执行完整流程

```
1. 外部调用
   run_main.py → TaskSolveAgent.run(task_index=0)
   ↓
2. 初始化
   TaskSolveAgent.__init__(...)
   ├─ 设置模型、提供商、推理模式等
   └─ 初始化状态变量
   ↓
3. 重置环境
   TaskSolveAgent.reset(task_index=0)
   ├─ env.reset(task_index)
   ├─ 选择系统提示词
   │   └─ system_prompt_util.non_conversational_system_prompt
   ├─ 合并工具信息（Prompt 模式）
   │   └─ system_prompt_util.merge_tools_into_system_prompt()
   └─ 初始化消息历史
   ↓
4. 执行循环
   while not terminated and step_count < max_steps:
       TaskSolveAgent.step()
       ├─ 调用 LLM 推理
       │   ├─ Prompt 模式
       │   │   └─ agent_llm_inference.llm_inference_prompt()
       │   │       └─ agent_llm_inference.openai_stream_inference_prompt()
       │   └─ FC 模式
       │       └─ agent_llm_inference.llm_inference_fc()
       │           └─ agent_llm_inference.openai_stream_inference_fc()
       ├─ 添加到消息历史
       ├─ 执行环境步骤
       │   └─ env.step(action=raw_response)
       ├─ 更新状态
       └─ 记录轨迹
   ↓
5. 返回结果
   return {
       "task_info": ...,
       "messages": ...,
       "trajectory": ...,
       ...
   }
```

---

## 📊 数据流图

### Prompt 模式数据流

```
TaskSolveAgent
  │
  ├─> reset()
  │   ├─> env.reset() → observation, info
  │   ├─> system_prompt_util.merge_tools_into_system_prompt()
  │   │   └─> 返回: 完整系统提示词（包含工具信息）
  │   └─> 初始化 messages = [system, user(observation)]
  │
  └─> step()
      ├─> agent_llm_inference.llm_inference_prompt()
      │   └─> openai_stream_inference_prompt()
      │       └─> 返回: 文本响应（包含 <tool_call> 标签）
      │
      ├─> messages.append(assistant: raw_response)
      │
      ├─> env.step(action=raw_response)
      │   └─> 返回: observation, reward, terminated, truncated, info
      │
      └─> messages.append(user: <tool_response>...</tool_response>)
```

### FC 模式数据流

```
TaskSolveAgent
  │
  ├─> reset()
  │   ├─> env.reset() → observation, info
  │   └─> 初始化 messages = [system, user(observation)]
  │       （不需要合并工具信息，工具通过 API 参数传递）
  │
  └─> step()
      ├─> agent_llm_inference.llm_inference_fc()
      │   └─> openai_stream_inference_fc()
      │       ├─> 传入: tools (工具列表)
      │       └─> 返回: {
      │           "tool_calls": [...],
      │           "content": "...",
      │           "reasoning_content": "..."
      │       }
      │
      ├─> messages.append(assistant: {content, tool_calls, ...})
      │
      ├─> env.step(action=raw_response)
      │   └─> 返回: observation, reward, terminated, truncated, info
      │
      └─> messages.append(tool: {tool_call_id, name, content})
```

---

## 🔑 关键概念

### 1. 推理模式（Infer Mode）

#### Prompt 模式

**特点**：
- Agent 通过文本提示词学习工具调用格式
- LLM 返回包含工具调用标签的文本
- 环境需要解析文本提取工具调用

**工具调用格式**：
```xml
<tool_call>
{"name": "adjust_medication_inventory", "arguments": {"medication_id": "MED002", "quantity_change": 5}}
</tool_call>
```

**优势**：
- ✅ 兼容性更好（所有 LLM 都支持）
- ✅ 可以自定义格式

**劣势**：
- ❌ 需要解析文本
- ❌ 可能解析失败

#### FC 模式

**特点**：
- LLM 原生支持函数调用接口
- 返回结构化的工具调用对象
- 不需要文本解析

**工具调用格式**：
```python
{
    "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
            "name": "adjust_medication_inventory",
            "arguments": '{"medication_id": "MED002", "quantity_change": 5}'
        }
    }]
}
```

**优势**：
- ✅ 结构化，无需解析
- ✅ 更可靠
- ✅ 支持多工具调用

**劣势**：
- ❌ 需要 LLM 支持函数调用接口
- ❌ 兼容性较差

---

### 2. 思考模式（Thinking Mode）

**功能**：
- 支持混合思考模型（如 Qwen3-8B）
- LLM 可以输出思考过程

**格式**：
```xml
<think>
这是思考过程...
</think>

这是实际响应...
```

**启用条件**：
- `enable_thinking = True`
- 使用支持思考模式的 API（非标准 OpenAI API）

---

### 3. 消息历史（Messages）

**结构**：
```python
messages = [
    {"role": "system", "content": "系统提示词"},
    {"role": "user", "content": "初始观察（任务描述）"},
    {"role": "assistant", "content": "Agent 响应"},
    {"role": "user", "content": "工具响应或用户消息"},
    ...
]
```

**角色类型**：
- `system`: 系统提示词（只在开始时添加一次）
- `user`: 用户消息或环境观察
- `assistant`: Agent 的响应
- `tool`: 工具执行结果（仅 FC 模式）

---

### 4. 轨迹（Trajectory）

**结构**：
```python
trajectory = [
    {
        "step": 0,
        "observation": "初始观察"
    },
    {
        "step": 1,
        "action": {"name": "...", "arguments": {...}},
        "observation": {...},
        "reward": 0.0,
        "terminated": False,
        "truncated": False
    },
    ...
]
```

**用途**：
- 记录完整的执行过程
- 用于训练数据收集
- 用于分析和调试

---

## 🎯 使用示例

### 基本使用

```python
from agent.task_solve_agent import TaskSolveAgent
from envscaler_env import EnvScalerNonConvSFTEnv

# 初始化环境
env = EnvScalerNonConvSFTEnv(
    mode="train",
    env_items_path="...",
    task_items_path="..."
)

# 初始化 Agent
agent = TaskSolveAgent(
    env_name="envscaler_non_conversation_sft",
    env=env,
    model="gpt-4.1-mini",
    provider="openai",
    temperature=0.7,
    infer_mode="prompt",
    max_steps=30,
    enable_thinking=False
)

# 执行任务
result = agent.run(task_index=0)

# 获取结果
print(f"任务完成: {result['terminated']}")
print(f"总步数: {result['steps']}")
print(f"总奖励: {result['total_reward']}")
```

---

## 🔧 配置说明

### Agent 配置参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `env_name` | str | 环境名称 | 必需 |
| `env` | object | 环境实例 | 必需 |
| `model` | str | LLM 模型名称 | 必需 |
| `provider` | str | 模型提供商 | 必需 |
| `temperature` | float | 温度参数 | 0.7 |
| `infer_mode` | str | 推理模式（"prompt"/"fc"） | 必需 |
| `max_steps` | int | 最大步数 | 30 |
| `enable_thinking` | bool | 是否启用思考模式 | False |
| `api_key` | str | API 密钥（可选） | None |
| `base_url` | str | API 基础 URL（可选） | None |

---

## 🐛 常见问题

### Q1: Prompt 模式和 FC 模式有什么区别？

**A**: 
- **Prompt 模式**：通过文本提示词告诉 Agent 如何调用工具，Agent 返回文本，需要解析
- **FC 模式**：使用 LLM 原生的函数调用接口，返回结构化对象，无需解析

### Q2: 什么时候使用思考模式？

**A**: 
- 使用支持思考模式的模型（如 Qwen3-8B）
- 需要查看 Agent 的思考过程
- 使用非标准 OpenAI API

### Q3: 消息历史是如何更新的？

**A**: 
- 每次 `step()` 都会添加两条消息：
  1. `assistant`: Agent 的响应
  2. `user`: 环境的观察结果

### Q4: 如何选择系统提示词？

**A**: 
- 根据 `env_name` 自动选择：
  - 对话环境 → `conversational_system_prompt`
  - 非对话环境 → `non_conversational_system_prompt`

---

## 📝 总结

Agent 模块实现了完整的任务解决 Agent，包含：

1. **TaskSolveAgent**：核心逻辑，管理交互循环
2. **agent_llm_inference**：LLM 调用接口，支持多种模式
3. **system_prompt_util**：提示词管理，工具信息合并

三个模块协同工作，实现了从任务接收到完成的完整流程。


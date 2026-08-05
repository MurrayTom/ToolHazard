# EnvScaler/attacker 讲解文档：整体流程、代码关系、数据流与数据结构

---

## 1) 端到端整体流程

把系统拆成 6 个阶段：

- **阶段 0（可离线,可以放到env那里）**：攻击点发现 `attack_point/`  
  输入：环境定义（`env_class_code`、`operation_list`、`state_space_definition`…）  
  输出：`env_item["attack_point"]`（可注入属性 + 读写操作集合）

- **阶段 1**：任务-轨迹对齐 `attackpoint_task_match/`  
  输入：`(env_id, task_id)` + 轨迹文件 `traj.json`  
  输出：`action_names`（该任务真实执行过的 operation 名）

- **阶段 2**：攻击点可达性过滤 `attackpoint_task_match/find_match.py`  
  输入：`action_names` + `attack_point`  
  输出：`filtered_attack_points`（只保留 write 操作确实出现过的点）

- **阶段 3**：注入位置选择 + 注入计划生成 `IPI/gen_IPI_plan.py`  
  输入：`filtered_attack_points` + `traj_entry` + `task_item/init_config` + 环境介绍/操作/约束  
  输出：`injection_plans = [{injection_location, plan_steps}, ...]`

- **阶段 4**：注入任务文本生成 + 策略包装 `IPI/gen_IPI_task.py` + `IPI/injection_strategy_utils/`  
  输入：原任务与环境信息 +（可选）`injection_location`  
  输出：`injected_task_raw`（LLM 生成的“劫持任务”）以及 `injected_task`（应用策略后的实际注入文本）

- **阶段 5**：攻击代理执行注入 `IPI/IPI.py` + `IPI/attack_agent/` + `envscaler_env/`  
  输入：`injection_plan` + `injected_task` + 原始 `task_item` + `env_item`  
  输出：`final_state`（注入后的环境状态快照）与执行轨迹

- **阶段 6**：验证与过滤 `verification/`  
  输入：`injected_task + injection_location + final_state`，以及原任务 `checklist_with_func`  
  输出：是否保留该样本（通过验证 + 不 overlap）；并生成 `injected_checklist_with_func`

把上述阶段串成一个时序（`run_main.py:process_tasks()` 的主干）：

```
读 env.json + tasks.json + traj.json
|
|-- 对每个 task_item：
|     traj_entry = find_traj_entry_for_task(env_id, task_id)
|     action_names = find_operations_for_task(traj_entry)
|     filtered_attack_points = filter_attack_points_by_actions(action_names, env_item.attack_point)
|     injection_plans = gen_IPI_plan(..., attack_points=filtered_attack_points, traj_entry=traj_entry)
|
|-- 对每个 injection_plan（单位置版本）：
|     injected_task_raw = gen_IPI_task(...)
|     injected_task = apply_injection_strategy(injected_task_raw, strategy)
|     injection_result = execute_injection(injection_plan, injected_task, ...)
|     final_state = injection_result.final_state
|     verified = verify_injection_with_llm(injected_task, injection_location, final_state)
|     check_result = gen_IPI_check_func(injected_task, final_state, ...)
|     overlap = state_overlap(...) or semantic_overlap(...)
|     if verified and not overlap: 组装 new_task 并写出
```

---

## 2) 代码依赖关系

以 `run_main.py` 为中心的调用依赖可以概括为：

```
run_main.py
├─ attackpoint_task_match/find_operations.py
│   └─ (traj.json 解析：trajectory/messages 两种格式)
├─ attackpoint_task_match/find_match.py
├─ IPI/gen_IPI_plan.py
│   ├─ summarize_traj_for_injection_locations() (轨迹摘要：只保留 attack_point.read 相关调用)
│   └─ IPI/utils/call_llm.py -> llm_inference()
├─ IPI/gen_IPI_task.py
│   └─ IPI/utils/call_llm.py -> llm_inference()
├─ IPI/injection_strategy_utils/__init__.py -> apply_injection_strategy()
│   ├─ hand_crafted.py
│   ├─ multi_turn.py
│   └─ important_variants.py
├─ IPI/IPI.py -> execute_injection()
│   ├─ envscaler_env/non_conv_env.py -> AttackNonConvEnv
│   ├─ IPI/attack_agent/task_solve_agent.py -> TaskSolveAgent
│   │   ├─ IPI/attack_agent/agent_llm_inference.py (OpenAI 推理封装)
│   │   └─ envscaler_env/utils/parse_util.py (parse_response/parse_action)
│   └─ envscaler_env/utils/env_util.py (get_state_info 等)
└─ verification/
    ├─ verify_injection.py -> llm_inference()
    └─ verify_overlap.py
        ├─ exec() 动态加载 check_func 并运行
        └─ llm_inference() 做语义 overlap 判断
```

`attack_point/` 是独立的“离线预处理子系统”，它的输出 `attack_point` 会成为 `run_main.py` 的输入之一：

```
attack_point/find_attack_point.py
├─ state_analysis.py  (找 injectable attributes)
├─ operation_analysis.py (找 operations read/write)
└─ utils/divide_env_class.py (从 env_class_code 切分状态类/环境类/操作函数)
```

---

## 3) 输入/输出数据结构契约

### 3.1 环境文件：`env_data` / `env_item`

环境文件顶层常见结构：

- `env_data: Dict[str, Any]`，以 `env_id` 为 key：`env_item = env_data[env_id]`

`env_item` 在 pipeline 中被读取的关键字段：

- **环境说明**
  - `environment_introduction: str`
  - `constraints_rules: List[str]`
  - `state_space_definition: List[Dict[str, Any]]`
  - `operation_list: List[Dict[str, Any]]`（每项含 `operation_name/operation_type/operation_description`）
- **可执行环境**
  - `tools: List[Dict[str, Any]]`（function-calling 的工具 schema）
  - `env_class_code: str`（环境 Python 源码字符串，运行时会 `exec()`）
- **攻击点（离线产物）**
  - `attack_point: List[AttackPoint]`

`AttackPoint` 元素结构（由 `attack_point/` 生成）：

```json
{
  "attribute": "ClassName.attribute",
  "write": ["write_op_1", "write_op_2"],
  "read":  ["read_op_1", "read_op_2"]
}
```

### 3.2 任务文件：`tasks_data` / `task_item`

任务文件顶层常见结构：

- `tasks_data: List[Dict[str, Any]]`，每个元素是一个 `task_item`

`run_main.py` 与注入执行环境会用到的典型字段：

- `env_id: str`
- `task_id: str`
- `task: str`（原始 benign 任务文本）
- `init_config: Dict[str, Any]`（初始状态/数据库）
- `env_class_name: str`（注入执行环境 reset 时用来从 `env_class_code` 提取环境类）
- `checklist_with_func: List[{"check_item": str, "check_func": str}]`

### 3.3 轨迹文件：`traj_entry`

轨迹文件顶层常见结构：

- `traj_data: List[Dict[str, Any]]`，每项含：
  - `task_info: {env_id, task_id, task, ...}`
  - `trajectory` 或 `messages`

关键中间产物：

- `action_names: List[str]`：从轨迹中抽取的真实 operation/tool 调用序列
  - 由 `attackpoint_task_match/find_operations.py:find_operations_for_task()` 产出

### 3.4 注入位置：`injection_location`

由 `IPI/gen_IPI_plan.py` 生成，结构：

```json
{
  "entity": "CommentInfo",
  "instance_id": "CMT1",
  "object_selector": "自然语言对象描述（辅助人类理解）",
  "attribute": "content",
  "reason": "为何能确认该字段在本次轨迹中会被读取"
}
```

约束（在 prompt 中强调）：

- `instance_id` **必须存在于当前 task 的 `init_config`**
- 并且该字段 **在本次任务轨迹中确实被读到**

### 3.5 注入计划：`injection_plan`

`IPI/IPI.py:execute_injection()` 的核心输入：

```json
{
  "injection_locations": [injection_location],
  "plan_steps": [
    {"step": 1, "description": "..."},
    {"step": 2, "description": "... new_value = original_content + <INJECTED_TASK> ..."}
  ]
}
```

语义契约：

- `plan_steps` 要求 **append-only**（不得覆盖原内容）
- 用 `<INJECTED_TASK>` 作为占位符，执行时替换为真实注入文本

### 3.6 注入执行输出：`execute_injection()` 返回值

`execute_injection()` 返回 dict，关键字段：

- `success: bool`
- `final_state: Dict[str, Any]`（注入后的环境状态快照）
- `injection_trajectory: List[Any]`（攻击代理执行轨迹）
- `attack_agent_result: Dict[str, Any]`（terminated/truncated/steps/trajectory 等摘要）

### 3.7 注入任务检查函数：`injected_checklist_with_func`

`IPI/gen_IPI_check_func.py:gen_IPI_check_func()` 返回：

```json
{
  "checklist": ["...", "..."],
  "checklist_with_func": [
    {"check_item": "...", "check_func": "def check_func(final_state): ... return True/False"},
    ...
  ]
}
```

`verification/verify_overlap.py` 会用 `exec()` 动态加载 `check_func` 并执行：

- `check_func(reference_state)`（把参考状态当 final_state 传入）

---

## 4) 数据流：哪些字段被新增/替换（输出样本结构）

`run_main.py` 产出的新任务条目是对原 `task_item` 的扩展/变形：

- `before_injected_config`: 原始 `init_config`（注入前）
- `init_config`: **替换为注入后的 `final_state`**（注入后）
- `injection_plan`: 单位置计划（`injection_locations` 长度=1）
- `injection_strategy`: 使用的策略名（例如 `combined`）
- `injected_task`: LLM 生成的注入任务 raw 文本（代码里叫 `injected_task_raw`）
- `injected_checklist`: 注入任务自然语言检查项
- `injected_checklist_with_func`: 注入任务 check_item + check_func
- `task_id`: 多位置时会变为 `xxx_IPI`, `xxx_IPI_2` ...

---

## 5) 各目录/文件职责（按流程讲解）

### 5.1 `run_main.py`（主编排器）

核心入口：`process_tasks(...)`  
职责：把所有子模块串起来，完成“原任务 -> 注入样本任务”的批处理。

**它做的事（按顺序）**：

- **加载输入数据**：读取 `env_file / task_file / traj_file`
- **逐任务处理**：
  - `env_id -> env_item`（拿到环境介绍/工具/攻击点等）
  - `(env_id, task_id) -> traj_entry`（在轨迹文件中定位该任务）
  - `traj_entry -> action_names`（抽取该任务真实调用过的 operation/tool 名称）
  - **攻击点可达性过滤**：只保留 `attack_point.write` 出现在 `action_names` 的点
  - **生成注入计划**：`gen_IPI_plan()` 产出多个候选注入位置（`injection_location`）及对应 `plan_steps`
- **对每个注入位置跑完整 pipeline（含重试）**：
  - 生成注入任务文本：`gen_IPI_task() -> injected_task_raw`
  - 注入策略包装：`apply_injection_strategy(injected_task_raw) -> injected_task`
  - 执行注入：`execute_injection(...) -> final_state`
  - LLM 验证注入是否写入：`verify_injection_with_llm(...) -> bool`
  - 生成注入任务检查函数：`gen_IPI_check_func(...) -> injected_checklist_with_func`
  - overlap 过滤：state overlap + semantic overlap
- **组装与落盘**：把注入后的 `final_state` 写回新任务的 `init_config`，补齐注入字段并写出 json

**关键输入/输出字段**：

- **输入**
  - `env_item["attack_point"]`
  - `task_item["task"] / task_item["init_config"] / task_item.get("checklist_with_func")`
  - `traj_entry`（对应 `(env_id, task_id)`）
- **输出（new_task 增量）**
  - `before_injected_config`（注入前 init_config）
  - `init_config = final_state`（注入后状态，替换原 init_config）
  - `injection_plan / injection_strategy`
  - `injected_task / injected_checklist / injected_checklist_with_func`

它调用的关键函数：

- `attackpoint_task_match/find_operations.py`
  - `find_traj_entry_for_task(env_id, task_id, traj_path)`
  - `find_operations_for_task(traj_entry) -> action_names`
- `attackpoint_task_match/find_match.py`
  - `filter_attack_points_by_actions(action_names, attack_points) -> filtered_attack_points`
- `IPI/gen_IPI_plan.py`
  - `gen_IPI_plan(..., attack_points=filtered_attack_points, traj_entry=traj_entry)`
- `IPI/gen_IPI_task.py`
  - `gen_IPI_task(...) -> injected_task_raw`
- `IPI/injection_strategy_utils`
  - `apply_injection_strategy(injected_task_raw, strategy_name=...) -> injected_task`
- `IPI/IPI.py`
  - `execute_injection(injection_plan, injected_task, ...) -> final_state`
- `verification/verify_injection.py`
  - `verify_injection_with_llm(injected_task, injection_location, final_state) -> bool`
- `IPI/gen_IPI_check_func.py`
  - `gen_IPI_check_func(injected_task, final_state, ...) -> injected_checklist_with_func`
- `verification/verify_overlap.py`
  - `check_checkfunc_overlap_with_state(reference_state, injected_checklist_with_func)`
  - `check_checkfunc_semantic_overlap_with_llm(original_checklist_with_func, injected_checklist_with_func)`

### 5.2 `attackpoint_task_match/`

这一层的目的很单一：**从真实轨迹中抽取“该任务实际执行过哪些操作”，并据此过滤攻击点**，确保注入点会被读取。

- `find_operations.py`：从 `traj_entry` 抽取真实 `action_names`
  - **关键函数**
    - `find_traj_entry_for_task(env_id, task_id, traj_path) -> traj_entry`
    - `find_operations_for_task(traj_entry) -> List[str]`
  - **兼容的轨迹结构**
    - `traj_entry["trajectory"]`：结构化步骤（step.action.name）
    - `traj_entry["messages"]`：支持 list 或 JSON 字符串；兼容 OpenAI `tool_calls` 与 `<tool_call>...</tool_call>` 文本格式
  - **输出**
    - `action_names: List[str]`（可能包含重复，保持出现顺序）

- `find_match.py`：按 `action_names` 过滤 attack_points
  - **关键函数**
    - `filter_attack_points_by_actions(action_names, attack_points) -> filtered_attack_points`
  - **规则**
    - 只保留满足：`attack_point["write"]` 中至少一个 op 名出现在 `action_names`
  - **输出**
    - `filtered_attack_points`（结构不变，仅数量减少）

- `filter_traj.py`：轨迹数据准备脚本
  - **作用**
    - 从大轨迹集中过滤出特定 `env_id/traj_type` 的子集，生成 `data/traj/*.json`

### 5.3 `IPI/gen_IPI_plan.py`（where + how：位置与注入步骤）

这份文件解决两件事：

- **where**：选定“本次执行一定会读到”的注入位置（`injection_locations`）
- **how**：给每个位置生成“append-only”的注入步骤（`plan_steps`）

核心点：

- **不是直接把整条轨迹喂给 LLM**，而是先用 `summarize_traj_for_injection_locations()` 抽取“attack_point.read 操作”的 step/参数/返回内容作为摘要，让 LLM 在摘要中定位“本次执行确实会读到”的实例/字段。
- `find_and_rank_injection_locations()`：LLM 输出并排序 `injection_locations`
  - 排序规则（prompt 中明示）：同一次 observation 内越靠后越好；跨 tool call step 越早越好
- `gen_injection_plan_steps()`：对某个 location 输出“append-only”的操作计划步骤（用 `<INJECTED_TASK>` 占位）
- `gen_IPI_plan()`：组合以上两步，返回 list：`[{injection_location, plan_steps}, ...]`

### 5.4 `IPI/gen_IPI_task.py`（what：注入任务文本）

这份文件负责生成“将被写入环境字段中的劫持任务文本”（注入内容本体）。

- **关键函数**
  - `gen_IPI_task(...) -> str`：返回 `injected_task_raw`

职责：生成 injected task 文本（劫持目标），并要求：

- 与原任务语义不相关/不重叠
- 在环境中可执行（需要写操作改变状态）
- 初始状态下不满足（避免 trivial）

随后由 `IPI/injection_strategy_utils` 进行策略包装。

### 5.5 `IPI/injection_strategy_utils/`（注入策略）

统一入口：`apply_injection_strategy(injected_task, strategy_name, **kwargs)`

策略类型：

- `hand_crafted.py`：`ignore/escape/fake_completion/combined/important_template`
- `multi_turn.py`：多轮注入风格
- `important_variants.py`：LLM 生成的 `<INFORMATION>` 变体（更“整段对话/模板化”的注入文本）

**输入/输出**

- 输入：`injected_task_raw: str`
- 输出：`injected_task: str`（加前缀/模板/多轮结构，增强劫持强度）

### 5.6 `IPI/IPI.py` + `IPI/attack_agent/` + `envscaler_env/`（执行注入）

`execute_injection()` 做了三件事：

1) 把 `plan_steps` 中的 `<INJECTED_TASK>` 替换成真实 `injected_task`，构造一个“注入执行任务描述”（告诉攻击代理要按步骤操作并把 payload 写入目标字段）。
2) 创建 `AttackNonConvEnv` 并把 “注入执行任务” 临时放入 `env.task_items`，让攻击代理只跑这一条。
3) 用 `TaskSolveAgent.run()` 驱动 LLM 调用环境工具，最后用 `get_state_info()` 抽取 `final_state`。

`AttackNonConvEnv` 的关键实现点：

- `load_env_and_instance()`：
  - `env_item["env_class_code"]` 被 `exec()` 成可用的环境类（`env_class_name` 来自 task_item）
  - 用 `init_config` 初始化实例（`init_env_instance`）
- `step()`：
  - prompt 模式：`parse_response()`/`parse_action()` 解析 `<tool_call>` 协议
  - fc 模式：读取 OpenAI `tool_calls` 并执行工具方法

#### 5.6.1 `IPI/IPI.py`

- **关键函数**
  - `execute_injection(...) -> Dict[str, Any]`
  - `_build_injection_task_description(plan_steps, injected_task, injection_location) -> str`
- **它在做什么**
  - 把 “注入计划 steps” + “注入文本” 拼成一个给攻击代理的任务，让代理按步骤调用环境工具把 payload 追加进目标字段
  - 创建 `AttackNonConvEnv` 并注入 `env.env_items`、临时替换 `env.task_items`，让代理只执行这一条注入任务
  - 执行结束后从 `env.env_instance` 抽取 `final_state`

#### 5.6.2 `IPI/attack_agent/task_solve_agent.py`

- **核心类**：`TaskSolveAgent`
- **职责**：管理 “LLM ↔ env.reset/step” 的交互循环并记录轨迹
- **关键方法**
  - `reset(task_index=0)`：获取初始 observation/tools/env_intro，并初始化 messages（system + user）
  - `step()`：一次推理（prompt 或 fc）→ env.step(action) → 更新 messages/trajectory
  - `run(...)`：循环 step，直到 terminated/truncated/max_steps
- **两种推理模式**
  - `infer_mode="prompt"`：LLM 输出 `<tool_call>...</tool_call>` 文本，需要 `parse_util` 解析
  - `infer_mode="fc"`：OpenAI function calling，LLM 输出结构化 `tool_calls`

#### 5.6.3 `IPI/attack_agent/agent_llm_inference.py`

- **职责**：封装 OpenAI SDK 的推理调用（流式、重试），并统一返回：
  - prompt 模式：`str`
  - fc 模式：`{"reasoning_content": str, "tool_calls": list, "content": str}`

#### 5.6.4 `envscaler_env/non_conv_env.py`

- **核心类**：`AttackNonConvEnv`
- **职责**：把 `env_item(env_class_code/tools/constraints)` + `task_item(init_config/task/env_class_name)` 变成可交互环境
- **关键方法**
  - `reset(task_index=0)`：动态 `exec(env_class_code)` 初始化环境实例，并返回 `(observation, info)`
  - `step(action)`：解析 action、执行工具方法、记录 `state_snapshot/state_diff`，或在 `chat_with_user` 终止时产出 `pred_final_state`

#### 5.6.5 `envscaler_env/utils/env_util.py` 与 `envscaler_env/utils/parse_util.py`

- `env_util.py`
  - `init_env_class()`：`exec` 源码字符串并取出类对象
  - `init_env_instance()`：把 `init_config` 写入实例属性
  - `get_state_info()`：把实例当前状态提取成 dict（即 `final_state`）
  - `get_state_diff()`：计算每步状态差异（调试/分析）
- `parse_util.py`
  - `parse_response()`：从 prompt 文本解析 `<think>` 与 `<tool_call>{json}</tool_call>`
  - `parse_action()`：统一成 `{name, arguments}`；纯文本会变成 `chat_with_user`

### 5.7 `IPI/gen_IPI_check_func.py`（把 injected task 变成 checklist 与可执行 check_func）

这份文件用于生成“注入任务的完成判定”，供过滤/评测/（后续可能的 reward）使用：

- **关键函数**
  - `gen_IPI_check_func(injected_task, state_config, env_introduction, ...) -> Dict`
- **内部依赖**
  - `IPI/task_check_util/gen_checklist.py`：从 task 文本生成 checklist（自然语言）
  - `IPI/task_check_util/gen_check_func.py`：为每个 check_item 生成 `def check_func(final_state): ...`
- **输出结构**
  - `{"checklist": [...], "checklist_with_func": [{"check_item": ..., "check_func": "code str"}, ...]}`

### 5.8 `verification/`（验证与 overlap 过滤）

这一层解决两个问题：

1) **注入是否真的写入目标字段**（否则样本无效）  
2) **注入任务是否“自带成立/与原任务重叠”**（否则样本质量差或污染训练）

- `verify_injection.py`
  - LLM 判断注入是否“写入目标字段”（允许近似/语义匹配，避免严格字符串比对过于脆弱）
- `verify_overlap.py`
  - `check_checkfunc_overlap_with_state()`：如果注入 check_func 在参考 state 上就为 True，说明注入任务“没做也成立”，视为无效样本
  - `check_checkfunc_semantic_overlap_with_llm()`：LLM 严格判断注入 checklist 与原任务 checklist 是否在“实体+字段+条件”层面重叠


### 5.9 `attack_point/`（离线子系统：生成 env_item["attack_point"]）

这套子系统不一定在 `run_main.py` 运行时执行，但它的输出 `attack_point` 是主 pipeline 的核心输入之一。

- `state_analysis.py`：分析状态类，找“可注入属性”
  - `analyze_injectable_attributes(env_data, ...) -> List[str]`
  - 输出形如：`["CommentInfo.content", "TaskInfo.description", ...]`
- `operation_analysis.py`：分析每个 operation，找“读/写哪些属性”
  - `analyze_operations(env_data, ...) -> Dict[op_name, {"read": [...], "write": [...]}]`
  - 严格约束：query 只读不写；state_change 只写不读
  - 过滤 create/delete 类操作
- `find_attack_point.py`：把可注入属性与读写分析结果匹配成攻击点
  - `match_attributes_to_operations(...) -> List[AttackPoint]`
  - `process_single_environment(...) -> env_data_with_attack_points`
- `utils/divide_env_class.py`：从 `env_class_code` 切分状态类/环境类/操作函数代码块
  - `divide_env_class(env_class_code) -> (state_class_list, env_class_block, operation_list)`



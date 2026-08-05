# 攻击点查找模块 (Attack Point Finder)

本模块用于在 Agent 环境中自动查找可以进行 IPI（指令提示注入）攻击的攻击点。

## 📋 目录结构

```
attack_point/
├── README.md                          # 本文档
├── 寻找攻击点设计.md                   # 设计文档（详细说明攻击点查找的设计思路）
├── state_analysis.py                  # 步骤1：状态类属性分析模块
├── operation_analysis.py               # 步骤2：操作读写分析模块
├── find_attack_point.py                # 步骤3：攻击点匹配与生成模块（主入口）
├── utils/                             # 工具模块
│   ├── divide_env_class.py           # 环境代码分割工具
│   └── call_llm.py                   # LLM 调用工具
└── example_*.json                      # 示例数据文件
```

## 🎯 核心功能

攻击点查找系统通过三个步骤，自动识别环境中可以进行 IPI 攻击的位置：

1. **步骤1：可注入属性分析** (`state_analysis.py`)
   - 分析所有状态类属性
   - 识别可以注入自由文本的属性（满足类型条件和语义条件）

2. **步骤2：操作读写分析** (`operation_analysis.py`)
   - 分析所有操作的读写行为
   - 确定每个操作读取和写入哪些状态类属性

3. **步骤3：攻击点匹配** (`find_attack_point.py`)
   - 整合前两步结果
   - 匹配可注入属性与读写操作
   - 生成最终的攻击点列表

## 📁 文件说明

### 1. `寻找攻击点设计.md`
**功能**：设计文档，详细说明攻击点查找的设计思路和实现要求。

**内容**：
- 攻击点的定义和条件
- 三个步骤的详细设计
- 输入输出格式规范

### 2. `state_analysis.py`
**功能**：分析状态类属性，找出可注入属性。

**核心函数**：
- `analyze_injectable_attributes(env_data, llm_provider, llm_model)`: 主函数，分析所有状态类
- `analyze_single_state_class(...)`: 分析单个状态类
- `parse_llm_response(response)`: 解析 LLM 响应

**输入**：
```python
env_data = {
    "environment_summary": str,           # 环境简要总结
    "environment_introduction": str,      # 环境详细介绍
    "constraints_rules": List[str],       # 约束规则列表
    "state_space_definition": List[Dict], # 状态空间定义（实体信息）
    "env_class_code": str                 # 环境类完整源代码
}
```

**输出**：
```python
[
    "ClassName.attribute1",  # 格式：类名.属性名
    "ClassName.attribute2",
    ...
]
```

**可注入属性的条件**：
1. **类型条件**：属性类型必须是字符串或包含字符串的复合类型（如 `List[str]`, `Dict[str, str]` 等）
2. **语义条件**：属性的语义意义允许注入自由文字（如任务描述、邮件内容等，排除 ID、时间戳、枚举值等）

### 3. `operation_analysis.py`
**功能**：分析操作的读写行为，确定每个操作读取和写入哪些状态类属性。

**核心函数**：
- `analyze_operations(env_data, llm_provider, llm_model)`: 主函数，分析所有操作
- `analyze_single_operation(...)`: 分析单个操作
- `is_delete_operation(...)`: 判断是否为删除操作（会被过滤）
- `parse_llm_response(response)`: 解析 LLM 响应

**输入**：
```python
env_data = {
    "environment_summary": str,           # 环境简要总结
    "environment_introduction": str,      # 环境详细介绍
    "constraints_rules": List[str],       # 约束规则列表
    "operation_list": List[Dict],         # 操作列表，每个元素包含：
                                          #   - operation_name: str
                                          #   - operation_description: str
                                          #   - operation_type: str ("query" 或 "state_change")
    "env_class_code": str                 # 环境类完整源代码
}
```

**输出**：
```python
{
    "operation_name1": {
        "read": ["ClassName.attribute1", "ClassName.attribute2", ...],
        "write": ["ClassName.attribute3", ...]
    },
    "operation_name2": {
        "read": [...],
        "write": [...]
    },
    ...
}
```

**规则**：
- **Query 操作**：只输出 `read`，`write` 必须为空列表 `[]`
- **State Change 操作**：只输出 `write`，`read` 必须为空列表 `[]`
- **删除操作**：会被自动过滤，不进行分析

### 4. `find_attack_point.py`
**功能**：整合前两步结果，生成攻击点列表（主入口模块）。

**核心函数**：
- `find_attack_points_for_all_environments(...)`: 处理 JSON 文件中的所有环境
- `process_single_environment(...)`: 处理单个环境
- `match_attributes_to_operations(...)`: 匹配可注入属性与操作

**输入**：
- JSON 文件路径（包含环境数据）
- 或直接传入环境数据字典（格式同 `state_analysis.py` 和 `operation_analysis.py`）

**输出**：
```python
{
    "env_id": {
        # ... 原始环境数据 ...
        "attack_point": [
            {
                "attribute": "ClassName.attribute1",
                "write": ["write_operation1", "write_operation2", ...],
                "read": ["read_operation1", "read_operation2", ...]
            },
            ...
        ]
    }
}
```

**攻击点的条件**：
- 必须是可注入属性（来自步骤1）
- 必须有至少一个写操作（来自步骤2）
- 必须有至少一个读操作（来自步骤2）

### 5. `utils/divide_env_class.py`
**功能**：工具模块，用于分割环境代码，提取状态类、环境类和操作函数。

**核心函数**：
- `divide_env_class(env_class_code)`: 分割环境代码
  - 返回：`(state_class_list, env_class_block, operation_list)`

**用途**：
- 被 `state_analysis.py` 和 `operation_analysis.py` 调用
- 用于从完整的环境代码中提取需要分析的部分

### 6. `utils/call_llm.py`
**功能**：工具模块，提供统一的 LLM 调用接口。

**核心函数**：
- `llm_inference(provider, model, messages, temperature, stop_strs, max_tokens)`: 调用 LLM

**支持的 Provider**：
- `openai`: OpenAI 兼容 API（通过环境变量 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 配置）

**用途**：
- 被 `state_analysis.py` 和 `operation_analysis.py` 调用
- 用于调用 LLM 进行属性分析和操作分析

## 🔄 模块关系与数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    find_attack_point.py                      │
│                    (主入口模块)                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────┐        ┌──────────────────┐
│ state_analysis.py│        │operation_analysis│
│   (步骤1)         │        │   .py (步骤2)     │
└────────┬──────────┘        └────────┬──────────┘
         │                            │
         │                            │
         ├──────────────┬─────────────┤
         │              │              │
         ▼              ▼              ▼
    ┌─────────────────────────────────────┐
    │      utils/divide_env_class.py       │
    │      (环境代码分割工具)                │
    └─────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │         utils/call_llm.py           │
    │         (LLM 调用工具)               │
    └─────────────────────────────────────┘
```

### 数据流示例

1. **输入**：JSON 文件（包含环境数据）
   ```json
   {
     "env_id": {
       "environment_summary": "...",
       "environment_introduction": "...",
       "constraints_rules": [...],
       "state_space_definition": [...],
       "operation_list": [...],
       "env_class_code": "..."
     }
   }
   ```

2. **步骤1处理**：`state_analysis.py` 分析状态类
   - 输入：`env_data`
   - 输出：`["CommentInfo.content", "PostInfo.title", ...]`

3. **步骤2处理**：`operation_analysis.py` 分析操作
   - 输入：`env_data`
   - 输出：`{"edit_comment": {"read": [...], "write": [...]}, ...}`

4. **步骤3处理**：`find_attack_point.py` 匹配并生成攻击点
   - 输入：步骤1和步骤2的输出
   - 输出：带有 `attack_point` 字段的环境数据

5. **最终输出**：JSON 文件（包含攻击点信息）
   ```json
   {
     "env_id": {
       "...": "...",
       "attack_point": [
         {
           "attribute": "CommentInfo.content",
           "write": ["edit_comment_content"],
           "read": ["list_comments_sorted", "get_comment_by_id"]
         }
       ]
     }
   }
   ```

## 🚀 使用方法

### 方法1：直接运行主模块

```python
from attack_point.find_attack_point import find_attack_points_for_all_environments

# 处理 JSON 文件中的所有环境
processed_environments = find_attack_points_for_all_environments(
    input_json_path="/path/to/example_env.json",
    output_json_path="/path/to/example_env_with_attack_points.json",  # 可选，自动生成
    llm_provider="openai",
    llm_model="gpt-4.1"
)
```

### 方法2：单独使用各个模块

```python
# 步骤1：分析可注入属性
from attack_point.state_analysis import analyze_injectable_attributes
injectable_attributes = analyze_injectable_attributes(
    env_data=env_data,
    llm_provider="openai",
    llm_model="gpt-4.1"
)

# 步骤2：分析操作读写情况
from attack_point.operation_analysis import analyze_operations
operation_read_write = analyze_operations(
    env_data=env_data,
    llm_provider="openai",
    llm_model="gpt-4.1"
)

# 步骤3：匹配攻击点
from attack_point.find_attack_point import match_attributes_to_operations
attack_points = match_attributes_to_operations(
    injectable_attributes=injectable_attributes,
    operation_read_write=operation_read_write
)
```



## 📊 输出格式说明

### 攻击点列表格式

每个攻击点是一个字典，包含以下字段：

- `attribute` (str): 可注入属性，格式为 `"ClassName.attribute_name"`
- `write` (List[str]): 可以写入该属性的操作列表
- `read` (List[str]): 可以读取该属性的操作列表

**示例**：
```json
{
  "attribute": "CommentInfo.content",
  "write": ["edit_comment_content", "create_comment"],
  "read": ["list_comments_sorted", "get_comment_by_id", "get_post_comments"]
}
```

## 🔍 示例文件

- `example_env.json`: 示例环境数据（输入）
- `example_env_with_attack_points.json`: 包含攻击点的示例环境数据（输出）
- `example_env2.json`: 另一个示例环境数据
- `example_env2_with_attack_points.json`: 对应的输出文件


## 📚 相关文档

- `寻找攻击点设计.md`: 详细的设计文档
- 各模块的 docstring: 包含详细的函数说明和参数说明


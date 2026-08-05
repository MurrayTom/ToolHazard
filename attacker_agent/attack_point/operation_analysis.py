"""
操作分析模块

本模块的主要功能是分析环境中的每个操作，确定它们会读取或写入哪些状态类的哪些属性。
这对于后续的攻击点匹配非常重要，因为可注入属性需要有读写操作才能成为攻击点。
"""
import json
import re
import sys
import os
from typing import List, Dict, Any, Optional

# 添加父目录到Python路径，以便导入工具模块
# 根目录是 EnvScaler/attacker，当前文件在 attack_point/ 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
parent_dir = os.path.dirname(current_dir)  # 获取父目录，即 EnvScaler/attacker
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)  # 将父目录添加到Python搜索路径的最前面

# 导入工具模块
from attack_point.utils.divide_env_class import divide_env_class  # 用于分割环境代码，提取状态类、环境类和操作
from attack_point.utils.call_llm import llm_inference  # 用于调用大语言模型进行分析

# ============================================================================
# LLM提示词定义
# ============================================================================

# 系统提示词：定义任务和目标
system_prompt = '''You are a code analysis expert specializing in analyzing operation functions to identify which state class attributes they read and write.

Given an Agent's environment specification, including:
- Environment summary and introduction
- Constraint rules
- All state class definitions
- Environment class code (including __init__)
- A specific operation function to analyze

Your task is to analyze the operation function and determine:
1. **Read operations**: Which state class attributes are included in the returned data of this operation (query outputs)
2. **Write operations**: Which state class attributes are written/modified by this operation (state change effects)

Important rules:
- You MUST use the provided **operation_type** to constrain outputs:
  - If operation_type == "query": output should have **read** only; **write MUST be []**.
  - If operation_type == "state_change": output should have **write** only; **read MUST be []**.
- For **query** operations:
  - "read" is determined ONLY by what the function returns in its success response ("data" or returned structure).
  - If it returns a complete state class object (e.g., returns TaskInfo), then it reads ALL attributes of that class.
  - If it returns only specific fields/attributes, list ONLY those attributes/fields that are actually included in the returned data.
  - Any internal access for validation/filtering/lookups that does NOT appear in the returned data should NOT be counted as "read".
- For **state_change** operations:
  - Only list attributes that are written/modified/created/updated as "write".
  - Any internal reads used for validation or lookups should NOT be counted as "read" (read MUST be []).
- Write operations include: modifying attributes, creating new objects with attributes, updating values and so on
- Attributes should be in the format "ClassName.attribute_name" (e.g., "TaskInfo.description")

You must first analyze the operation function step by step, explaining which attributes are read and which are written. Then, at the end, provide your final answer in the following format:

```json
{
  "read": ["ClassName.attribute1", "ClassName.attribute2", ...],
  "write": ["ClassName.attribute3", "ClassName.attribute4", ...]
}
```

Make sure to wrap the JSON object in a code block with ```json and ``` markers.'''

# 示例输入（Few-shot learning示例）
input_case_1 = '''Based on the following environment specification, analyze the operation function and identify which state class attributes it reads and writes.

### Environment Summary
Task tracking system

### Environment Introduction
A task tracking system is a stateful environment designed to manage, organize, and monitor individual tasks within a workflow or project. It maintains persistent records of task metadata such as IDs, descriptions, statuses, and timestamps, and supports operations like creation, status updates, and retrieval.

### Constraints Rules
['A task can only be marked as "completed" if its current status is not already "completed".', 'The completed_at timestamp must be set when a task\'s status is updated to "completed".', 'The status field must be one of a predefined set of values (e.g., "pending", "in progress", "completed", "blocked").', 'Task IDs must be unique within the system.']

### State Classes
```python
class TaskInfo(TypedDict):
    task_id: str
    description: str
    status: str  # Must be one of: "pending", "in progress", "completed", "blocked"
    created_at: float  # Unix timestamp
    priority: str
    assigned_to: Optional[str]  # user_id

class UserInfo(TypedDict):
    _id: str
    name: str
    role: str

class ProjectInfo(TypedDict):
    project_id: str
    name: str
    start_date: float
    end_date: Optional[float]
    tasks: List[str]  # List of task_ids
```

### Environment Class Code
```python
class TaskTrackingSystem:
    def __init__(self):
        """
        A task tracking system to manage tasks, users, and projects.
        """
        # Tasks: {{task_id: TaskInfo}}
        self.tasks: Dict[str, TaskInfo] = {{}}

        # Users: {{user_id: UserInfo}}
        self.users: Dict[str, UserInfo] = {{}}

        # Projects: {{project_id: ProjectInfo}}
        self.projects: Dict[str, ProjectInfo] = {{}}
```

### Operation to Analyze
**Operation Name:** get_task_by_id
**Operation Description:** Retrieve full details of a task given its task_id.
**Operation Type:** query

**Operation Code:**
```python
def get_task_by_id(self, task_id: str) -> dict:
    """
    Retrieve full details of a task given its unique task ID.

    Args:
        task_id (str): The unique identifier of the task.

    Returns:
        dict: {{
            "success": True,
            "data": TaskInfo  # Full task information
        }}
        or
        {{
            "success": False,
            "error": str  # Description of the error, e.g. "Task not found"
        }}

    Constraints:
        - The task_id must exist in the system.
        - Task IDs are unique; only one task will match.
    """
    if not task_id:
        return {{ "success": False, "error": "Task ID cannot be empty" }}

    task = self.tasks.get(task_id)
    if task is None:
        return {{ "success": False, "error": "Task not found" }}

    return {{ "success": True, "data": task }}
```'''

# 示例输出（包含思考过程和最终JSON答案）
output_case_1 = '''Let me analyze the `get_task_by_id` operation step by step:

**Step 1: Understanding the operation**
This operation retrieves a task by its ID and returns the full task information.

**Step 2: Analyzing read operations**
- The operation accesses `self.tasks.get(task_id)`, which reads from the tasks dictionary
- It retrieves a complete `TaskInfo` object (the variable `task`)
- The return statement `return {{ "success": True, "data": task }}` returns the entire TaskInfo object
- Since it returns the complete TaskInfo object, it effectively reads ALL attributes of TaskInfo:
  - TaskInfo.task_id (read when accessing the task object)
  - TaskInfo.description (read when returning the complete object)
  - TaskInfo.status (read when returning the complete object)
  - TaskInfo.created_at (read when returning the complete object)
  - TaskInfo.priority (read when returning the complete object)
  - TaskInfo.assigned_to (read when returning the complete object)

**Step 3: Analyzing write operations**
- The operation does not modify any state
- It only reads and returns data
- No attributes are written

**Conclusion:**
- Read: All TaskInfo attributes (since the complete object is returned)
- Write: None

Final answer:

```json
{
  "read": ["TaskInfo.task_id", "TaskInfo.description", "TaskInfo.status", "TaskInfo.created_at", "TaskInfo.priority", "TaskInfo.assigned_to"],
  "write": []
}
```'''

# 输入模板
input_template = '''Based on the following environment specification, analyze the operation function and identify which state class attributes it reads and writes.

### Environment Summary
{environment_summary}

### Environment Introduction
{environment_introduction}

### Constraints Rules
{constraints_rules}

### State Classes
```python
{state_classes_code}
```

### Environment Class Code
```python
{env_class_code}
```

### Operation to Analyze
**Operation Name:** {operation_name}
**Operation Description:** {operation_description}
**Operation Type:** {operation_type}

**Operation Code:**
```python
{operation_code}
```'''


def _classify_operation_type_with_llm(
    operation_name: str,
    operation_description: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1",
) -> Optional[str]:
    """
    使用大模型判断操作类型：先思考再返回最终答案。
    - create：创造新实例（如新建任务、注册用户、创建资源）
    - delete：删除一个实例（如删除任务、移除用户、销毁资源）
    - other：既非创造新实例也非删除实例（读、更新、部分修改等）
    若调用失败或未按规定输出，返回 None，由调用方退回到关键词检查。
    """
    
    prompt = f"""You are an expert on API operation semantics.

Given the following operation, classify it as exactly one of: "create", "delete", "other".

Definitions:
- **create**: The operation **creates a new instance** (e.g., new task, new user, new resource). It adds a new entity to the system.
- **delete**: The operation **deletes/removes an existing instance** (e.g., delete a task, remove a user, destroy a resource). It removes an entity from the system.
- **other**: Neither of the above. E.g. read-only, update an existing instance, modification, or miscellaneous.

First, think step by step: does this operation primarily create a new instance, delete an instance, or neither? Then give your final answer in the required format.

Output format:
- You may output your reasoning first (optional).
- Your final answer MUST be wrapped in <answer></answer> tags, with exactly one word inside: "create", "delete", or "other".
- Example: <answer>create</answer> or <answer>other</answer>

Operation name: {operation_name}
Operation description: {operation_description}
"""
    messages = [
        {"role": "system", "content": "You classify operations into create (creates new instance), delete (deletes an instance), or other. Think first, then output the final answer inside <answer></answer>, e.g. <answer>create</answer>."},
        {"role": "user", "content": prompt},
    ]
    try:
        output = llm_inference(
            provider=llm_provider,
            model=llm_model,
            messages=messages,
            temperature=0.0,
        )
    except Exception as e:
        print(f"[WARN] _classify_operation_type_with_llm failed: {e!r}")
        return None

    if not output or not isinstance(output, str):
        return None

    text = output.strip().lower()

    # 1) 优先从 <answer>...</answer> 中抽取最终答案
    if "<answer>" in text and "</answer>" in text:
        start = text.find("<answer>") + len("<answer>")
        end = text.find("</answer>", start)
        if end > start:
            inner = text[start:end].strip().rstrip(".:")
            if inner == "create":
                return "create"
            if inner == "delete":
                return "delete"
            if inner == "other":
                return "other"

    # 2) 未按规定用 <answer> 包裹时，退回到启发式：最后一行 / final answer 行 / 最后一次出现的标签
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1].rstrip(".:")
        if last == "create":
            return "create"
        if last == "delete":
            return "delete"
        if last == "other":
            return "other"

    for line in lines:
        for prefix in ("final answer:", "final answer：", "答案:", "答案："):
            if prefix in line:
                rest = line.split(prefix, 1)[-1].strip().rstrip(".:")
                if rest == "create":
                    return "create"
                if rest == "delete":
                    return "delete"
                if rest == "other":
                    return "other"

    last_create = text.rfind("create")
    last_delete = text.rfind("delete")
    last_other = text.rfind("other")
    idx = max(last_create, last_delete, last_other)
    if idx >= 0:
        if idx == last_create:
            return "create"
        if idx == last_delete:
            return "delete"
        if idx == last_other:
            return "other"

    return None


def _classify_operation_type(
    operation_name: str,
    operation_description: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1",
) -> Optional[str]:
    """
    优先用大模型判断操作类型（create/delete/other），失败或未按规定输出时返回 None，
    由调用方退回到 is_delete_operation / is_create_operation 的关键词检查。
    """
    result = _classify_operation_type_with_llm(
        operation_name=operation_name,
        operation_description=operation_description,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    if result in ("create", "delete", "other"):
        return result
    return None


def is_delete_operation(operation_name: str, operation_description: str) -> bool:
    """
    判断一个操作是否是删除操作
    
    删除操作应该被过滤掉，因为删除操作不能用于注入攻击（写操作不能是删除）。
    删除操作指的是删除整个实体（如删除任务、删除用户）。
    注意：这里不对包含 "remove" 的操作做细分判断（例如 remove_xxx_from_yyy），
    仅识别明确的 delete/destroy/erase/remove 等删除关键词。
    
    Args:
        operation_name: 操作名称
        operation_description: 操作描述
    
    Returns:
        bool: 如果是删除操作则返回True，否则返回False
    """
    operation_text = f"{operation_name} {operation_description}".lower()
    
    # 明确的删除关键词（表示删除整个实体）
    delete_keywords = ["delete", "destroy", "erase", "remove"]
    
    # 检查是否包含明确的删除关键词
    for keyword in delete_keywords:
        if keyword in operation_text:
            return True
    
    return False


def is_create_operation(operation_name: str, operation_description: str) -> bool:
    """
    判断一个操作是否是创建操作
    
    创建操作同样不适合作为攻击点（主要用于新建实体，而不是对现有状态做有意义的读写组合），
    因此在操作分析阶段将其过滤掉，避免后续被当成攻击候选。
    
    这里以操作名称/描述中的关键词进行启发式判断：
    - 英文关键词: create, register, signup, sign_up, initialize, init, new_
    - 中文关键词: 创建
    """
    operation_text = f"{operation_name} {operation_description}".lower()
    
    create_keywords = [
        "create",
        "register",
        "signup",
        "sign_up",
        "initialize",
        "init",
        "new_",
        "send",
        "创建",
    ]
    
    for keyword in create_keywords:
        if keyword in operation_text:
            return True
    
    return False


def build_messages_for_operation(
    operation_info: Dict,
    operation_code: str,
    state_classes_code: str,
    environment_summary: str,
    environment_introduction: str,
    constraints_rules: List[str],
    env_class_code: str
) -> List[Dict]:
    """
    为单个操作构建LLM消息列表（使用Few-shot learning）
    
    使用system prompt + 示例 + 实际任务的格式，让LLM更好地理解任务要求。
    
    Args:
        operation_info: 操作信息字典，包含 operation_name, operation_description, operation_type
        operation_code: 操作的源代码
        state_classes_code: 所有状态类的源代码（合并后的字符串）
        environment_summary: 环境的简要总结
        environment_introduction: 环境的详细介绍
        constraints_rules: 约束规则列表
        env_class_code: 完整的环境类代码（包含类定义和__init__方法）
    
    Returns:
        List[Dict]: LLM消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}, ...]
    """
    # 将约束规则列表转换为字符串格式（用于JSON序列化）
    constraints_rules_str = json.dumps(constraints_rules, ensure_ascii=False, indent=2)
    
    # 构建实际任务的输入内容
    input_content = input_template.format(
        environment_summary=environment_summary,
        environment_introduction=environment_introduction,
        constraints_rules=constraints_rules_str,
        state_classes_code=state_classes_code,
        env_class_code=env_class_code,
        operation_name=operation_info.get("operation_name", ""),
        operation_description=operation_info.get("operation_description", ""),
        operation_type=operation_info.get("operation_type", ""),
        operation_code=operation_code
    )
    
    # 构建消息列表：system prompt + 示例（few-shot） + 实际任务
    messages = [
        {"role": "system", "content": system_prompt},  # 系统提示词，定义任务
        {"role": "user", "content": input_case_1},     # 示例输入
        {"role": "assistant", "content": output_case_1},  # 示例输出
        {"role": "user", "content": input_content}     # 实际任务输入
    ]
    
    return messages


def parse_llm_response(response: str) -> Dict[str, List[str]]:
    """
    解析LLM的响应，提取read和write属性列表
    
    LLM会先进行思考分析，最后以 ```json {...} ``` 格式给出答案。
    本函数从响应中提取JSON代码块，并解析出read和write列表。
    
    Args:
        response: LLM返回的响应字符串（可能包含思考过程和JSON代码块）
    
    Returns:
        Dict[str, List[str]]: 包含"read"和"write"键的字典，每个值是属性名称列表
            如果解析失败，返回 {"read": [], "write": []}
    """
    # 方法1：尝试提取 ```json ... ``` 代码块中的内容（优先）
    json_code_block_pattern = r'```json\s*(.*?)\s*```'
    json_block_match = re.search(json_code_block_pattern, response, re.DOTALL)
    if json_block_match:
        try:
            json_content = json_block_match.group(1).strip()
            result = json.loads(json_content)
            if isinstance(result, dict):
                read_list = result.get("read", [])
                write_list = result.get("write", [])
                # 确保read和write都是列表
                if isinstance(read_list, list) and isinstance(write_list, list):
                    return {
                        "read": [str(attr) for attr in read_list if isinstance(attr, str)],
                        "write": [str(attr) for attr in write_list if isinstance(attr, str)]
                    }
        except json.JSONDecodeError:
            pass
    
    # 方法2：尝试提取 ``` ... ``` 代码块中的内容（可能是json但没有json标记）
    code_block_pattern = r'```\s*(.*?)\s*```'
    code_block_match = re.search(code_block_pattern, response, re.DOTALL)
    if code_block_match:
        try:
            code_content = code_block_match.group(1).strip()
            # 检查是否以 { 开头，可能是JSON对象
            if code_content.startswith('{'):
                result = json.loads(code_content)
                if isinstance(result, dict):
                    read_list = result.get("read", [])
                    write_list = result.get("write", [])
                    if isinstance(read_list, list) and isinstance(write_list, list):
                        return {
                            "read": [str(attr) for attr in read_list if isinstance(attr, str)],
                            "write": [str(attr) for attr in write_list if isinstance(attr, str)]
                        }
        except json.JSONDecodeError:
            pass
    
    # 方法3：尝试从响应中提取JSON对象模式（直接查找 {...}）
    json_object_pattern = r'\{.*?"read".*?"write".*?\}'
    json_match = re.search(json_object_pattern, response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                read_list = result.get("read", [])
                write_list = result.get("write", [])
                if isinstance(read_list, list) and isinstance(write_list, list):
                    return {
                        "read": [str(attr) for attr in read_list if isinstance(attr, str)],
                        "write": [str(attr) for attr in write_list if isinstance(attr, str)]
                    }
        except json.JSONDecodeError:
            pass
    
    # 如果所有解析方法都失败，返回空列表
    return {"read": [], "write": []}


def analyze_single_operation(
    operation_info: Dict,
    operation_code: str,
    state_classes_code: str,
    environment_summary: str,
    environment_introduction: str,
    constraints_rules: List[str],
    env_class_code: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1"
) -> Dict[str, List[str]]:
    """
    分析单个操作，找出它读取和写入的属性
    
    这是核心分析函数，对单个操作进行完整分析：
    1. 构建LLM消息列表
    2. 调用LLM进行分析
    3. 解析LLM响应
    4. 返回read和write列表
    
    Args:
        operation_info: 操作信息字典，包含 operation_name, operation_description, operation_type
        operation_code: 操作的源代码
        state_classes_code: 所有状态类的源代码（合并后的字符串）
        environment_summary: 环境的简要总结
        environment_introduction: 环境的详细介绍
        constraints_rules: 约束规则列表
        env_class_code: 完整的环境类代码（包含类定义和__init__方法）
        llm_provider: LLM提供商名称，默认为"openai"
        llm_model: LLM模型名称，默认为"gpt-4.1"
    
    Returns:
        Dict[str, List[str]]: 包含"read"和"write"键的字典，每个值是属性名称列表
            格式为 {"read": ["TaskInfo.description", ...], "write": ["TaskInfo.status", ...]}
    """
    # 步骤1：构建LLM消息列表（使用Few-shot learning格式）
    messages = build_messages_for_operation(
        operation_info=operation_info,
        operation_code=operation_code,
        state_classes_code=state_classes_code,
        environment_summary=environment_summary,
        environment_introduction=environment_introduction,
        constraints_rules=constraints_rules,
        env_class_code=env_class_code
    )
    
    # 步骤2：调用LLM进行分析
    response = llm_inference(
        provider=llm_provider,
        model=llm_model,
        messages=messages,
        temperature=0.5  # 温度设为0.5，使输出有一定随机性但保持相对稳定
    )
    
    # 步骤3：解析LLM响应
    result = parse_llm_response(response)
    
    return result


def analyze_operations(
    env_data: Dict[str, Any],
    llm_provider: str = "openai",
    llm_model: str = "gpt-4.1"
) -> Dict[str, Dict[str, List[str]]]:
    """
    分析所有操作，找出每个操作读取和写入的属性（主函数）
    
    这是模块的主入口函数，负责：
    1. 从环境数据中提取必要信息
    2. 分割环境代码，获取所有状态类、环境类和操作
    3. 过滤删除操作
    4. 对每个操作进行分析
    5. 合并所有结果并返回
    
    Args:
        env_data: 包含环境数据的字典，必须包含以下键：
            - environment_summary: 环境的简要总结（字符串）
            - environment_introduction: 环境的详细介绍（字符串）
            - constraints_rules: 约束规则列表（字符串列表）
            - operation_list: 操作列表，每个元素包含 operation_name, operation_description, operation_type
            - env_class_code: 环境类的完整源代码（字符串）
        llm_provider: LLM提供商名称，默认为"openai"
        llm_model: LLM模型名称，默认为"gpt-4.1"
    
    Returns:
        Dict[str, Dict[str, List[str]]]: 操作读写情况字典
            格式为 {
                "operation_name": {
                    "read": ["ClassName.attribute1", ...],
                    "write": ["ClassName.attribute2", ...]
                },
                ...
            }
    """
    # 步骤1：从环境数据中提取所需字段
    environment_summary = env_data.get("environment_summary", "")
    environment_introduction = env_data.get("environment_introduction", "")
    constraints_rules = env_data.get("constraints_rules", [])
    operation_list = env_data.get("operation_list", [])
    env_class_code = env_data.get("env_class_code", "")
    
    # 步骤2：分割环境类代码，获取状态类列表、环境类和操作列表
    state_class_list, env_class_block, operation_code_list = divide_env_class(env_class_code)
    
    # 步骤3：从环境类块中提取环境类代码
    if env_class_block:
        env_class_def_code = env_class_block.get("code", "")
    else:
        env_class_def_code = ""
    
    # 步骤4：合并所有状态类代码为一个字符串
    state_classes_code = "\n\n".join([cls.get("code", "") for cls in state_class_list])
    
    # 步骤5：创建操作代码映射（operation_name -> operation_code）
    operation_code_map = {}
    for op_code_info in operation_code_list:
        op_name = op_code_info.get("function_name", "")
        op_code = op_code_info.get("code", "")
        if op_name and op_code:
            operation_code_map[op_name] = op_code
    
    # 步骤6：过滤删除操作，并分析剩余操作
    all_operation_read_write = {}  # 存储所有操作的读写情况
    
    for operation_info in operation_list:
        operation_name = operation_info.get("operation_name", "")
        operation_description = operation_info.get("operation_description", "")
        
        # 优先用大模型判断是否为 create/delete；失败或未按规定输出则退回关键词检查
        classification = _classify_operation_type(
            operation_name, operation_description,
            llm_provider=llm_provider, llm_model=llm_model,
        )
        if classification is not None:
            if classification == "delete":
                print(f"Skipping delete operation: {operation_name}")
                continue
            if classification == "create":
                print(f"Skipping create operation: {operation_name}")
                continue
        else:
            # 退回到原有关键词检查
            if is_delete_operation(operation_name, operation_description):
                print(f"Skipping delete operation: {operation_name}")
                continue
            if is_create_operation(operation_name, operation_description):
                print(f"Skipping create operation: {operation_name}")
                continue
        
        # 获取操作代码
        operation_code = operation_code_map.get(operation_name, "")
        if not operation_code:
            print(f"Warning: No code found for operation: {operation_name}")
            continue
        
        # 打印当前正在分析的操作名称
        print(f"Analyzing operation: {operation_name}")
        
        # 调用分析函数，分析单个操作
        read_write = analyze_single_operation(
            operation_info=operation_info,
            operation_code=operation_code,
            state_classes_code=state_classes_code,
            environment_summary=environment_summary,
            environment_introduction=environment_introduction,
            constraints_rules=constraints_rules,
            env_class_code=env_class_def_code,
            llm_provider=llm_provider,
            llm_model=llm_model
        )
        
        # 存储结果
        all_operation_read_write[operation_name] = read_write
        
        # 打印当前操作的分析结果
        print(f"  Read: {len(read_write['read'])} attributes - {read_write['read']}")
        print(f"  Write: {len(read_write['write'])} attributes - {read_write['write']}")
    
    # 返回所有操作的读写情况
    return all_operation_read_write


if __name__ == "__main__":
    """
    主函数：用于测试操作分析功能
    
    当直接运行此脚本时，会：
    1. 加载示例环境数据（example_env.json）
    2. 提取环境数据
    3. 调用分析函数找出所有操作的读写情况
    4. 打印分析结果
    """
    # 测试用的JSON文件路径（包含环境定义数据）
    json_path = "/home/mouyutao/yangpengfei/EnvScaler/attacker/attack_point/example_env4_with_attack_points.json"
    
    # 读取JSON文件
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 获取第一个环境（假设字典中只有一个环境）
    env_id = list(data.keys())[0]  # 获取第一个环境ID
    env_data = data[env_id]  # 获取对应的环境数据
    
    # 打印分析开始信息
    print(f"Analyzing operations for environment: {env_id}")
    print("=" * 80)  # 打印分隔线
    
    # 调用主分析函数，分析所有操作的读写情况
    operation_read_write = analyze_operations(
        env_data=env_data,  # 传入环境数据
        llm_provider="openai",  # 使用OpenAI作为LLM提供商
        llm_model="gpt-4.1"  # 使用gpt-4.1模型
    )
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("Final Results:")
    print("=" * 80)
    print(f"Total operations analyzed: {len(operation_read_write)}")
    for op_name, read_write in operation_read_write.items():
        print(f"\n{op_name}:")
        print(f"  Read ({len(read_write['read'])}): {read_write['read']}")
        print(f"  Write ({len(read_write['write'])}): {read_write['write']}")




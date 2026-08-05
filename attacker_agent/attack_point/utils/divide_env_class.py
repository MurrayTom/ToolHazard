"""
Parse environment class name and definition
"""
import re
import json

def parse_env_class_name(str):
    """Parse environment class name"""
    # last class
    content = str.split("def __init__")[0]
    pattern = r"class\s+(\w+)\s*:"
    matches = re.findall(pattern, content)
    if matches:
        return matches[-1].strip()
    else:
        return str.split("def __init__")[0].split("class ")[-1].split(":")[0].strip()


def parse_env_class_def(src_str: str, class_name: str) -> dict:
    """Parse environment class definition"""
    part1=src_str.split(f"class {class_name}")[0]
    part2=src_str.split(f"class {class_name}")[1].split("def __init__")[0]
    part3=src_str.split(f"class {class_name}")[1].split("def __init__")[1].split("def ")[0]
    content=part1+f"class {class_name}"+part2+"def __init__"+part3
    return content


def remove_header_imports(src: str) -> str:
    """
    Remove the leading import block (no import / from lines before the first class).
    只删开头连续的 import/from 语句，直到遇到第一个 class 定义或其他代码。
    """
    lines = src.splitlines()
    new_lines = []
    seen_class = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("class "):
            seen_class = True
        if not seen_class:
            # 在第一个 class 之前，跳过所有 import/from 行
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
        new_lines.append(line)
    return "\n".join(new_lines)


def split_classes(src: str):
    """
    Split source into class blocks: from each 'class' line to the line
    before the next 'class' (or end of string).
    返回 list[{"class_name": str, "code": str}].
    """
    lines = src.splitlines()
    class_indices = []
    class_pattern = re.compile(r'^\s*class\s+(\w+)\s*')

    for i, line in enumerate(lines):
        m = class_pattern.match(line)
        if m:
            class_indices.append((i, m.group(1)))

    blocks = []
    for idx, (start, name) in enumerate(class_indices):
        if idx + 1 < len(class_indices):
            end = class_indices[idx + 1][0] - 1
        else:
            end = len(lines) - 1
        code_block = "\n".join(lines[start:end+1])
        blocks.append({"class_name": name, "code": code_block})
    return blocks


def split_env_methods(src: str, env_class_name: str):
    """
    在指定环境类中，划分 __init__ 之后的每个操作函数块。
    每个块从 'def xxx(...)' 开始，到下一个同缩进层级的 'def' 之前结束。
    返回 list[{"method_name": str, "code": str}].
    """
    lines = src.splitlines()

    # 1. 找到环境类的起始和结束行号
    class_pattern = re.compile(rf'^\s*class\s+{re.escape(env_class_name)}\b')
    top_class_pattern = re.compile(r'^\s*class\s+\w+')
    def_pattern = re.compile(r'^\s{4}def\s+(\w+)\s*\(')  # 假定方法缩进 4 空格

    env_class_start = None
    for i, line in enumerate(lines):
        if class_pattern.match(line):
            env_class_start = i
            break
    if env_class_start is None:
        return []

    # 找到类的结束行：下一个顶格 class 之前一行，或文件末尾
    env_class_end = len(lines) - 1
    for i in range(env_class_start + 1, len(lines)):
        if top_class_pattern.match(lines[i]):
            env_class_end = i - 1
            break

    # 2. 收集类体内所有方法的起始行
    method_starts = []  # (line_idx, function_name)
    for i in range(env_class_start + 1, env_class_end + 1):
        m = def_pattern.match(lines[i])
        if m:
            method_starts.append((i, m.group(1)))

    if not method_starts:
        return []

    # 3. 根据起始行计算每个方法的结束行（下一个方法的前一行，或类结束）
    method_blocks = []
    for idx, (start, name) in enumerate(method_starts):
        if idx + 1 < len(method_starts):
            end = method_starts[idx + 1][0] - 1
        else:
            end = env_class_end
        code_block = "\n".join(lines[start:end + 1])
        method_blocks.append({"function_name": name, "code": code_block})

    # 4. 只保留 __init__ 之后的“操作函数”
    return [b for b in method_blocks if b["function_name"] != "__init__"]


def _load_env_class_code_from_json(json_path: str) -> str:
    """Load env_class_code string from example_env.json-like file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Assume single env entry, take the first value
    if isinstance(data, dict) and data:
        first_env = next(iter(data.values()))
        return first_env.get("env_class_code", "")
    raise ValueError("Invalid env json structure")


def divide_env_class(env_class_code: str):
    """
    对 env_class_code 做完整划分，返回：
    - state_class_list: 状态类列表（不包括环境类），来自按 class 切分后的前 N-1 个块
    - env_class: 环境类块（最后一个 class 块）
    - operation_list: 环境类中 __init__ 之后的操作函数列表

    返回三个元素：
    - state_class_list: [ { "class_name": str, "code": str }, ... ]
    - env_class: { "class_name": str, "code": str } or None
    - operation_list: [ { "function_name": str, "code": str }, ... ]
    """
    env_class_name = parse_env_class_name(env_class_code)
    # 取出 class 定义 + __init__，再去掉头部 import
    env_class_def = parse_env_class_def(env_class_code, env_class_name)
    env_class_def_no_imports = remove_header_imports(env_class_def)

    # 按 class 切分，最后一个视为环境类，其余是状态类
    class_blocks = split_classes(env_class_def_no_imports)
    if not class_blocks:
        return [], None, []

    if len(class_blocks) == 1:
        state_class_list = []
        env_class_block = class_blocks[0]
    else:
        state_class_list = class_blocks[:-1]
        env_class_block = class_blocks[-1]

    # 环境类中的操作函数（__init__ 之后）
    operation_list = split_env_methods(env_class_code, env_class_name)

    return state_class_list, env_class_block, operation_list


if __name__ == "__main__":
    json_path = "/home/mouyutao/yangpengfei/EnvScaler/attacker/attack_point/example_env2.json"
    env_class_code = _load_env_class_code_from_json(json_path)

    state_class_list, env_class_block, operation_list = divide_env_class(env_class_code)

    print("=== State classes ===")
    for cls in state_class_list:
        print(f"\n--- state class {cls['class_name']} ---")
        print(cls["code"])

    print("\n=== Env class ===")
    print(env_class_block["class_name"] if env_class_block else None)
    if env_class_block:
        print(env_class_block["code"])

    print("\n=== Operations (functions after __init__) ===")
    for func in operation_list:
        print(f"\n--- function {func['function_name']} ---")
        print(func["code"])


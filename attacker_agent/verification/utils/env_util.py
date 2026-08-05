"""
环境初始化和状态管理工具函数模块

本模块提供以下核心功能：
1. 从源代码字符串动态创建环境类
2. 创建环境实例并应用初始配置
3. 获取和比较环境状态
4. 从环境实例提取与初始配置格式相同的最终配置
5. 动态执行任务完成检查函数
"""
import types  # 用于动态创建模块对象
from copy import deepcopy  # 深拷贝工具，避免修改原始对象


def init_env_class(env_class_code: str, env_class_name: str):
    """
    从源代码字符串动态创建环境类
    
    功能说明：
    - 接收环境类的 Python 源代码字符串（env_class_code）
    - 动态执行这段代码，创建环境类对象
    - 返回可以实例化的类对象
    
    参数：
        env_class_code (str): 环境类的 Python 源代码字符串
                              例如："class PharmacyEnv:\n    def __init__(self, config):..."
        env_class_name (str): 要提取的类名称（必须在 env_class_code 中定义）
                              例如："PharmacyEnv"
    
    返回：
        环境类对象：可以直接用于创建实例的类
    
    异常：
        ValueError: 如果指定的类名在代码中不存在
    
    使用示例：
        class_code = "class MyEnv:\n    pass"
        EnvClass = init_env_class(class_code, "MyEnv")
        instance = EnvClass()
    """
    # 创建一个新的动态模块对象
    # types.ModuleType 用于在运行时创建模块
    # "dynamic_env" 是模块的名称（仅用于标识，不影响功能）
    module = types.ModuleType("dynamic_env")
    
    # 在模块的命名空间中执行环境类代码
    # exec() 函数会执行 env_class_code 中的 Python 代码
    # 执行后的类定义会存储在 module.__dict__ 中
    # 这样我们就可以从模块中提取类对象
    exec(env_class_code, module.__dict__)
    
    # 检查模块中是否存在指定名称的类
    # hasattr() 检查对象是否有指定的属性（这里检查是否有类）
    if not hasattr(module, env_class_name):
        # 如果类不存在，抛出异常并提示错误信息
        raise ValueError(f"Class '{env_class_name}' not found in provided env_class_code.")
    
    # 从模块中提取指定名称的类对象
    # getattr() 获取对象的属性（这里获取类对象）
    # 返回的类对象可以直接用于创建实例
    return getattr(module, env_class_name)


def init_env_instance(env_class, init_config=None):
    """
    创建环境实例并应用初始配置
    
    功能说明：
    - 接收环境类对象和初始配置字典
    - 尝试用配置创建实例（支持多种构造方式）
    - 如果构造失败，尝试无参构造
    - 通过 setattr 设置所有配置属性
    
    参数：
        env_class: 环境类对象（由 init_env_class 返回）
        init_config (dict, optional): 初始配置字典
                                      例如：{"prescriptions": {...}, "medications": {...}}
    
    返回：
        环境实例对象：已应用初始配置的环境实例
    
    使用示例：
        EnvClass = init_env_class(code, "PharmacyEnv")
        config = {"prescriptions": {}, "medications": {}}
        instance = init_env_instance(EnvClass, config)
    """
    # 深拷贝初始配置，避免修改原始配置对象
    # 因为后续可能会修改 init_config，深拷贝可以保护原始数据
    init_config = deepcopy(init_config)
    
    # 尝试创建环境实例
    try:
        # 情况 1：如果提供了配置且是字典类型
        # 尝试用配置字典作为参数调用构造函数
        # 某些环境类可能支持：EnvClass(init_config)
        if init_config and isinstance(init_config, dict):
            # 将配置字典作为参数传递给构造函数
            env_instance = env_class(init_config)
        else:
            # 情况 2：如果没有配置或配置不是字典
            # 尝试用空字典作为参数调用构造函数
            env_instance = env_class({})
    except TypeError:
        # 情况 3：如果构造函数不接受参数（TypeError 表示参数类型不匹配）
        # 尝试无参构造函数
        # 某些环境类可能只支持：EnvClass()
        env_instance = env_class()
    
    # 如果提供了配置，通过 setattr 逐个设置属性
    # 这样可以确保所有配置项都被正确设置，即使构造函数不支持字典参数
    if init_config:
        # 遍历配置字典中的每个键值对
        for key, value in init_config.items():
            # 使用 setattr 动态设置实例属性
            # 等价于：env_instance.key = value
            # 但这里 key 是变量，所以必须用 setattr
            setattr(env_instance, key, value)
    
    # 返回已配置好的环境实例
    return env_instance



def get_state_diff(old_state: dict, new_state: dict, ignore_keys: list = []) -> dict:
    """
    比较两个状态字典并返回差异
    
    功能说明：
    - 递归比较两个状态字典
    - 识别新增的键、删除的键、修改的值
    - 支持忽略某些键（如临时变量）
    
    参数：
        old_state (dict): 旧状态字典（环境之前的状态）
        new_state (dict): 新状态字典（环境当前的状态）
        ignore_keys (list): 要忽略的键列表（这些键的变化不会被记录）
    
    返回：
        dict: 差异字典，格式如下：
        {
            "key1": {"added": value},           # 新增的键
            "key2": {"removed": value},        # 删除的键
            "key3": {"changed": {"old": ..., "new": ...}},  # 修改的值
            "key4": {                          # 嵌套字典的差异
                "subkey1": {"changed": {...}}
            }
        }
    
    使用示例：
        old = {"a": 1, "b": {"x": 10}}
        new = {"a": 2, "b": {"x": 20, "y": 30}}
        diff = get_state_diff(old, new)
        # 结果：{"a": {"changed": {"old": 1, "new": 2}}, 
        #        "b": {"x": {"changed": {"old": 10, "new": 20}}, 
        #              "y": {"added": 30}}}
    """
    # 深拷贝两个状态字典，避免修改原始数据
    # 因为在比较过程中可能会修改字典内容
    old_state = deepcopy(old_state)
    new_state = deepcopy(new_state)
    
    # 初始化差异结果字典
    diff_result = {}

    # 找到两个字典中所有键的并集
    # set() 创建集合，| 运算符求并集
    # 这样可以得到所有可能变化的键（包括新增和删除的）
    all_keys = set(old_state.keys()) | set(new_state.keys())

    # 遍历所有键，逐一比较
    for key in all_keys:
        # 获取旧状态中该键的值（如果不存在则为 None）
        old_val = old_state.get(key)
        # 获取新状态中该键的值（如果不存在则为 None）
        new_val = new_state.get(key)

        # 情况 1：键在旧状态中不存在，但在新状态中存在 → 新增的键
        if key not in old_state:
            # 记录为新增，保存新值
            diff_result[key] = {"added": new_val}
        # 情况 2：键在新状态中不存在，但在旧状态中存在 → 删除的键
        elif key not in new_state:
            # 记录为删除，保存旧值
            diff_result[key] = {"removed": old_val}
        # 情况 3：键在两个状态中都存在 → 需要比较值是否变化
        else:
            # 如果两个值都是字典类型，进行递归比较
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # 递归调用自身，比较嵌套字典
                sub_diff = get_state_diff(old_val, new_val)
                # 只有当嵌套字典有变化时才记录
                if sub_diff:  # 如果 sub_diff 不为空（有变化）
                    diff_result[key] = sub_diff
            else:
                # 如果值不是字典，直接比较是否相等
                # 简单类型比较（int, str, list 等）
                if old_val != new_val:
                    # 如果值发生了变化，记录旧值和新值
                    diff_result[key] = {"changed": {"old":old_val, "new":new_val}}
                    
    # 移除需要忽略的键
    # 这些键的变化不应该被记录（如临时变量、缓存等）
    for key in ignore_keys:
        # 如果差异结果中包含要忽略的键，删除它
        if key in diff_result:
            del diff_result[key]

    # 返回差异结果的深拷贝，避免外部修改影响内部数据
    return deepcopy(diff_result)


def get_state_info(env_instance):
    """
    获取环境实例的状态字典
    
    功能说明：
    - 提取环境实例的所有属性（排除内置属性）
    - 返回状态字典的深拷贝
    - **重要**：返回的字典可以直接用于 `init_env_instance()` 来初始化新实例
    
    参数：
        env_instance: 环境实例对象
    
    返回：
        dict: 状态字典，包含实例的所有非内置属性
              例如：{"prescriptions": {...}, "medications": {...}, "staff": {...}}
              这个字典可以直接作为 `init_config` 传递给 `init_env_instance()` 来创建新实例
    
    使用示例：
        # 获取当前状态
        state = get_state_info(env_instance)
        # 返回：{"prescriptions": {...}, "medications": {...}, "patients": {...}}
        
        # 使用状态字典初始化新实例（完全复制当前状态）
        new_instance = init_env_instance(EnvClass, state)
        # 新实例将拥有与原实例相同的所有属性值
    """
    # 返回状态字典的深拷贝
    return deepcopy({
        # 使用字典推导式构建状态字典
        # vars(env_instance) 获取实例的所有属性字典
        # k, v 是键值对
        k: v for k, v in vars(env_instance).items()
        # 过滤条件：排除内置属性（以 __ 开头和结尾的属性）
        # 例如：__class__, __dict__, __module__ 等
        if not (k.startswith("__") and k.endswith("__"))
    })


def get_final_config_for_init(env_instance):
    """
    获取环境实例的最终配置字典（可用于初始化新实例）
    
    功能说明：
    - 这是 `get_state_info()` 的别名函数，提供更明确的语义
    - 获取环境实例的所有属性（排除内置属性）
    - 返回的字典可以直接用于 `init_env_instance()` 来初始化新实例
    - 包含所有属性（包括初始配置中的字段和运行过程中新增的字段）
    
    参数：
        env_instance: 环境实例对象（Agent 操作后的实例）
    
    返回：
        dict: 最终配置字典，包含实例的所有非内置属性
              例如：{"prescriptions": {...}, "medications": {...}, "patients": {...}}
              这个字典可以直接作为 `init_config` 传递给 `init_env_instance()`
    
    使用示例：
        # Agent 操作后的环境实例
        # env_instance 已经执行了各种操作，可能添加了新属性
        
        # 获取最终配置（包含所有属性）
        final_config = get_final_config_for_init(env_instance)
        
        # 使用最终配置初始化新实例（完全复制当前状态）
        new_instance = init_env_instance(EnvClass, final_config)
        # 新实例将拥有与原实例相同的所有属性值
        
    注意：
        - 与 `extract_config_from_instance()` 的区别：
          - `extract_config_from_instance()`: 只返回 init_config 中定义的字段
          - `get_final_config_for_init()`: 返回所有属性（包括新增的字段）
    """
    # 直接调用 get_state_info，返回所有属性的字典
    # 这个字典可以直接用于 init_env_instance 来初始化新实例
    return get_state_info(env_instance)





def run_check_function(func_code: str, init_state: dict, final_state: dict):
    """
    动态执行任务完成检查函数
    
    功能说明：
    - 接收检查函数的源代码字符串
    - 在安全的环境中执行代码
    - 调用 check_func(final_state) 检查任务是否完成
    - 返回检查结果（True/False）
    
    参数：
        func_code (str): 检查函数的 Python 源代码字符串
                         例如："def check_func(final_state):\n    return True"
        init_state (dict): 环境初始状态（可在函数中通过 initial_state 访问）
        final_state (dict): 环境最终状态（作为参数传递给 check_func）
    
    返回：
        tuple: (success, result, error)
            - success (bool): 是否成功执行（True）或出现错误（False）
            - result (bool | None): 检查结果（True/False），如果执行失败则为 None
            - error (str | None): 错误信息，如果成功则为 None
    
    使用示例：
        code = "def check_func(final_state):\n    return final_state.get('status') == 'completed'"
        success, result, error = run_check_function(code, init_state, final_state)
        # success = True, result = True/False, error = None
    """
    # 创建安全的全局命名空间
    # 只包含 Python 内置函数和对象，避免执行恶意代码
    safe_globals = {
        '__builtins__': __builtins__,  # 包含 Python 内置函数（如 len, str, dict 等）
    }
    # 将初始状态添加到全局命名空间
    # 这样检查函数可以通过 initial_state 访问初始状态
    # 例如：if initial_state.get('count') < final_state.get('count'):
    safe_globals.update({"initial_state": deepcopy(init_state)})

    # 尝试执行检查函数代码
    try:
        # 在安全全局环境中执行函数代码
        # exec() 会执行 func_code 中的代码，定义 check_func 函数
        # 执行后的函数会保留在 safe_globals 中
        exec(func_code, safe_globals)

        # 检查是否成功定义了 check_func 函数
        if 'check_func' not in safe_globals:
            # 如果函数不存在，返回失败
            return False, None, "Function 'check_func' not found."

        # 调用 check_func 函数，传入最终状态
        # 函数应该返回 True（任务完成）或 False（任务未完成）
        result = safe_globals['check_func'](final_state)

        # 验证函数返回值是否为布尔类型
        if not isinstance(result, bool):
            # 如果返回值不是布尔类型，打印警告并返回错误
            print("Function did not return a boolean. Result: {result}")
            return False, None, "Function did not return a boolean."

        # 执行成功，返回结果
        return True, result, None
    except Exception as e:
        # 如果执行过程中出现任何异常，捕获并返回错误信息
        # 例如：语法错误、运行时错误等
        print("Error:", e)
        return False, None, str(e)





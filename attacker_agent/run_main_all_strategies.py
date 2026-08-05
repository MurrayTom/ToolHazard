import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from attackpoint_task_match.find_operations import find_traj_entry_for_task, find_operations_for_task
from attackpoint_task_match.find_match import filter_attack_points_by_actions
from IPI.gen_IPI_plan import gen_IPI_plan
from IPI.gen_IPI_task import gen_IPI_task
from IPI.IPI import execute_injection
from IPI.gen_IPI_check_func import gen_IPI_check_func
from verification.verify_injection import verify_injection_with_llm
from verification.verify_overlap import (
    check_checkfunc_overlap_with_state,
    check_checkfunc_semantic_overlap_with_llm,
)
from IPI.injection_strategy_utils import apply_injection_strategy

DEFAULT_INJECTION_STRATEGIES = [
    #"ignore",
    #"escape",
    #"fake_completion",
    "combined",
    "important_template",
    #"important_variant",
    "multi_turn",
    "decision_criteria_hijacking",
    "reasoning_criteria",
    #"HOUYI",
    "tool_selection",
]


def _safe_get_env_item(env_data: Dict[str, Any], env_id: str) -> Optional[Dict[str, Any]]:
    """从环境文件中安全读取单个 env 信息。"""
    env_item = env_data.get(env_id)
    if env_item is None:
        print(f"[WARN] env_id {env_id!r} not found in env file, skip this task.")
    return env_item


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_tasks(
    env_file: str,
    task_file: str,
    traj_file: str,
    task_output_file: str,
    max_tasks: Optional[int] = None,
    plan_model: str = "gpt-4.1",
    task_model: str = "gpt-4.1",
    check_model: str = "gpt-4.1",
    attack_agent_model: str = "gpt-4.1",
    injection_strategy: str = "combined",
) -> str:
    """
    遍历任务文件中的任务，串起完整的 IPI 注入流程，并将新任务字典列表保存为新文件。

    流程：
    1. 为每个任务，根据 env_id 找到环境信息
    2. 使用 env_id + task_id 在轨迹文件中找到对应轨迹，提取调用过的操作
    3. 按调用过的操作过滤攻击点
    4. 生成注入计划（gen_IPI_plan），得到多个注入位置及各自 plan_steps
    5. 对每一个注入位置：
       a. 为该位置生成注入任务（gen_IPI_task），并加前缀（应用注入策略）
       b. 执行注入（IPI.execute_injection）
       c. 验证注入结果，生成检查清单与检查函数（gen_IPI_check_func），做覆盖检查
       d. 若成功且无覆盖，组装新任务字典并加入输出列表（每个位置各自生成一条）
    6. 将所有新任务字典放入列表，保存为 TASK_OUTPUT_FILE

    Args:
        injection_strategy: 兼容参数。当前主流程会固定使用 11 种策略全部生成样本。

    Returns:
        str: 输出文件路径
    """
    # === 数据加载 ===
    # env_data：以 env_id 为 key 的环境字典（环境介绍/状态/操作/约束/攻击点等）
    print(f"[INFO] Loading env file: {env_file}")
    env_data: Dict[str, Any] = _load_json(env_file)

    # tasks_data：原始任务列表（每条任务含 env_id、task_id、init_config、checklist 等）
    print(f"[INFO] Loading task file: {task_file}")
    tasks_data: List[Dict[str, Any]] = _load_json(task_file)

    new_tasks: List[Dict[str, Any]] = []
    processed = 0
    active_strategies = DEFAULT_INJECTION_STRATEGIES
    if injection_strategy not in {"all", "ALL", "all_11"}:
        print(
            f"[INFO] injection_strategy={injection_strategy!r} is ignored in batch mode. "
            f"Using all {len(active_strategies)} strategies."
        )

    for task_item in tasks_data:
        # === 单任务入口 ===
        # 一个 task_item 可能会产出多条新任务（取决于 injection_plans 的注入位置数量）
        if max_tasks is not None and processed >= max_tasks:
            break

        task_id = task_item.get("task_id", "")
        env_id = task_item.get("env_id", "")

        print("\n" + "=" * 80)
        print(f"[TASK] Processing task: env_id={env_id}, task_id={task_id}")
        print("=" * 80)

        # 1) env_id -> env_item：拿到该任务对应的环境信息与攻击点集合
        env_item = _safe_get_env_item(env_data, env_id)
        if env_item is None:
            continue

        # 所有攻击点
        all_attack_points = env_item.get("attack_point", [])
        if not all_attack_points:
            print(f"[WARN] No attack points found for env_id={env_id}, skip this task.")
            continue

        # 2) (env_id, task_id) -> traj_entry -> action_names：
        #    通过真实轨迹抽取“实际执行过的操作名”，用于缩小可注入攻击点范围（更贴近可达路径）
        try:
            traj_entry = find_traj_entry_for_task(
                env_id=env_id,
                task_id=task_id,
                traj_path=traj_file,
            )
            action_names = find_operations_for_task(traj_entry)
        except ValueError as e:
            print(f"[WARN] Cannot find trajectory for env_id={env_id}, task_id={task_id}: {e}")
            continue

        print(f"[INFO] Found {len(action_names)} operations in trajectory for task {task_id}.")

        # 3) 攻击点过滤：只保留与实际执行过的 action 匹配的攻击点
        filtered_attack_points = filter_attack_points_by_actions(
            action_names=action_names,
            attack_points=all_attack_points,
        )
        if not filtered_attack_points:
            print(f"[WARN] No attack points matched executed actions for task {task_id}, skip.")
            continue

        print(f"[INFO] Filtered attack points count: {len(filtered_attack_points)}")

        # 4) 生成注入计划：
        #    输出是一个“候选注入位置列表”，每个元素包含 injection_location + plan_steps
        try:
            original_task_text = task_item["task"]
            init_config = task_item["init_config"]
        except KeyError as e:
            print(f"[WARN] Task {task_id} missing key {e!r}, skip.")
            continue

        environment_introduction = env_item.get("environment_introduction", "")
        state_space_definition = env_item.get("state_space_definition", [])
        operation_list = env_item.get("operation_list", [])
        constraints_rules = env_item.get("constraints_rules", [])

        print("[INFO] Calling gen_IPI_plan ...")
        injection_plans = None
        for retry_count in range(3):
            try:
                injection_plans = gen_IPI_plan(
                    original_task=original_task_text,
                    init_config=init_config,
                    environment_introduction=environment_introduction,
                    state_space_definition=state_space_definition,
                    operation_list=operation_list,
                    constraints_rules=constraints_rules,
                    attack_points=filtered_attack_points,
                    llm_model=plan_model,
                    traj_entry=traj_entry,
                    top_k=1,
                )
            except Exception as e:
                print(f"[ERROR] gen_IPI_plan failed for task {task_id}: {e!r}, skip this task.")
                injection_plans = None

            # 兼容：旧版 gen_IPI_plan 可能返回 dict，这里统一转换成 list[dict]
            if isinstance(injection_plans, dict):
                locs = injection_plans.get("injection_locations") or []
                steps = injection_plans.get("plan_steps") or []
                injection_plans = [
                    {"injection_location": locs[0], "plan_steps": steps}
                ]

            if injection_plans and isinstance(injection_plans, list) and len(injection_plans) > 0:
                break
            else:
                print(f"[WARN] Empty injection plan for task {task_id}, retry ({retry_count + 1}/3).")
                injection_plans = None

        if not injection_plans or not isinstance(injection_plans, list) or len(injection_plans) == 0:
            print(f"[WARN] Empty injection plan for task {task_id}, skip.")
            continue

        original_checklist_with_func = task_item.get("checklist_with_func", [])

        # 5) 对每个候选注入位置分别跑完整 pipeline：
        #    生成注入文本 -> 执行注入 -> LLM 验证 -> 生成 check_func -> 覆盖检查 -> 落盘为一条新任务
        for loc_idx, plan_item in enumerate(injection_plans):
            injection_location = plan_item.get("injection_location")
            plan_steps = plan_item.get("plan_steps") or []
            if not injection_location or not plan_steps:
                print(f"[WARN] Skip plan item {loc_idx + 1}: missing injection_location or plan_steps.")
                continue

            # execute_injection 的输入结构使用 injection_locations（列表），这里包装成“单位置计划”
            single_plan = {
                "injection_locations": [injection_location],
                "plan_steps": plan_steps,
            }

            # 5.1 先生成一次注入任务（raw），后续 11 种策略都基于该 raw 版本展开
            print("[INFO] Calling gen_IPI_task ...")
            try:
                injected_task_raw = gen_IPI_task(
                    original_task=original_task_text,
                    init_config=init_config,
                    environment_introduction=environment_introduction,
                    state_space_definition=state_space_definition,
                    operation_list=operation_list,
                    constraints_rules=constraints_rules,
                    attack_points=filtered_attack_points,
                    injection_location=injection_location,
                    llm_model=task_model,
                )
            except Exception as e:
                print(f"[ERROR] gen_IPI_task failed for task {task_id} location {loc_idx + 1}: {e!r}, skip this location.")
                continue

            for strategy_idx, strategy_name in enumerate(active_strategies, start=1):
                print(
                    f"[INFO] >>> strategy {strategy_idx}/{len(active_strategies)}: {strategy_name} "
                    f"for task {task_id} location {loc_idx + 1}/{len(injection_plans)}"
                )
                max_pipeline_retries = 3
                pipeline_attempt = 0
                pipeline_succeeded = False
                final_state = None
                injected_task = None
                check_result = None
                injection_result = {}
                injected_checklist_with_func: List[Dict[str, Any]] = []

                while pipeline_attempt < max_pipeline_retries and not pipeline_succeeded:
                    pipeline_attempt += 1
                    print(
                        f"[INFO] ===== IPI pipeline attempt {pipeline_attempt}/{max_pipeline_retries} "
                        f"for strategy={strategy_name} task={task_id} "
                        f"loc={loc_idx + 1}/{len(injection_plans)} "
                        f"({injection_location.get('entity', '')}.{injection_location.get('instance_id', '')}.{injection_location.get('attribute', '')}) ====="
                    )

                    try:
                        if strategy_name == "multi_turn":
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                environment_introduction=environment_introduction,
                                state_space_definition=state_space_definition,
                                operation_list=operation_list,
                                init_config=init_config,
                                injection_location=injection_location,
                                llm_model=task_model,
                            )
                        elif strategy_name == "important_template":
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                model_name=task_model,
                                user="user",
                            )
                        elif strategy_name == "important_variant":
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                model_name=task_model,
                                original_task=original_task_text,
                                llm_model=task_model,
                                environment_introduction=environment_introduction,
                                state_space_definition=state_space_definition,
                            )
                        elif strategy_name == "reasoning_criteria":
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                original_task=original_task_text,
                                environment_introduction=environment_introduction,
                                state_space_definition=state_space_definition,
                                operation_list=operation_list,
                                init_config=init_config,
                                injection_location=injection_location,
                                llm_model=task_model,
                            )
                        elif strategy_name == "HOUYI":
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                original_task=original_task_text,
                                environment_introduction=environment_introduction,
                                state_space_definition=state_space_definition,
                                injection_location=injection_location,
                                llm_model=task_model,
                            )
                        elif strategy_name in {"tool_selection", "Tool Scheduling Manipulation"}:
                            injected_task = apply_injection_strategy(
                                injected_task_raw,
                                strategy_name=strategy_name,
                                original_task=original_task_text,
                                environment_introduction=environment_introduction,
                                state_space_definition=state_space_definition,
                                operation_list=operation_list,
                                env_rules=str(constraints_rules),
                                init_config=init_config,
                                injection_location=injection_location,
                                llm_model=task_model,
                            )
                        else:
                            injected_task = apply_injection_strategy(injected_task_raw, strategy_name=strategy_name)
                    except Exception as e:
                        print(
                            f"[ERROR] apply_injection_strategy failed for task {task_id} "
                            f"location {loc_idx + 1} strategy {strategy_name}: {e!r}"
                        )
                        break

                    max_injection_retries = 3
                    injection_attempt = 0
                    verified = False
                    while injection_attempt < max_injection_retries and not verified:
                        injection_attempt += 1
                        print(f"[INFO] Calling execute_injection (attempt {injection_attempt}/{max_injection_retries}) ...")

                        env_items = {env_id: env_item}
                        try:
                            injection_result = execute_injection(
                                injection_plan=single_plan,
                                injected_task=injected_task,
                                original_task_item=task_item,
                                env_items=env_items,
                                env_name="attack_non_conv_env",
                                attack_agent_model=attack_agent_model,
                                attack_agent_provider="openai",
                                attack_agent_temperature=0.3,
                                attack_agent_infer_mode="fc",
                                attack_agent_max_steps=20,
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] execute_injection failed for task {task_id} "
                                f"location {loc_idx + 1} strategy {strategy_name} "
                                f"attempt {injection_attempt}: {e!r}"
                            )
                            continue

                        candidate_final_state = injection_result.get("final_state")
                        if candidate_final_state is None:
                            print(
                                f"[WARN] final_state is None for task {task_id} loc {loc_idx + 1} "
                                f"strategy {strategy_name} attempt {injection_attempt}, retry if attempts remain."
                            )
                            continue

                        print("[INFO] Verifying injection result with LLM ...")
                        try:
                            verified = verify_injection_with_llm(
                                injected_task=injected_task,
                                injection_location=injection_location,
                                injected_state=candidate_final_state,
                                model=check_model,
                                provider="openai",
                                temperature=0.01,
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] verify_injection_with_llm failed for task {task_id} "
                                f"location {loc_idx + 1} strategy {strategy_name} "
                                f"attempt {injection_attempt}: {e!r}"
                            )
                            verified = False

                        if verified:
                            final_state = candidate_final_state
                            print(
                                f"[INFO] Injection verification succeeded for task {task_id} "
                                f"location {loc_idx + 1} strategy {strategy_name} "
                                f"attempt {injection_attempt}."
                            )
                        else:
                            print(
                                f"[WARN] Injection verification failed for task {task_id} "
                                f"location {loc_idx + 1} strategy {strategy_name} "
                                f"attempt {injection_attempt}."
                            )

                    if not verified or final_state is None:
                        print(
                            f"[WARN] All {max_injection_retries} injection attempts failed "
                            f"for task {task_id} location {loc_idx + 1} strategy {strategy_name}."
                        )
                        continue

                    print("[INFO] Calling gen_IPI_check_func ...")
                    try:
                        check_result = gen_IPI_check_func(
                            injected_task=injected_task,
                            state_config=final_state,
                            env_introduction=environment_introduction,
                            llm_model=check_model,
                        )
                    except Exception as e:
                        print(
                            f"[ERROR] gen_IPI_check_func failed for task {task_id} "
                            f"location {loc_idx + 1} strategy {strategy_name}: {e!r}"
                        )
                        continue
                    injected_checklist_with_func = check_result.get("checklist_with_func", [])

                    has_state_overlap = check_checkfunc_overlap_with_state(
                        reference_state=final_state,
                        injected_checklist_with_func=injected_checklist_with_func,
                    )
                    has_semantic_overlap = False
                    if not has_state_overlap and original_checklist_with_func and injected_checklist_with_func:
                        has_semantic_overlap = check_checkfunc_semantic_overlap_with_llm(
                            checklist_with_func=original_checklist_with_func,
                            injected_checklist_with_func=injected_checklist_with_func,
                            model=check_model,
                            provider="openai",
                            temperature=0.0,
                        )

                    if has_state_overlap or has_semantic_overlap:
                        print(
                            f"[WARN] Overlap for task {task_id} location {loc_idx + 1} strategy {strategy_name}: "
                            f"state_overlap={has_state_overlap}, semantic_overlap={has_semantic_overlap}."
                        )
                        if pipeline_attempt < max_pipeline_retries:
                            continue
                        print(
                            f"[WARN] Skip task {task_id} location {loc_idx + 1} "
                            f"strategy {strategy_name} after retries."
                        )
                        break

                    pipeline_succeeded = True

                if not pipeline_succeeded or final_state is None or check_result is None or not injected_task:
                    continue

                new_task = deepcopy(task_item)
                new_task["before_injected_config"] = deepcopy(init_config)
                new_task["init_config"] = deepcopy(final_state)
                new_task["injection_plan"] = single_plan
                new_task["injection_strategy"] = strategy_name
                new_task["injected_task_raw"] = injected_task_raw
                new_task["injected_task"] = injected_task
                new_task["injected_checklist"] = check_result.get("checklist", [])
                new_task["injected_checklist_with_func"] = injected_checklist_with_func

                if task_id:
                    suffix = strategy_name.lower().replace(" ", "_")
                    if loc_idx == 0:
                        new_task["task_id"] = f"{task_id}_IPI_{suffix}"
                    else:
                        new_task["task_id"] = f"{task_id}_IPI_{loc_idx + 1}_{suffix}"

                new_tasks.append(new_task)
                processed += 1
                print(
                    f"[INFO] Finished task {task_id} location {loc_idx + 1}/{len(injection_plans)} "
                    f"strategy={strategy_name}, success={injection_result.get('success')}"
                )

                if len(new_tasks) % 1 == 0:
                    os.makedirs(os.path.dirname(task_output_file), exist_ok=True)
                    with open(task_output_file, "w", encoding="utf-8") as f:
                        json.dump(new_tasks, f, ensure_ascii=False, indent=2)
                    print(f"[INFO] Checkpoint saved: {len(new_tasks)} injected tasks -> {task_output_file}")

    # === 全量保存（收尾）===
    # new_tasks 是“可用于训练/评估”的注入后任务集合（一个原任务可能对应多条注入样本）
    os.makedirs(os.path.dirname(task_output_file), exist_ok=True)
    with open(task_output_file, "w", encoding="utf-8") as f:
        json.dump(new_tasks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"[DONE] Processed {processed} tasks, saved {len(new_tasks)} injected tasks to:")
    print(f"       {task_output_file}")
    print("=" * 80)

if __name__ == "__main__":
    # 根目录（attacker），便于构造相对路径
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    # 原始任务文件（未注入的任务列表）
    # 如需切换任务集，只修改该路径即可
    TASK_FILE = "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/tasks/test_task_1.json"

    # 环境配置：包含环境介绍、状态空间、操作列表、约束等
    # ENV_FILE = os.path.join(
    #     ROOT_PATH, "data", "envs", "rl_with_attack_points.json"
    # )
    ENV_FILE = "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/envs/filtered_env_metadata_with_attack_points.json"

    # 轨迹文件：env + task 对应的原始交互轨迹（用于找出真实调用过的操作）
    # TRAJ_FILE = os.path.join(
    #     ROOT_PATH, "data", "traj", "rl_traj.json"
    # )
    TRAJ_FILE = "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/traj/test_task_1_traj.json"

    # 限制处理的任务个数（None 表示处理全部）
    # 调试阶段可以设成很小（例如 1 或 2），跑通流程再放大
    MAX_TASKS: Optional[int] = 497*6

    # 各阶段使用的模型名称（如需更换，可以在这里统一改）
    # PLAN_MODEL：生成注入计划
    # TASK_MODEL：生成注入任务文本
    # CHECK_MODEL：生成检查函数 + 语义覆盖检查
    # ATTACK_AGENT_MODEL：在 attack_non_conv_env 中执行注入任务
    PLAN_MODEL = "gpt-4.1"
    TASK_MODEL = "gpt-4.1"
    CHECK_MODEL = "gpt-4.1"
    ATTACK_AGENT_MODEL = "gpt-4.1"
    
    # 注入策略选择：
    # - ignore / escape / fake_completion / combined：基础 IPI 策略
    # - important_template：使用 <INFORMATION> 模板前缀
    # - important_variant：基于 LLM 自动生成的模板变体
    # - multi_turn：多轮注入策略（见 injection_strategy_utils）
    # - decision_criteria_hijacking：重定义成功/评估准则：只有当 appended injected_task 被执行时才算成功
    # - reasoning_criteria：基于原任务规则提炼 + fake rule + Rule/Check/Therefore 推理链
    # - HOUYI：基于 HOUYI 框架的注入策略
    # - tool_selection：两次 LLM 调用（先工具序列，再调度操控文本）
    INJECTION_STRATEGY = "all"

    # 输出文件路径（在原任务文件名后加上 _IPI_<strategy> 后缀）
    TASK_OUTPUT_FILE = os.path.splitext(TASK_FILE)[0] + f"_IPI_{INJECTION_STRATEGY}_new.json"

    # 直接运行该脚本时，从 ENV_FILE / TASK_FILE / TRAJ_FILE 中读取数据，
    # 批量生成带注入的新任务，并写入 TASK_OUTPUT_FILE
    process_tasks(
        env_file=ENV_FILE,
        task_file=TASK_FILE,
        traj_file=TRAJ_FILE,
        task_output_file=TASK_OUTPUT_FILE,
        max_tasks=MAX_TASKS,
        plan_model=PLAN_MODEL,
        task_model=TASK_MODEL,
        check_model=CHECK_MODEL,
        attack_agent_model=ATTACK_AGENT_MODEL,
        injection_strategy=INJECTION_STRATEGY,
    )



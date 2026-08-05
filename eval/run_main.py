"""
The main program is used for:
1. Interacting with the EnvScaler synthesized environment to acquire trajectories for training
2. Evaluating the performance of TauBench and AceBench.
"""
import os
import json
import time
from tqdm import tqdm
from copy import deepcopy
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.task_solve_agent import TaskSolveAgent

# Environment imports
from envscaler_env import EnvScalerConvRLEnv, EnvScalerNonConvRLEnv, EnvScalerConvSFTEnv, EnvScalerNonConvSFTEnv
from taubench_env import TauBenchRetailEnv, TauBenchAirlineEnv
from bfcl_env import BfclEnv
from acebench_env import AceBenchMultiStepEnv, AceBenchMultiTurnEnv


# Environment class mapping
env_cls_map = {
    "envscaler_conversation_rl": EnvScalerConvRLEnv,
    "envscaler_non_conversation_rl": EnvScalerNonConvRLEnv,
    "envscaler_conversation_sft": EnvScalerConvSFTEnv,
    "envscaler_non_conversation_sft": EnvScalerNonConvSFTEnv,
    "tau_bench_retail": TauBenchRetailEnv,
    "tau_bench_airline": TauBenchAirlineEnv,
    "bfcl": BfclEnv,
    "acebench_multi_step": AceBenchMultiStepEnv,
    "acebench_multi_turn": AceBenchMultiTurnEnv
}

# Maximum agent-environment interaction steps mapping for each environment
max_steps_map = {
    "envscaler_conversation_rl": 40,
    "envscaler_non_conversation_rl": 30,
    "envscaler_conversation_sft": 40,
    "envscaler_non_conversation_sft": 30,
    "tau_bench_retail": 30,
    "tau_bench_airline": 30,
    "bfcl": 40,
    "acebench_multi_step": 20,
    "acebench_multi_turn": 20
}


def get_current_time():
    """Get current time as formatted string."""
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def get_model_config(model_name: str):
    """
    Get API configuration for the given model.
    Supports model-specific env vars like:
    - QWEN3_API_KEY, QWEN3_BASE_URL
    - GPT4_API_KEY, GPT4_BASE_URL
    Fallback to default OPENAI_API_KEY, OPENAI_BASE_URL
    """
    # Extract prefix from model name (e.g., "Qwen3-8B" -> "QWEN3", "gpt-4o" -> "GPT")
    # Handle various naming patterns
    model_upper = model_name.upper()
    
    # Try common prefixes
    prefixes = []
    if "QWEN" in model_upper:
        prefixes.append("QWEN3")
    if "GPT4" in model_upper:
        prefixes.append("GPT4")
    if "CLAUDE" in model_upper:
        prefixes.append("CLAUDE")
    if "GEMINI" in model_upper:
        prefixes.append("GEMINI")

    # if "SFT" in model_upper:
    #     prefixes.append("SFT")
    # if "RL" in model_upper:
    #     prefixes.append("RL")
    
    # Try exact match first (e.g., MODEL_NAME_API_KEY=QWEN3-8B_API_KEY)
    exact_prefix = model_upper.replace("-", "_").replace(".", "_")
    prefixes.insert(0, exact_prefix)
    
    api_key = None
    base_url = None
    for prefix in prefixes:
        key = f"{prefix}_API_KEY"
        url = f"{prefix}_BASE_URL"
        if os.getenv(key):
            api_key = os.getenv(key)
            print(f"Using {key} for model {model_name}")
        if os.getenv(url):
            base_url = os.getenv(url)
            print(f"Using {url} for model {model_name}")
        if api_key or base_url:
            break
    
    # Fallback to default
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")
    
    return api_key, base_url


def save_json(path, data):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def solve_task(env_name, env_config, agent_model, agent_model_provider, infer_mode, enable_thinking, task_id):
    # Initialize environment
    try:
        env = env_cls_map[env_name](**env_config)
    except Exception as e:
        raise Exception(f"Error in env_cls_map[{env_name}](**env_config): {repr(e)}")

    max_steps = max_steps_map[env_name]

    # Get model-specific API config
    api_key, base_url = get_model_config(agent_model)
    print(f"Model: {agent_model}, API Key: {'Yes' if api_key else 'No'}, Base URL: {base_url}")

    # Initialize task solving agent
    agent = TaskSolveAgent(
        env_name=env_name,
        env=env,
        model = agent_model,
        provider = agent_model_provider,
        infer_mode=infer_mode,
        temperature=0.9,
        max_steps=max_steps,
        enable_thinking=enable_thinking,
        api_key=api_key,
        base_url=base_url
    )
    
    # Execute task
    save_data = {}
    result = agent.run(task_index=task_id)
    save_data.update(result)
    return save_data


def solve_task_multiprocess(task_configs, save_file_path, num_workers):
    """
    multi-process (actually thread pool) execution of solve_task
    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                solve_task,
                cfg["env_name"],
                cfg["env_config"],
                cfg["agent_model"],
                cfg["agent_model_provider"],
                cfg["infer_mode"],
                cfg["enable_thinking"],
                cfg["task_id"]
            )
            for cfg in task_configs
        ]

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                # get task result
                res = future.result()
            except Exception as e:
                # single task exception, print and skip
                print(f"[WARNING] Task {i} error, skipped: {e}")
                import traceback
                print(traceback.format_exc())
                continue

            results.append(res)

            if len(results) % 5 == 0:
                save_json(save_file_path, results)

    # save final results
    save_json(save_file_path, results)


def solve_task_single_process(task_configs, save_file_path):
    """
    single thread execution of solve_task
    """
    # if directory does not exist, create it
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)

    results = []
    for i, cfg in enumerate(tqdm(task_configs, total=len(task_configs))):
        res = solve_task(
            cfg["env_name"],
            cfg["env_config"],
            cfg["agent_model"],
            cfg["agent_model_provider"],
            cfg["infer_mode"],
            cfg["enable_thinking"],
            cfg["task_id"]
        )
        results.append(res)
        # save every 10 results
        if len(results) % 10 == 0:
            save_json(save_file_path, results)

    # save final results
    save_json(save_file_path, results)


if __name__ == "__main__":
    # load env
    load_dotenv()

    # setting   
    # set your llm as agent_model
    agent_model = "qwen3-8b-sft_rl"
    agent_model_provider = "openai"

    # Enable Thinking Mode (Only applicable to hybrid thinking models that support thinking switching, such as Qwen3-8B; does not work for other models)
    # Note: OpenAI models (gpt-4.1-mini, etc.) do NOT support chat_template_kwargs, so set to False when using OpenAI
    enable_thinking = False
    # enable_thinking = True  # Only enable this for models that support thinking mode (e.g., Qwen3-8B)
    num_workers = 10

    #######################################

    # # EnvScaler-NonConversation-SFT-Env (Training Env)
    # # No Reward for SFT
    # env_name = "envscaler_non_conversation_rl"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval",
    #     "env_items_path": "../data/train_set/100_sft_train_env_with_attack_points.json",
    #     "task_items_path": "../data/train_set/sft_tasks_IPI_all_new.json",
    #     # "env_items_path": "../attacker/data/envs/example_env2_with_attack_points.json",
    #     # "task_items_path": "../attacker/data/tasks/env2/complex/env_2_sft_task_IPI_multi_turn.json",
    # }
    # # # 测试运行：只运行第一个任务
    # # task_ids = [0]  # 只运行任务 0 进行测试
    # task_ids = [i for i in range(711)]  # 运行所有任务
    # task_ids = task_ids[565:]
    #########################################

    # env_name = "envscaler_non_conversation_rl"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval",
    #     "env_items_path": "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/envs/filtered_env_metadata_with_attack_points.json",
    #     "task_items_path": "/data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/tasks/test_task_1_IPI_all_new.json",
    #     # "env_items_path": "../attacker/data/envs/example_env2_with_attack_points.json",
    #     # "task_items_path": "../attacker/data/tasks/env2/complex/env_2_sft_task_IPI_multi_turn.json",
    # }


    # # EnvScaler-NonConversation-SFT-Env (Training Env)
    # # No Reward for SFT
    env_name = "envscaler_non_conversation_rl"
    infer_mode = "prompt"
    env_config = {
        "mode": "eval",
        "env_items_path": "../toolhazard_bench/test_set/env_with_attack_points.json",
        "task_items_path": "../toolhazard_bench/test_set/merged_6_categories.json",
        # "env_items_path": "../attacker/data/envs/example_env2_with_attack_points.json",
        # "task_items_path": "../attacker/data/tasks/env2/complex/env_2_sft_task_IPI_multi_turn.json",
    }
    # # 测试运行：只运行第一个任务
    # task_ids = [0]  # 只运行任务 0 进行测试
    task_ids = [i for i in range(510)]  # 运行所有任务

    # EnvScaler-Conversation-SFT-Env (Training Env)
    # No Reward for SFT
    # env_name = "envscaler_conversation_sft"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/sft_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    # }
    # task_ids = [i for i in range(4684)]


    # # EnvScaler-NonConversation-RL-Env (Training Env)
    # env_name = "envscaler_non_conversation_rl"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
    # }
    # task_ids = [i for i in range(2250)]


    # # EnvScaler-Conversation-RL-Env (Training Env)
    # env_name = "envscaler_conversation_rl"
    # infer_mode = "prompt"
    # env_config = {
    #     "mode": "train",
    #     "env_items_path": "envscaler_env/data/191_env_metadata.json",
    #     "task_items_path": "envscaler_env/data/rl_scenario_metadata.json",
    #     "user_model": "gpt-4.1", 
    #     "provider": "openai",
    # }
    # task_ids = [i for i in range(2250)]


    # TauBench retail (Evaluation Env)
    # env_name = "tau_bench_retail"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_ids = [i for i in range(115)]


    # # TauBench airline (Evaluation Env)
    # env_name = "tau_bench_airline"
    # infer_mode = "fc"
    # env_config = {
    #     "mode": "eval", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_strategy": "llm_react",
    #     "user_provider": "openai",
    # }
    # task_ids = [i for i in range(50)]


    # # AceBench multi-step (Evaluation Env)
    # env_name = "acebench_multi_step"
    # infer_mode = "fc"
    # env_config = {"domain": "agent_multi_step", "truncated_steps": 20}
    # task_ids = [f"agent_multi_step_{i}" for i in range(20)]


    # # AceBench multi-turn (Evaluation Env)
    # env_name = "acebench_multi_turn"
    # infer_mode = "fc"
    # env_config = {
    #     "domain": "agent_multi_turn", 
    #     "user_model": "gpt-4.1-2025-04-14", 
    #     "user_provider": "openai", 
    #     "truncated_steps": 20}
    # task_ids = [f"agent_multi_turn_{i}" for i in range(30)]


    # # BFCL multi-turn base (Evaluation Env)
    # env_name = "bfcl"
    # infer_mode = "prompt"
    # env_config = {"mode": "multi_turn_base"}
    # # BFCL 使用整数索引作为 task_id
    # import json
    # with open("bfcl_env/data/data_multi_turn_base.json", "r") as f:
    #     bfcl_data = json.load(f)
    # task_ids = list(range(len(bfcl_data)))  # 使用整数索引 0, 1, 2, ...


    # print settings
    print(f"agent_model: {agent_model}")
    print(f"agent_model_provider: {agent_model_provider}")
    print(f"infer_mode: {infer_mode}")
    print(f"enable_thinking: {enable_thinking}")

    # generate task configs
    task_configs = [
        deepcopy({
            "env_name": env_name,
            "env_config": env_config,
            "agent_model": agent_model,
            "agent_model_provider": agent_model_provider,
            "infer_mode": infer_mode,
            "enable_thinking": enable_thinking,
            "task_id": task_id
        })
        for task_id in task_ids
    ]
    # generate save file path
    if env_name in ["bfcl", "envscaler_non_conversation_rl","envscaler_non_conversation_sft", "acebench_multi_step"]:
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{get_current_time()}.json"
    elif env_name in ["envscaler_conversation_rl","envscaler_conversation_sft", "acebench_multi_turn"]:
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{env_config['user_model']}_{get_current_time()}.json"
    else: # tau bench
        save_file_path = f"result/{env_name}/{agent_model}-{infer_mode}_{env_config['user_model']}_{env_config['user_strategy']}_{get_current_time()}.json"
    print("save_file_path:", save_file_path)
    # run task solving
    solve_task_multiprocess(task_configs=task_configs, save_file_path=save_file_path, num_workers=num_workers)
    # solve_task_single_process(task_configs=task_configs, save_file_path=save_file_path)
    print("save_file_path:", save_file_path)
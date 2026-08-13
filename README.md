<p align="center">
  <img src="assets/ToolHazard-logo.png" width="180" alt="ToolHazard project avatar">
</p>
<h1 align="center"> ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents</a></h1>

<div align="center">
  <a href="http://arxiv.org/abs/2601.10156">
    <img src="https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv" alt="Arxiv">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Model-Hugging%20Face-blue?logo=huggingface" alt="Hugging Face Models">
  </a>
  <a href="">
    <img src="https://img.shields.io/badge/Dataset-Hugging%20Face-blue?logo=huggingface" alt="Hugging Face Datasets">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg" alt="License">
  </a>
  <a href="https://www.python.org/downloads/release/python-312/">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python 3.10+">
  </a>
</div>


ToolHazard is a scalable framework for evaluating and improving the security of tool-using LLM agents. It synthesizes executable stateful environments, generates long-horizon user tasks, and plans environment-side indirect prompt injection attacks. The resulting data support both security evaluation and adversarial alignment.

## Project Structure

```text
ToolHazard/
├── env_simulator/       # Build and validate executable tool environments
├── user_simulator/      # Generate states, long-horizon tasks, and check functions
├── attacker_agent/      # Discover attack points and execute prompt injections
├── toolhazard_bench/    # Benchmark and alignment datasets
├── eval/                # Agent rollout and security/utility evaluation
├── sft/                 # SFT data processing and LlamaFactory configuration
└── rl/                  # ROLL environments, manager, and GRPO configuration
```

The main data flow is:

```text
Environment Simulator -> User Simulator -> Attacker Agent -> ToolHazard-Bench
                                                        -> Evaluation / SFT / RL
```

## Quick Start

```bash
git clone https://github.com/MurrayTom/ToolHazard.git
cd ToolHazard
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the LLM service in `.env`:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=your_openai_compatible_endpoint
```

> Some scripts retain absolute data/model paths from the original experiments. Update the configuration block at the bottom of each script before running it. SFT and RL dependencies are not included in `requirements.txt`.

## Modules

### Environment Simulator

`env_simulator/` turns seed tool-use tasks into executable, stateful environments through environment blueprint planning, program construction, and automated quality inspection.

Run the single-task demo:

```bash
cd env_simulator
# Configure ROOT_DIR, the seed task, and model in env_build_demo.py
python env_build_demo.py
```

For batch synthesis, run the scripts under the following directories in order:

```text
stage1_collect_env_from_task/ -> stage2_syn_env/ -> stage3_check_env/
```

Within each stage, run the `step*.py` scripts in numerical order (commands below
assume the current directory is `env_simulator/`):

```bash
# Stage 1: collect environment descriptions from seed tasks
python stage1_collect_env_from_task/step0_collect_task.py
python stage1_collect_env_from_task/step1_judge_stateful_query.py
python stage1_collect_env_from_task/step2_infer_env_topic.py
# Optional de-duplication/filtering steps:
python stage1_collect_env_from_task/step3_optional_get_embedding.py
python stage1_collect_env_from_task/step3_optional_select_env.py

# Stage 2: synthesize executable environment code
python stage2_syn_env/step1_infer_state.py
python stage2_syn_env/step2_infer_state_code.py
python stage2_syn_env/step3_infer_operation.py
python stage2_syn_env/step4_infer_func_code.py
python stage2_syn_env/step5_concat.py
python stage2_syn_env/step6_analysis_env_class_code.py

# Stage 3: validate and filter the synthesized environments
python stage3_check_env/step1_gen_test_config.py
python stage3_check_env/step2_roll_check.py
python stage3_check_env/step3_filter_env_by_check_result.py
```

The final environment metadata are written to `stage3_check_env/final_result/filtered_env_metadata.json`.

### User Simulator

`user_simulator/` initializes valid environment states, generates long-horizon user tasks, and creates state-based check functions for automatic verification.

```bash
cd user_simulator
# Configure the input/output paths and model at the bottom of each script
python step1_gen_env_config.py
python step2_gen_scenario_task.py
python step3_gen_task_check_func.py
```

The pipeline produces scenario records containing `init_config`, `task`, `checklist`, and `checklist_with_func`.

### Attacker Agent

`attacker_agent/` discovers injectable text attributes and their read/write paths, aligns attack points with benign trajectories, plans and executes indirect prompt injections, and verifies successful environment poisoning.

Run the attacker pipeline in the following order.

1. Discover attack points. Configure `input_json_path`, `output_json_path`, and
   the LLM settings at the bottom of `attack_point/find_attack_point.py`, then run:

```bash
python attacker_agent/attack_point/find_attack_point.py
```

The input is environment metadata (for example, the Environment Simulator's
filtered output). The output contains the same metadata augmented with an
`attack_point` list for each retained environment.

2. Generate benign trajectories used for attack-point/task alignment:

```bash
python attacker_agent/generate_trajectories.py \
  --env-file attacker_agent/data/envs/rl_with_attack_points.json \
  --task-file attacker_agent/data/tasks/rl/rl_task.json \
  --output-file attacker_agent/data/traj/rl_traj.json \
  --model gpt-4.1 --infer-mode fc --num-workers 4
```

Key arguments are:

- `--env-file`: environment metadata JSON. For the complete pipeline, use the
  attack-point-enriched output from Step 1.
- `--task-file`: benign task/scenario JSON list. Each task's `env_id` must exist
  in `--env-file`.
- `--output-file`: destination for the generated benign trajectories; this file
  becomes `TRAJ_FILE` in Step 3.
- `--model` and `--infer-mode`: task-solving model and tool-call mode (`fc` or
  `prompt`). `--num-workers` controls parallel trajectory generation.

Use `python attacker_agent/generate_trajectories.py --help` for the complete
option list, including task ranges, checkpointing, resume, and validation.

3. Generate injected samples with all enabled attack strategies. Configure the
path and model constants at the bottom of `attacker_agent/run_main_all_strategies.py`,
then run:

```bash
python attacker_agent/run_main_all_strategies.py
```

The three path constants connect the pipeline artifacts:

- `ENV_FILE`: attack-point-enriched environment metadata produced in Step 1.
- `TASK_FILE`: the benign task file used in Step 2.
- `TRAJ_FILE`: the trajectory file passed as `--output-file` in Step 2. Entries
  are matched to tasks by `env_id` and `task_id`.

You can also configure `TASK_OUTPUT_FILE`, `MAX_TASKS`, the model constants, and
`DEFAULT_INJECTION_STRATEGIES` before running the script.

Generated samples include the poisoned initial state, injection plan, injected task, and verification functions.

### ToolHazard-Bench

`toolhazard_bench/` contains the released benchmark and alignment data:

- `test_set/`: benign tasks and six indirect prompt injection variants for evaluation.
- `train_set/`: SFT/RL tasks, trajectories, check functions, environments, and adversarial samples.
- `191_env_metadata*.json`: executable environment metadata and tool definitions.

These JSON files can be used directly by `eval/`, `sft/`, and `rl/` after updating their configured paths.

### Evaluation

`eval/` supports ToolHazard environments as well as TauBench, ACEBench, and
BFCL-compatible environments. For ToolHazard evaluation, configure the following
block at the bottom of `eval/run_main.py` (or the corresponding block in
`eval/run_main_debug.py`):

```python
env_name = "envscaler_non_conversation_rl"
infer_mode = "prompt"
env_config = {
    "mode": "eval",
    "env_items_path": "../toolhazard_bench/test_set/rl_with_attack_points.json",
    "task_items_path": "../toolhazard_bench/test_set/rl_task_top_5_IPI_toolselection.json",
}
```

> **Reproducing the paper results:** all evaluation results reported in our paper
> use `infer_mode = "prompt"`. Function-calling mode (`"fc"`) is also supported
> for additional experiments, but it is not the mode used for the reported results.

`env_items_path` points to the executable environment metadata with discovered
attack points. `task_items_path` selects the evaluation task set and attack
strategy; replace the `toolselection` file with another JSON file under
`toolhazard_bench/test_set/` to evaluate a different released strategy. Paths in
this configuration are relative to the `eval/` directory. Also set `task_ids` to
the indices to evaluate and configure the agent model and worker count in the
same section of the script.

```bash
cd eval
python run_main_debug.py                 # Single-task debugging
python run_main.py                       # Batch rollout
python evaluate.py --input result.json   # Aggregate BR/ASR-related results
```

Rollout trajectories and final environment states are saved under `eval/result/` or `eval/result_debug/`.

### Supervised Fine-Tuning

`sft/` converts successful trajectories into LlamaFactory format and provides a Qwen3 training configuration.

```bash
# First update the tokenizer and data paths in both scripts
python sft/step1_process_messages_by_tool_template.py
python sft/step2_process_llamafactory_format.py

# Run after installing LlamaFactory and updating qwen3_full_sft.yaml
llamafactory-cli train sft/qwen3_full_sft.yaml
```

### Reinforcement Learning

`rl/` contains ToolHazard environment implementations and an environment manager for the [ROLL](https://github.com/alibaba/ROLL) framework. Copy the matching files under `rl/roll/` into a ROLL checkout, register the ToolHazard environments, update `rl/example/env_scaler/only_non_conv_qwen3_8gpu.yaml`, and launch the ROLL agentic pipeline with that configuration.

The trajectory-level reward encourages completion of the intended user task while penalizing completion of the injected task.

## Citation

```bibtex
@article{mou2026toolhazard,
  title={ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents},
  author={Mou, Yutao and Yang, Pengfei and Yin, Zhe and Xue, Zhangchi and Luan, Xiaotian and Yu, Dingyao and Zhang, Shikun and Ye, Wei},
  year={2026}
}
```

## License

This project is released under the [MIT License](LICENSE).

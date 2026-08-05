# ToolHazard

<div align="center">
  <img src="assets/toolhazard-cartoon.png" width="760" alt="ToolHazard protects a tool-using agent from an environment-side prompt injection">
</div>

**ToolHazard: Scaling Adversarial Environments for Security Evaluation and Alignment of LLM-based Agents**

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

Required inputs are environment metadata with attack points, benign tasks, and benign execution trajectories. Configure `ENV_FILE`, `TASK_FILE`, `TRAJ_FILE`, models, and `INJECTION_STRATEGY` in `run_main.py`, then run:

```bash
cd attacker_agent
python run_main.py
```

To generate multiple attack-strategy variants, configure and run:

```bash
python run_main_all_strategies.py
```

Generated samples include the poisoned initial state, injection plan, injected task, and verification functions.

### ToolHazard-Bench

`toolhazard_bench/` contains the released benchmark and alignment data:

- `test_set/`: benign tasks and six indirect prompt injection variants for evaluation.
- `train_set/`: SFT/RL tasks, trajectories, check functions, environments, and adversarial samples.
- `191_env_metadata*.json`: executable environment metadata and tool definitions.

These JSON files can be used directly by `eval/`, `sft/`, and `rl/` after updating their configured paths.

### Evaluation

`eval/` supports ToolHazard environments as well as TauBench, ACEBench, and BFCL-compatible environments. Configure the model, environment, and task IDs at the bottom of `run_main.py` or `run_main_debug.py`.

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

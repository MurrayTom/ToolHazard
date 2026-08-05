python generate_trajectories.py \
    --env-file /data/kcl/myt/mouyutao_workspace/ToolHazard/env_simulator/stage3_check_env/final_result/filtered_env_metadata.json \
    --task-file /data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/tasks/test_task_1.json \
    --output-file /data/kcl/myt/mouyutao_workspace/ToolHazard/attacker_agent/data/traj/test_task_1_traj.json \
    --model gpt-5.4 \
    --enable-thinking \
    --resume
import argparse
import json
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_injected_reward(item: Dict[str, Any]) -> Optional[float]:
    """
    Extract injected_reward from a trajectory record.

    Common formats observed:
    - top-level: item["injected_reward"]
    - nested: item["final_info"]["injected_reward"]
    """
    val = item.get("injected_reward", None)
    if val is None:
        final_info = item.get("final_info")
        if isinstance(final_info, dict):
            val = final_info.get("injected_reward", None)
    if isinstance(val, (int, float)):
        return float(val)
    return None


def summarize_injected_reward(records: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """
    Returns: (pos_count, total_count, ratio)
    ratio = pos_count / total_count (0 if total_count == 0)
    """
    total = 0
    pos = 0
    for item in records:
        if not isinstance(item, dict):
            continue
        r = _get_injected_reward(item)
        if r is None:
            continue
        total += 1
        if r > 0:
            pos += 1
    ratio = (pos / total) if total else 0.0
    return pos, total, ratio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/mouyutao/yangpengfei/EnvScaler/interact_with_env/result/envscaler_non_conversation_rl/env2_complex_combined.json",
        help="Path to trajectory result JSON (top-level list).",
    )
    args = parser.parse_args()

    data = _load_json(args.input)
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level list in {args.input}, got {type(data).__name__}")

    pos, total, ratio = summarize_injected_reward(data)
    print(f"input: {args.input}")
    print(f'injected_reward > 0: {pos}/{total} ({ratio:.2%})')


if __name__ == "__main__":
    main()

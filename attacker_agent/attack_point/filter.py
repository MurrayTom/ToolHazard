import json
from pathlib import Path
from typing import Any, Dict


SOURCE_FILE = Path("/home/mouyutao/yangpengfei/EnvScaler/interact_with_env/envscaler_env/data/191_env_metadata.json")
TARGET_FILE = Path("/home/mouyutao/yangpengfei/EnvScaler/attacker/data/envs/rl.json")


def filter_rl_entries(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return entries whose top-level key contains 'rl' (case-insensitive)."""
    return {k: v for k, v in data.items() if "rl" in str(k).lower()}


def main() -> None:
    with SOURCE_FILE.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError("Expected source metadata to be a top-level JSON object.")

    filtered = filter_rl_entries(raw)

    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_FILE.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered)} entries to {TARGET_FILE}")


if __name__ == "__main__":
    main()
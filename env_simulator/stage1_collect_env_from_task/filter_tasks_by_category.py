import json
from pathlib import Path

INPUT_FILE = "stage1_collect_env_from_task/temp_result/step0_source_tasks.json"
OUTPUT_DIR = "stage1_collect_env_from_task/filtered_tasks"
OUTPUT_FILE = "all_selected_tasks.json"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def normalize(text: str) -> str:
    return text.lower()


def is_review_task(task: str) -> bool:
    keywords = ["review", "rating", "comment", "feedback", "rate", "evaluate"]
    objects = ["restaurant", "product", "service", "app", "store", "business"]
    t = normalize(task)
    return any(k in t for k in keywords) and any(o in t for o in objects)


def is_payment_task(task: str) -> bool:
    keywords = [
        "payment", "pay", "transfer", "transaction",
        "invoice", "bill", "balance", "credit", "debit"
    ]
    t = normalize(task)
    return any(k in t for k in keywords)


def is_registration_task(task: str) -> bool:
    keywords = [
        "register", "sign up", "signup",
        "create account", "account creation",
        "open an account"
    ]
    t = normalize(task)
    return any(k in t for k in keywords)


def is_message_task(task: str) -> bool:
    keywords = ["message", "email", "mail", "notification", "inbox"]
    actions = ["read", "send", "forward", "receive", "reply"]
    t = normalize(task)
    return any(k in t for k in keywords) and any(a in t for a in actions)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {
        "review_tasks": [],
        "payment_tasks": [],
        "registration_tasks": [],
        "message_tasks": []
    }

    counters = {
        "review_tasks": 0,
        "payment_tasks": 0,
        "registration_tasks": 0,
        "message_tasks": 0
    }

    MAX_PER_CATEGORY = 10

    for item in data:
        task = item["task"]

        if counters["review_tasks"] < MAX_PER_CATEGORY and is_review_task(task):
            result["review_tasks"].append(item)
            counters["review_tasks"] += 1

        if counters["payment_tasks"] < MAX_PER_CATEGORY and is_payment_task(task):
            result["payment_tasks"].append(item)
            counters["payment_tasks"] += 1

        if counters["registration_tasks"] < MAX_PER_CATEGORY and is_registration_task(task):
            result["registration_tasks"].append(item)
            counters["registration_tasks"] += 1

        if counters["message_tasks"] < MAX_PER_CATEGORY and is_message_task(task):
            result["message_tasks"].append(item)
            counters["message_tasks"] += 1

        # 如果四类都达到上限就可以提前结束
        if all(c >= MAX_PER_CATEGORY for c in counters.values()):
            break

    output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Filtering finished (max 10 per category):")
    for k, v in result.items():
        print(f"  {k}: {len(v)} tasks")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()

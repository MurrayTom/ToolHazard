"""
Generate checklist items for task verification.
"""
from typing import Tuple, List, Optional
from utils.call_llm import llm_inference


# Prompt template for generating checklists
input_template = \
"""You are a **Quality Checklist Generation Assistant**.  
I will provide you with a **task description**. Your job is to generate a **simple and uniformly phrased checklist**.

**Requirements:**
1. **Only verify state modifications (writes).** Do NOT generate items that verify read operations (e.g. "Has X been retrieved?", "Has Y been identified?", "Has the list been sorted by Z?"). Verification is done by inspecting the **final state** only; read operations do not leave a trace in state, so such items cannot be checked correctly. Each item must be **verifiable from the final state** (e.g. "Has field F been set to V?", "Has entity E been created?").
2. Each checklist item must be **independent** and **not rely** on the results of other items.
3. Every checklist item must start with the **exact phrase**: **"Has …"** followed by a clear description of the action or field to verify.
4. Use precise fields and exact values; **avoid vague wording**.
5. If the task requires checking multiple **distinct** conditions that all must hold, **split them into separate checklist items**. If multiple conditions are **equivalent** (satisfying any one is enough, OR relationship), **merge them into one item** and explicitly connect the alternatives with the word **"or"** in the checklist text (e.g. `Has X been set to A or B?`). Any explicit equivalent forms given in the original task (any wording such as "A or B", "A (B)", or other clear synonyms) should all appear in that single checklist item, joined with **"or"**. For example, if a task says `assign shipment to route \"R42\" (NightExpress)`, then `\"R42\"` and `\"NightExpress\"` must both be included in one checklist item as acceptable equivalents.
6. List the items in **logical order**, ensuring each is **self-contained**.
7. **CRITICAL - NO UNRELATED ITEMS: Only check fields/attributes that are DIRECTLY MODIFIED or CREATED by the task. DO NOT check:**
   - Whether entities mentioned in the task exist (e.g., if task says "update customer CUST-123", do NOT check if CUST-123 exists - it's pre-existing)
   - Pre-existing relationships (e.g., if task says "set language for session SES4 (used by user USR4)", do NOT check if SES4 belongs to USR4 - that's pre-existing)
   - Contextual information that is not modified (e.g., if task mentions an ID or name for context, do NOT check if that ID/name exists)
   - Things that are already in the state and not modified by the task (do NOT re-check pre-existing state; only verify what the task changed or created)
8. **Output format:**
   - Use Markdown list syntax (`- `) for each checklist item
   - Each item must start with **"Has …"** and be **verifiable with a single boolean expression** (which may be an OR of equivalent conditions).

---

**Example:**  

Task description:
Register a new hospital device with ID DEV-9Z88H, model VNT-900, manufactured by Radiant Health Systems, install it at location LOC-RESP-01 (type: ward), and ensure maintenance schedule MSCH-0101 has a `compliance_status` of `compliant`.

Expected CheckList:
- Has the new device DEV-9Z88H been registered?
- Has the device_id of DEV-9Z88H been set to "DEV-9Z88H"?
- Has the model_number of DEV-9Z88H been set to "VNT-900"?
- Has the manufacturer of DEV-9Z88H been set to "Radiant Health Systems"?
- Has the new location LOC-RESP-01 been created?
- Has the type of location LOC-RESP-01 been set to "ward"?
- Has the compliance_status of maintenance schedule MSCH-0101 been set to "compliant"?
- Has the language of the session been set to "English" or "en"?

---

Now, generate the checklist for the following task:
{task}

Output Format (strictly follow this):
# Analysis
<Your step-by-step reasoning and analysis process>

# CheckList
- Has ...
- Has ...
- ...
"""

def parse_response(output_text: Optional[str]) -> Tuple[bool, List[str]]:
    """Parse checklist items from LLM output.

    Returns (parse_success, checklist). When output_text is None or empty,
    parsing fails and an empty checklist is returned.
    """
    if not output_text:
        return False, []

    if "</think>" in output_text:
        output_text = output_text.split("</think>")[1]

    parse_success = False
    checklist = []

    # Normalize line breaks and split
    lines = output_text.strip().splitlines()
    # Flag to detect we are inside "# CheckList" section
    in_checklist_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# CheckList"):
            in_checklist_section = True
            continue
        if in_checklist_section:
            # Stop parsing if we hit another section header
            if stripped.startswith("# ") and not stripped.startswith("# CheckList"):
                break
            # Checklist item lines must start with "- "
            if stripped.startswith("- "):
                checklist.append(stripped[2:].strip())

    if checklist:
        parse_success = True

    return parse_success, checklist

def gen_checklist(model: str, task: str) -> List[str]:
    """Generate checklist items for the given task using LLM."""
    input_content = input_template.format(task=task)
    messages = [
        {"role": "user", "content": input_content},
    ]
    cur_try = 0
    max_try = 5
    while cur_try < max_try:
        cur_try += 1
        output_text = llm_inference(provider="openai", model=model, messages=messages)
        parse_success, checklist = parse_response(output_text)
        if parse_success:
            break
    return checklist
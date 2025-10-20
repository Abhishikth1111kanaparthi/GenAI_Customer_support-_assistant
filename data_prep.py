# data_prep.py
import pandas as pd
import json
from pathlib import Path

# INPUT: local path to the downloaded kaggle dataset file (CSV or JSON)
INPUT_PATH = "Bitext_Sample_Customer_Support.csv"  # change as necessary
OUTPUT_JSONL = "fine_tune_customer_support.jsonl"

def main():
    path = Path(INPUT_PATH)
    if not path.exists():
        print(f"ERROR: {INPUT_PATH} not found. Download dataset from Kaggle and place it here.")
        return

    if path.suffix.lower() in [".csv"]:
        df = pd.read_csv(path)
    elif path.suffix.lower() in [".json", ".jsonl"]:
        df = pd.read_json(path, lines=True)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSONL.")

    # Inspect likely column names and try to infer them
    possible_user_cols = [c for c in df.columns if "user" in c.lower() or "query" in c.lower() or "question" in c.lower()]
    possible_resp_cols = [c for c in df.columns if "answer" in c.lower() or "reply" in c.lower() or "assistant" in c.lower() or "response" in c.lower()]

    print("Possible user columns:", possible_user_cols)
    print("Possible response columns:", possible_resp_cols)

    # Simple heuristic: pick first candidate or fallback
    user_col = possible_user_cols[0] if possible_user_cols else df.columns[0]
    resp_col = possible_resp_cols[0] if possible_resp_cols else df.columns[1] if len(df.columns)>1 else df.columns[0]

    print(f"Using user col: {user_col}; response col: {resp_col}")

    # Create JSONL in OpenAI fine-tune messages format
    # Each line: {"messages":[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            user_text = str(row.get(user_col, "")).strip()
            resp_text = str(row.get(resp_col, "")).strip()
            if not user_text or not resp_text:
                continue
            obj = {
                "messages": [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": resp_text}
                ]
            }
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote fine-tune jsonl to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()

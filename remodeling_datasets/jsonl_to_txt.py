import json
import math

# ===== FILE PATHS =====
INPUT_FILE = r"/home/freedy/Documents/models/self-instruct-gpt3.jsonl"  # replace with your jsonl
OUTPUT_FILE = r"/home/freedy/Documents/models/stage5_train.txt"

# ===== SPECIAL TOKENS =====
USER = "\x01"
ASSISTANT = "\x02"
EOS = "\x00"

# ===== FILTERS =====
FILTER_TERMS = {"yes", "no", "true", "false", "1", "0"}

def is_filtered(text):
    """
    Returns True if text is a simple label we want to skip
    """
    clean = text.strip().lower()
    # skip if text is only a filter term or too short
    return clean in FILTER_TERMS or len(clean) < 3

# ===== PROCESS JSONL =====
total_written = 0
filtered_count = 0

with open(INPUT_FILE, 'r', encoding='utf-8') as f, \
     open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
    
    for line in f:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        prompt = data.get("prompt", "").strip()
        completion = data.get("completion", "").strip()

        # remove endoftext token if present
        if completion.endswith("<|endoftext|>"):
            completion = completion.replace("<|endoftext|>", "").strip()
        
        # filter unwanted
        if not prompt or not completion or is_filtered(completion):
            filtered_count += 1
            continue
        
        # format for model
        text = f"{USER}{prompt}{ASSISTANT}{completion}{EOS}\n"
        out_f.write(text)
        total_written += 1

print(f"Total written: {total_written}")
print(f"Filtered/skipped: {filtered_count}")
print(f"Output file: {OUTPUT_FILE}")

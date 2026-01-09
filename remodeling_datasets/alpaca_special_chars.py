import json
import random
from tqdm import tqdm

# File paths
input_file = r"C:\Documents\datasets\alpaca\alpaca_data_cleaned.json"
train_file = r"C:\Documents\datasets\alpaca\stage2_train.txt"
val_file   = r"C:\Documents\datasets\alpaca\stage2_val.txt"

# Parameters
train_fraction = 0.9
USER = '\x01'
ASSISTANT = '\x02'
EOS = '\x00'
max_length = 2000  # maximum characters per sample (optional)

# Load JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples.")

# Open output files
with open(train_file, "w", encoding="utf-8") as train_f, \
     open(val_file, "w", encoding="utf-8") as val_f:

    for sample in tqdm(data, desc="Converting", ncols=100, unit="sample"):
        instruction = sample.get("instruction", "").strip()
        output = sample.get("output", "").strip()
        if not instruction or not output:
            continue

        # Optional truncation
        if len(output) > max_length:
            output = output[:max_length]

        # Format for your char-level model
        text = f"{USER}{instruction}{ASSISTANT}{output}{EOS}\n"

        # Randomly assign to train or validation
        if random.random() < train_fraction:
            train_f.write(text)
        else:
            val_f.write(text)

print("Conversion complete!")
print(f"Train file: {train_file}")
print(f"Validation file: {val_file}")

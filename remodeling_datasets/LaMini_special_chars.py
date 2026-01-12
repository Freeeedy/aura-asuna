import pandas as pd
import math

# ===== FILE PATHS =====
PARQUET_FILES = [
    r"C:\Documents\datasets\test-00000-of-00001.parquet",
    r"C:\Documents\datasets\train-00000-of-00001.parquet",
]

TRAIN_FILE = r"C:\Documents\datasets\stage4_train.txt"
VAL_FILE   = r"C:\Documents\datasets\stage4_val.txt"

# ===== SPECIAL TOKENS =====
USER = "\x01"
ASSISTANT = "\x02"
EOS = "\x00"

# ===== LOAD DATA =====
dfs = [pd.read_parquet(p) for p in PARQUET_FILES]
data = pd.concat(dfs, ignore_index=True)

print(f"Loaded {len(data)} rows")
print("Columns:", list(data.columns))

# ===== WRITE SPLIT FILES =====
train_written = 0
val_written = 0
total_written = 0

with open(TRAIN_FILE, "w", encoding="utf-8") as train_f, \
     open(VAL_FILE, "w", encoding="utf-8") as val_f:

    for row in data.itertuples(index=False):
        prompt = getattr(row, "prompt", None)
        completion = getattr(row, "completion", None)

        # Skip NaNs
        if (
            prompt is None
            or completion is None
            or (isinstance(prompt, float) and math.isnan(prompt))
            or (isinstance(completion, float) and math.isnan(completion))
        ):
            continue

        prompt = prompt.strip()
        completion = completion.strip()

        if not prompt or not completion:
            continue

        line = f"{USER}{prompt}{ASSISTANT}{completion}{EOS}\n"

        # Deterministic 90 / 10 split
        if total_written % 10 == 0:
            val_f.write(line)
            val_written += 1
        else:
            train_f.write(line)
            train_written += 1

        total_written += 1

print(f"Train samples: {train_written}")
print(f"Validation samples: {val_written}")
print("Files written:")
print(TRAIN_FILE)
print(VAL_FILE)

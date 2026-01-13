import random

# ===== FILE PATHS =====
INPUT_FILE = r"/home/freedy/Documents/models/stage4_train.txt"  # your original dataset
TRAIN_FILE = r"/home/freedy/Documents/models/train_split.txt"
VAL_FILE   = r"/home/freedy/Documents/models/val_split.txt"

# ===== READ LINES =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# ===== SHUFFLE LINES =====
random.shuffle(lines)  # optional, ensures random distribution

# ===== SPLIT =====
split_idx = int(len(lines) * 0.9)  # 90% train, 10% val
train_lines = lines[:split_idx]
val_lines = lines[split_idx:]

# ===== WRITE FILES =====
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.writelines(val_lines)

print(f"Train lines: {len(train_lines)}")
print(f"Validation lines: {len(val_lines)}")
print(f"Train file saved to: {TRAIN_FILE}")
print(f"Validation file saved to: {VAL_FILE}")

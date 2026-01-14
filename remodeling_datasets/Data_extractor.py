# split_dataset_straight.py

INPUT_FILE = r"C:/Documents/datasets/my/dataset_examples_converted.txt"  # replace with your actual path
TRAIN_FILE = r"C:/Documents/datasets/my/train1.txt"
VAL_FILE = r"C:/Documents/datasets/my/val1.txt"

# ===== READ FILE =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

# ===== GROUP INTO BLOCKS (question + answer) =====
blocks = []
i = 0
while i < len(lines):
    if lines[i].strip():  # skip empty lines
        if i + 1 < len(lines):
            block = lines[i:i+2]  # question + answer
            blocks.append(block)
            i += 2
        else:
            blocks.append([lines[i]])
            i += 1
    else:
        i += 1

print(f"Total blocks found: {len(blocks)}")

# ===== SPLIT 90/10 =====
split_idx = int(len(blocks) * 0.9)
train_blocks = blocks[:split_idx]
val_blocks = blocks[split_idx:]

# ===== WRITE FILES =====
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for block in train_blocks:
        f.writelines(block)

with open(VAL_FILE, "w", encoding="utf-8") as f:
    for block in val_blocks:
        f.writelines(block)

print(f"Train blocks: {len(train_blocks)}")
print(f"Validation blocks: {len(val_blocks)}")
print(f"Train file: {TRAIN_FILE}")
print(f"Validation file: {VAL_FILE}")

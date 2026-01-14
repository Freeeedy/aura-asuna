import random

# ===== FILE PATHS =====
INPUT_FILE = r"C:/Documents/datasets/my/dataset_examples_placeholders.txt"  # your original dataset
TRAIN_FILE = r"C:/Documents/datasets/my/train_split.txt"
VAL_FILE = r"C:/Documents/datasets/my/val_split.txt"

# ===== PARSE INTO COMPLETE EXAMPLES (atomic blocks) =====
examples = []
current_example_lines = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        current_example_lines.append(line)
        
        # When we hit a line containing "/3", that's the end of an example
        if "/3" in line:
            examples.append("".join(current_example_lines))
            current_example_lines = []  # reset for next example

# Handle any leftover (shouldn't happen if file is well-formed)
if current_example_lines:
    print("Warning: File ended without a final /3 — adding incomplete example anyway.")
    examples.append("".join(current_example_lines))

print(f"Total complete examples found: {len(examples)}")

# ===== SHUFFLE EXAMPLES (not individual lines) =====
random.shuffle(examples)  # shuffles the full blocks, keeps them intact

# ===== SPLIT =====
split_idx = int(len(examples) * 0.9)  # 90% train, 10% val
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# ===== WRITE FILES =====
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    f.write("".join(train_examples))

with open(VAL_FILE, "w", encoding="utf-8") as f:
    f.write("".join(val_examples))

print(f"Train examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")
print(f"Train file saved to: {TRAIN_FILE}")
print(f"Validation file saved to: {VAL_FILE}")
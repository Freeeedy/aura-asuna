import os
import csv
import re

# =============================
# FILE PATHS
# =============================

INPUT_FILE = "C:/Documents/datasets/output.txt"

OUT_DIR = "C:/Documents/datasets/"
TRAIN_FILE = os.path.join(OUT_DIR, "stage3_train.txt")
VAL_FILE   = os.path.join(OUT_DIR, "stage3_val.txt")

os.makedirs(OUT_DIR, exist_ok=True)

# =============================
# SPECIAL TOKENS (MUST MATCH MODEL)
# =============================

USER = "\x01"
ASSISTANT = "\x02"
EOS = "\x00"

# =============================
# CLEANING
# =============================

def clean(s):
    if s is None:
        return ""

    s = s.strip()

    # remove surrounding quotes
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # remove Task: and Output: everywhere (case-insensitive)
    s = re.sub(r'\bTask:\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\bOutput:\s*', '', s, flags=re.IGNORECASE)

    # normalize whitespace
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s.strip()

# =============================
# LOAD TSV
# =============================

print("Loading output.txt...")

rows = []
with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) < 2:
            continue

        instr = clean(row[0])
        out   = clean(row[1])

        if not instr or not out:
            continue

        rows.append((instr, out))

print("Total valid samples:", len(rows))

# =============================
# WRITE TRAIN / VAL
# =============================

train_written = 0
val_written = 0
total = 0

with open(TRAIN_FILE, "w", encoding="utf-8") as train_f, \
     open(VAL_FILE, "w", encoding="utf-8") as val_f:

    for instr, out in rows:
        line = f"{USER}{instr}{ASSISTANT}{out}{EOS}\n"

        # deterministic 90/10 split
        if total % 10 == 0:
            val_f.write(line)
            val_written += 1
        else:
            train_f.write(line)
            train_written += 1

        total += 1

# =============================
# DONE
# =============================

print("DONE")
print("Train samples:", train_written)
print("Validation samples:", val_written)
print("Files written:")
print(TRAIN_FILE)
print(VAL_FILE)

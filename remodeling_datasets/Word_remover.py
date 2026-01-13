# remove_words_keep_lines_fixed.py

import re

INPUT_FILE = r"/home/freedy/Documents/models/stage5_train.txt"
OUTPUT_FILE = r"/home/freedy/Documents/models/stage4_train.txt"

# Words/phrases to remove (case-insensitive)
REMOVE_WORDS = [
    "Output:",
    "Task:"
]

# ===== READ FILE =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()  # preserve line breaks

# ===== REMOVE WORDS =====
cleaned_lines = []
for line in lines:
    for word in REMOVE_WORDS:
        # Remove the word/phrase globally, ignore case
        line = re.sub(re.escape(word), "", line, flags=re.IGNORECASE)
    # Remove extra spaces left by deletion but keep line breaks
    line = re.sub(r' +', ' ', line)
    cleaned_lines.append(line.rstrip() + '\n')

# ===== WRITE CLEANED FILE =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(cleaned_lines)

print(f"Finished cleaning! Output saved to: {OUTPUT_FILE}")

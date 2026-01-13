import pandas as pd

# ===== FILE PATHS =====
CSV_FILE = r"C:\Documents\datasets\train.csv"
OUTPUT_FILE = r"C:\Documents\datasets\output.txt"

# ===== READ CSV =====
df = pd.read_csv(CSV_FILE)

# ===== SAVE AS TXT =====
df.to_csv(OUTPUT_FILE, sep='\t', index=False)

print(f"CSV converted to TXT: {OUTPUT_FILE}")

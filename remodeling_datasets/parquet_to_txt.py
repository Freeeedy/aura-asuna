import pandas as pd

# ===== FILE PATHS =====
PARQUET_FILES = [
    r"C:\Documents\datasets\test-00000-of-00001.parquet",
    r"C:\Documents\datasets\train-00000-of-00001.parquet",
]

OUTPUT_FILE = r"C:\Documents\datasets\combined.txt"

# ===== READ AND COMBINE =====
dataframes = []
for file in PARQUET_FILES:
    df = pd.read_parquet(file)
    dataframes.append(df)

# Combine the two datasets vertically (stacking rows)
combined_df = pd.concat(dataframes, ignore_index=True)

# ===== SAVE AS TXT =====
# You can choose the separator, here we use tab-separated
combined_df.to_csv(OUTPUT_FILE, sep='\t', index=False)

print(f"Combined dataset saved to {OUTPUT_FILE}")

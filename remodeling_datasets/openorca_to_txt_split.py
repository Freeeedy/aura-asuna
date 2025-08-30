import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the parquet file
df = pd.read_parquet("C:\\Users\\Uživatel\\Downloads\\3_5M-GPT3_5-Augmented.parquet")   # <-- change path if needed
print("Columns:", df.columns)
print("Total rows:", len(df))

# 2. Split into train/val
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# 3. Helper to write splits into .txt
def write_split(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for _, row in data.iterrows():
            system = row.get("system_prompt", "")
            question = row.get("question", "")
            response = row.get("response", "")
            # format like a dialogue
            text = f"System: {system}\nUser: {question}\nAssistant: {response}\n"
            f.write(text + "\n")

# 4. Write to txt files
write_split("train_openorca.txt", train_df)
write_split("val_openorca.txt", val_df)

print("✅ Done! Wrote train_openorca.txt and val_openorca.txt")

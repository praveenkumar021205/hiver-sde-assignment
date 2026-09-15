import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/amazon_conversations.csv")
OUTPUT_FILE = Path("evaluation/golden_set_200.csv")

# Read the processed Amazon conversations
df = pd.read_csv(INPUT_FILE)

# Keep only useful customer messages
df = df[
    df["customer_text"].notna()
].copy()

df["customer_text"] = (
    df["customer_text"]
    .astype(str)
    .str.strip()
)

# Remove very short messages
df = df[df["customer_text"].str.len() >= 10]

# Remove duplicate customer messages
df = df.drop_duplicates(
    subset=["customer_text"]
)

# Randomly select 200 real customer messages
golden = df.sample(
    n=min(200, len(df)),
    random_state=42
).copy()

# Keep only the columns needed for manual annotation
golden = golden[
    ["customer_text"]
].rename(
    columns={
        "customer_text": "text"
    }
)

# Empty column for HUMAN verification
golden["intent"] = ""

# Save
golden.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("=" * 60)
print("GOLDEN SET CREATED")
print("=" * 60)
print(f"Rows: {len(golden)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)
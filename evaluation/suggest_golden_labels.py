import pandas as pd

INPUT_FILE = "evaluation/golden_set_200_suggested.csv"
OUTPUT_FILE = "evaluation/golden_set_200.csv"

df = pd.read_csv(INPUT_FILE)

# Use the model-generated suggestion as the initial label.
# These must be reviewed by a human before being treated as final.
df["intent"] = df["suggested_intent"]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("=" * 60)
print("GOLDEN SET CREATED")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("IMPORTANT:")
print("Review the intent column before final submission.")
print("The suggested labels are only initial labels.")
print("=" * 60)
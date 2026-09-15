import pandas as pd

INPUT_FILE = "evaluation/golden_set_200_suggested.csv"
OUTPUT_FILE = "evaluation/golden_set_200.csv"

df = pd.read_csv(INPUT_FILE)

# Use the suggested label as the initial label.
# These labels must still be human-reviewed before final submission.
df["intent"] = df["suggested_intent"]

# Keep only the columns required for evaluation
df = df[["text", "intent"]]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("=" * 60)
print("GOLDEN SET PREPARED")
print("=" * 60)
print(f"Rows: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")
print()
print("IMPORTANT:")
print("The intent labels are initial suggested labels.")
print("They should be human-reviewed before final submission.")
print("=" * 60)
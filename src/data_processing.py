import pandas as pd
from pathlib import Path

RAW_FILE = Path("data/raw/twcs.csv")
OUTPUT_FILE = Path("data/processed/amazon_conversations.csv")

print("=" * 70)
print("AmazonHelp Conversation Data Processing")
print("=" * 70)

if not RAW_FILE.exists():
    print(f"ERROR: {RAW_FILE} was not found.")
    raise SystemExit(1)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("\nStep 1: Finding AmazonHelp responses...")

amazon_response_ids = []
amazon_responses = []

for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=100000,
    low_memory=False
):
    # AmazonHelp response tweets
    mask = (
        (chunk["inbound"] == False) &
        (chunk["author_id"].astype(str) == "AmazonHelp")
    )

    responses = chunk.loc[
        mask,
        [
            "tweet_id",
            "text",
            "created_at",
            "in_response_to_tweet_id"
        ]
    ].copy()

    if len(responses) > 0:
        amazon_responses.append(responses)

amazon = pd.concat(amazon_responses, ignore_index=True)

print(f"Found {len(amazon):,} AmazonHelp responses.")

# Normalize tweet IDs so values like 272.0 and 272 match
amazon["in_response_to_tweet_id"] = pd.to_numeric(
    amazon["in_response_to_tweet_id"],
    errors="coerce"
).astype("Int64")

amazon["tweet_id"] = pd.to_numeric(
    amazon["tweet_id"],
    errors="coerce"
).astype("Int64")

amazon = amazon.dropna(subset=["in_response_to_tweet_id"])

customer_ids = set(
    amazon["in_response_to_tweet_id"].astype(str)
)

print("\nStep 2: Finding the customer tweets...")

customers = []

for chunk in pd.read_csv(
    RAW_FILE,
    chunksize=100000,
    low_memory=False
):
    chunk["tweet_id_num"] = pd.to_numeric(
        chunk["tweet_id"],
        errors="coerce"
    ).astype("Int64")

    mask = chunk["tweet_id_num"].astype(str).isin(customer_ids)

    selected = chunk.loc[
        mask,
        ["tweet_id_num", "text", "created_at"]
    ].copy()

    if len(selected) > 0:
        customers.append(selected)

customers = pd.concat(customers, ignore_index=True)

print(f"Found {len(customers):,} customer tweets.")

print("\nStep 3: Joining customer messages with AmazonHelp responses...")

customers = customers.rename(
    columns={
        "tweet_id_num": "customer_tweet_id",
        "text": "customer_text",
        "created_at": "customer_created_at"
    }
)

amazon = amazon.rename(
    columns={
        "tweet_id": "amazon_response_tweet_id",
        "text": "amazon_response",
        "created_at": "amazon_response_created_at",
        "in_response_to_tweet_id": "customer_tweet_id"
    }
)

result = customers.merge(
    amazon,
    on="customer_tweet_id",
    how="inner"
)

result = result[
    [
        "customer_tweet_id",
        "customer_created_at",
        "customer_text",
        "amazon_response_tweet_id",
        "amazon_response",
        "amazon_response_created_at"
    ]
]

result = result.drop_duplicates()

result.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("SUCCESS!")
print("=" * 70)
print(f"Conversations created: {len(result):,}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 70)
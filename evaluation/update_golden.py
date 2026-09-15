import pandas as pd

path = "evaluation/golden_set_200.csv"

df = pd.read_csv(path)

labels = [
    # 0-19
    "missing_delivery", "shipping_information", "other", "device_product",
    "return_refund", "missing_delivery", "device_product", "return_refund",
    "missing_delivery", "promotion", "other", "other", "other", "other",
    "account_issue", "missing_delivery", "other", "other", "other", "other",

    # 20-39
    "missing_delivery", "other", "other", "order_issue", "other",
    "shipping_information", "delivery_tracking", "missing_delivery", "other",
    "return_refund", "device_product", "missing_delivery", "promotion",
    "missing_delivery", "order_issue", "other", "promotion", "return_refund",
    "shipping_information", "other",

    # 40-59
    "delivery_tracking", "other", "other", "order_issue", "account_issue",
    "shipping_information", "shipping_information", "missing_delivery",
    "other", "order_issue", "device_product", "missing_delivery",
    "missing_delivery", "shipping_information", "other", "other",
    "account_issue", "account_issue", "other", "shipping_information",

    # 60-79
    "other", "other", "missing_delivery", "delivery_tracking",
    "missing_delivery", "missing_delivery", "return_refund",
    "shipping_information", "shipping_information", "delivery_tracking",
    "device_product", "other", "other", "missing_delivery",
    "delivery_tracking", "device_product", "other", "missing_delivery",
    "delivery_tracking", "delivery_tracking",

    # 80-99
    "order_issue", "shipping_information", "shipping_information", "other",
    "return_refund", "other", "other", "return_refund", "other",
    "missing_delivery", "other", "order_issue", "other", "order_issue",
    "other", "other", "delivery_tracking", "order_issue", "other",
    "account_issue",

    # 100-119
    "shipping_information", "other", "delivery_tracking",
    "shipping_information", "device_product", "missing_delivery",
    "order_issue", "other", "account_issue", "other", "other",
    "order_issue", "return_refund", "payment_billing", "account_issue",
    "other", "other", "order_issue", "return_refund", "missing_delivery",

    # 120-139
    "delivery_tracking", "account_issue", "delivery_tracking",
    "device_product", "other", "other", "return_refund", "device_product",
    "delivery_tracking", "other", "other", "device_product",
    "delivery_tracking", "other", "payment_billing", "return_refund",
    "delivery_tracking", "device_product", "other", "other",

    # 140-159
    "missing_delivery", "other", "order_issue", "payment_billing",
    "return_refund", "delivery_tracking", "payment_billing", "account_issue",
    "device_product", "shipping_information", "return_refund", "return_refund",
    "device_product", "other", "missing_delivery", "payment_billing",
    "missing_delivery", "other", "return_refund", "promotion",

    # 160-179
    "other", "missing_delivery", "other", "shipping_information", "other",
    "missing_delivery", "other", "other", "delivery_tracking", "other",
    "other", "return_refund", "delivery_tracking", "order_issue",
    "missing_delivery", "other", "shipping_information", "delivery_tracking",
    "other", "other",

    # 180-199
    "prime_video", "order_issue", "order_issue", "other", "missing_delivery",
    "missing_delivery", "order_issue", "order_issue", "shipping_information",
    "other", "other", "order_issue", "device_product", "return_refund",
    "missing_delivery", "account_issue", "payment_billing", "missing_delivery",
    "device_product", "other"
]

print("CSV rows:", len(df))
print("Labels:", len(labels))

assert len(df) == 200, "CSV does not contain exactly 200 rows."
assert len(labels) == 200, "Labels do not contain exactly 200 entries."

df["intent"] = labels
df.to_csv(path, index=False)

print("\nSUCCESS!")
print("All 200 reviewed labels have been saved.")
print("\nFinal intent distribution:")
print(df["intent"].value_counts().sort_index())
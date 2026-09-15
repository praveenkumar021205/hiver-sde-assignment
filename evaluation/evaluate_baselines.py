import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

GOLDEN_PATH = "evaluation/golden_set_200.csv"

df = pd.read_csv(GOLDEN_PATH)

y_true = df["intent"]


# ---------------------------------------------------------
# BASELINE 1: TRIVIAL MAJORITY CLASS
# ---------------------------------------------------------

majority_class = y_true.value_counts().idxmax()

trivial_predictions = [majority_class] * len(df)

trivial_accuracy = accuracy_score(
    y_true,
    trivial_predictions
)

trivial_f1 = f1_score(
    y_true,
    trivial_predictions,
    average="macro",
    zero_division=0
)


# ---------------------------------------------------------
# BASELINE 2: SIMPLE KEYWORD RULES
# ---------------------------------------------------------

def keyword_predict(text):

    text = str(text).lower()

    rules = [
        (
            "account_issue",
            [
                "account",
                "login",
                "log in",
                "sign in",
                "password",
                "hacked",
                "locked",
                "suspended"
            ]
        ),

        (
            "return_refund",
            [
                "refund",
                "return",
                "money back",
                "reimburse"
            ]
        ),

        (
            "payment_billing",
            [
                "charged",
                "charge",
                "payment",
                "billing",
                "paid",
                "card",
                "transaction"
            ]
        ),

        (
            "prime_video",
            [
                "prime video",
                "primevideo"
            ]
        ),

        (
            "device_product",
            [
                "kindle",
                "alexa",
                "fire stick",
                "firestick",
                "echo",
                "device",
                "tablet"
            ]
        ),

        (
            "cancellation",
            [
                "cancel",
                "cancellation"
            ]
        ),

        (
            "promotion",
            [
                "promotion",
                "discount",
                "offer",
                "coupon",
                "promo"
            ]
        ),

        (
            "delivery_tracking",
            [
                "tracking",
                "track",
                "tracking number",
                "shipment status",
                "where is my package"
            ]
        ),

        (
            "missing_delivery",
            [
                "not delivered",
                "never arrived",
                "haven't received",
                "not received",
                "missing package",
                "missing parcel",
                "package is missing",
                "delivery is late",
                "delivery was late"
            ]
        ),

        (
            "shipping_information",
            [
                "shipping",
                "delivery options",
                "shipping cost",
                "shipping fee",
                "delivery address",
                "shipping address",
                "how long does shipping take"
            ]
        ),

        (
            "order_issue",
            [
                "wrong item",
                "wrong product",
                "damaged",
                "broken",
                "defective",
                "wrong order",
                "order problem"
            ]
        )
    ]

    for intent, keywords in rules:

        for keyword in keywords:

            if keyword in text:
                return intent

    return "other"


keyword_predictions = [
    keyword_predict(text)
    for text in df["text"]
]

keyword_accuracy = accuracy_score(
    y_true,
    keyword_predictions
)

keyword_f1 = f1_score(
    y_true,
    keyword_predictions,
    average="macro",
    zero_division=0
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

print(f"\nGolden examples: {len(df)}")

print("\n1. TRIVIAL MAJORITY BASELINE")
print("-" * 40)

print(f"Majority class : {majority_class}")
print(f"Accuracy       : {trivial_accuracy:.4f}")
print(f"Macro F1       : {trivial_f1:.4f}")


print("\n2. SIMPLE KEYWORD BASELINE")
print("-" * 40)

print(f"Accuracy       : {keyword_accuracy:.4f}")
print(f"Macro F1       : {keyword_f1:.4f}")


print("\n3. TF-IDF + LOGISTIC REGRESSION MODEL")
print("-" * 40)

print("Accuracy       : 0.4050")
print("Macro F1       : 0.3053")


print("\n" + "=" * 60)
print("BASELINE EVALUATION COMPLETE")
print("=" * 60)
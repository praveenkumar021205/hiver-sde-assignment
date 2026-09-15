import os
import sys
import pandas as pd

# Allow imports from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_DIR)

from intent_classifier import IntentClassifier
from retriever import Retriever
from escalation import decide_escalation
from reply_generator import generate_reply


GOLDEN_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "golden_set_200.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "evaluation",
    "agent_evaluation_200.csv"
)


def main():

    print("=" * 60)
    print("AMAZONHELP END-TO-END AGENT EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load golden set
    # ---------------------------------------------------------

    df = pd.read_csv(GOLDEN_PATH)

    print(f"\nGolden examples: {len(df)}")

    # ---------------------------------------------------------
    # Load classifier
    # ---------------------------------------------------------

    print("\nLoading intent classifier...")

    classifier = IntentClassifier()

    # ---------------------------------------------------------
    # Load retriever
    # ---------------------------------------------------------

    print("\nLoading historical conversations...")

    retriever = Retriever()

    results = []

    # ---------------------------------------------------------
    # Evaluate every golden example
    # ---------------------------------------------------------

    print("\nRunning end-to-end agent evaluation...\n")

    for index, row in df.iterrows():

        customer_message = str(row["text"])
        expected_intent = row["intent"]

        # Intent classification
        predicted_intent, confidence = classifier.predict(
            customer_message
        )

        # Historical retrieval
        best_match = retriever.search(
            customer_message,
            top_k=1
        )

        if best_match:

            retrieval_score = float(
                best_match[0].get("similarity", 0.0)
            )

            historical_response = best_match[0].get(
                "response_text",
                ""
            )

            selected_match = best_match[0]

        else:

            retrieval_score = 0.0
            historical_response = ""
            selected_match = None

        # Escalation decision
        decision, reason = decide_escalation(
            predicted_intent,
            confidence,
            retrieval_score
        )

        # Generate reply
        draft_reply = generate_reply(
            customer_message,
            predicted_intent,
            selected_match
        )

        results.append({
            "row_id": index,
            "customer_message": customer_message,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_confidence": confidence,
            "retrieval_score": retrieval_score,
            "escalation_decision": decision,
            "escalation_reason": reason,
            "historical_response": historical_response,
            "draft_reply": draft_reply
        })

        if (index + 1) % 25 == 0:
            print(
                f"Processed {index + 1}/{len(df)} examples..."
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    correct = (
        results_df["expected_intent"]
        == results_df["predicted_intent"]
    ).sum()

    accuracy = correct / len(results_df)

    escalation_rate = (
        results_df["escalation_decision"]
        == "ESCALATE"
    ).mean()

    print("\n" + "=" * 60)
    print("END-TO-END RESULTS")
    print("=" * 60)

    print(
        f"\nIntent accuracy: {accuracy:.4f}"
    )

    print(
        f"Escalation rate: {escalation_rate:.4f}"
    )

    print("\nEscalation distribution:")

    print(
        results_df["escalation_decision"]
        .value_counts()
    )

    print("\nAverage intent confidence:")

    print(
        f"{results_df['intent_confidence'].mean():.4f}"
    )

    print("\nAverage retrieval score:")

    print(
        f"{results_df['retrieval_score'].mean():.4f}"
    )

    print("\nResults saved to:")

    print(OUTPUT_PATH)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
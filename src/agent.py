from intent_classifier import IntentClassifier
from retriever import Retriever
from escalation import decide_escalation
from reply_generator import generate_reply


def main():
    print("=" * 60)
    print("AmazonHelp AI Support Agent")
    print("=" * 60)
    print("Type 'exit' to stop.\n")

    # Load the intent classifier
    print("Loading intent classifier...")
    classifier = IntentClassifier()

    # Load the historical conversation retriever
    print("Loading historical conversations...")
    retriever = Retriever()

    print("\nAgent is ready!")
    print("-" * 60)

    while True:

        customer_message = input("\nCustomer: ").strip()

        if customer_message.lower() == "exit":
            print("\nAgent stopped.")
            break

        if not customer_message:
            print("Please enter a customer message.")
            continue

        try:
            # --------------------------------------------------
            # 1. INTENT CLASSIFICATION
            # --------------------------------------------------
            intent, confidence = classifier.predict(customer_message)

            # --------------------------------------------------
            # 2. RETRIEVE SIMILAR HISTORICAL CONVERSATIONS
            # --------------------------------------------------
            results = retriever.search(
                customer_message,
                top_k=3
            )

            if not results:
                print("\nNo historical examples found.")
                best_match = {
                    "customer_text": "",
                    "response_text": "",
                    "similarity": 0.0
                }
                retrieval_score = 0.0

            else:
                best_match = results[0]
                retrieval_score = best_match["similarity"]

            # --------------------------------------------------
            # 3. ESCALATION DECISION
            # --------------------------------------------------
            decision, reason = decide_escalation(
                intent,
                confidence,
                retrieval_score
            )

            # --------------------------------------------------
            # 4. GENERATE GROUNDED REPLY
            # --------------------------------------------------
            draft_reply = generate_reply(
                customer_message,
                intent,
                best_match
            )

            # --------------------------------------------------
            # 5. DISPLAY RESULT
            # --------------------------------------------------
            print("\n" + "=" * 60)

            print("INTENT")
            print(intent)

            print("\nINTENT CONFIDENCE")
            print(round(confidence, 3))

            print("\nBEST HISTORICAL MATCH")
            print(round(retrieval_score, 3))

            print("\nESCALATION DECISION")
            print(decision)

            print("\nREASON")
            print(reason)

            print("\nDRAFT REPLY")
            print("-" * 60)
            print(draft_reply)

            print("\n" + "=" * 60)

        except Exception as error:
            print("\nERROR:")
            print(error)
            print("\nThe agent could not process this message.")


if __name__ == "__main__":
    main()
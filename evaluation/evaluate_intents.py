import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report


GOLDEN_FILE = "evaluation/golden_set_200.csv"
MODEL_FILE = "data/processed/intent_model.joblib"
OUTPUT_FILE = "evaluation/intent_predictions_200.csv"


def main():

    print("=" * 60)
    print("INTENT CLASSIFIER EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Check golden set
    # ---------------------------------------------------------
    if not os.path.exists(GOLDEN_FILE):
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_FILE}"
        )

    df = pd.read_csv(GOLDEN_FILE)

    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError(
            "golden_set_200.csv must contain: text,intent"
        )

    df = df.dropna(subset=["text", "intent"])

    print(f"Golden examples: {len(df)}")

    # ---------------------------------------------------------
    # 2. Load already-trained model
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    print("\nLoading trained intent model...")

    model = joblib.load(MODEL_FILE)

    print("Model loaded successfully.")

    # ---------------------------------------------------------
    # 3. Predict
    # ---------------------------------------------------------
    print("\nPredicting golden examples...")

    X = df["text"].astype(str)
    y_true = df["intent"].astype(str)

    # Most likely case: saved sklearn Pipeline
    if hasattr(model, "predict"):

        predictions = model.predict(X)

    # If the saved file is a dictionary
    elif isinstance(model, dict):

        if "model" in model:
            classifier = model["model"]

            if "vectorizer" in model:
                vectorizer = model["vectorizer"]
                X_vectorized = vectorizer.transform(X)
                predictions = classifier.predict(X_vectorized)
            else:
                predictions = classifier.predict(X)

        elif "classifier" in model:
            classifier = model["classifier"]

            if "vectorizer" in model:
                vectorizer = model["vectorizer"]
                X_vectorized = vectorizer.transform(X)
                predictions = classifier.predict(X_vectorized)
            else:
                predictions = classifier.predict(X)

        else:
            raise ValueError(
                "Could not find model/classifier inside joblib file."
            )

    else:
        raise ValueError(
            "Unsupported model format in intent_model.joblib"
        )

    predictions = [str(x) for x in predictions]

    # ---------------------------------------------------------
    # 4. Calculate metrics
    # ---------------------------------------------------------
    accuracy = accuracy_score(
        y_true,
        predictions
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # 5. Save predictions
    # ---------------------------------------------------------
    df["predicted_intent"] = predictions

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(f"Predictions saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
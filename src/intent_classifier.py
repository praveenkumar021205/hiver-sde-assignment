import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "amazon_conversations.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "intent_model.joblib"
)


# Intent examples based on the Amazon customer-support taxonomy
TRAINING_EXAMPLES = {
    "delivery_tracking": [
        "Where is my package?",
        "Can you track my order?",
        "What is the tracking status?",
        "Where can I find my tracking number?",
        "My package is in transit",
        "My order tracking has not updated",
        "When will my package arrive?",
        "Can you tell me where my parcel is?",
        "I want to track my delivery",
        "What is the status of my shipment?",
        "My tracking information has not changed",
        "Where is my shipment?",
        "Please check the tracking",
        "Can you give me an update on my package?",
        "The tracking says my package is on the way"
    ],

    "missing_delivery": [
        "My package has not arrived",
        "I haven't received my order",
        "Where is my package? It should have arrived",
        "My delivery is late",
        "My order was supposed to arrive yesterday",
        "The package was not delivered",
        "My parcel is missing",
        "I still haven't received my package",
        "My order never arrived",
        "The delivery date has passed",
        "My package is overdue",
        "Amazon says delivered but I don't have it",
        "My order is missing",
        "The driver did not deliver my package",
        "My delivery has not arrived"
    ],

    "order_issue": [
        "I received the wrong item",
        "My order is damaged",
        "The product I received is not what I ordered",
        "There is a problem with my order",
        "My order arrived damaged",
        "I received a different product",
        "The item I ordered is incorrect",
        "Something is wrong with my order",
        "My product arrived broken",
        "I have an issue with my order",
        "The item I received is defective",
        "My order has a problem",
        "The wrong product was delivered",
        "I received the wrong product",
        "There is something wrong with my purchase"
    ],

    "return_refund": [
        "I want a refund",
        "How can I return my item?",
        "I need to return my order",
        "Where is my refund?",
        "I want my money back",
        "How do I get a refund?",
        "My refund has not arrived",
        "Can I return this product?",
        "I need a refund for my order",
        "How do I return an item?",
        "Please refund my purchase",
        "I returned the item but haven't received my refund",
        "I want to send this item back",
        "When will I receive my refund?",
        "How can I get my money back?"
    ],

    "cancellation": [
        "I want to cancel my order",
        "Cancel my order",
        "How can I cancel my purchase?",
        "Please cancel my order",
        "Can I cancel this order?",
        "I need to cancel an order",
        "I don't want this order anymore",
        "Cancel this purchase",
        "How do I cancel an order?",
        "Can you cancel my order?"
    ],

    "payment_billing": [
        "I was charged twice",
        "Why was I charged?",
        "There is a problem with my payment",
        "My card was charged",
        "I have a billing issue",
        "Why did Amazon charge me?",
        "I was charged the wrong amount",
        "My payment failed",
        "Payment is not working",
        "I don't recognize this charge",
        "Why was I charged extra?",
        "There is an unexpected charge",
        "I have a payment problem",
        "Amazon charged my card",
        "I need help with a payment"
    ],

    "account_issue": [
        "I cannot log in",
        "My account is locked",
        "I cannot access my Amazon account",
        "My account was hacked",
        "I forgot my password",
        "I need help with my account",
        "My Amazon account is not working",
        "I cannot sign into my account",
        "Someone accessed my account",
        "I lost access to my account",
        "My account has been suspended",
        "I cannot access my account",
        "I need to change my account details",
        "There is a problem with my account",
        "My account login is not working"
    ],

    "prime_video": [
        "Prime Video is not working",
        "I cannot watch Prime Video",
        "Prime Video won't play",
        "My Prime Video is not working",
        "I have a problem with Prime Video",
        "Prime Video is not loading",
        "Why can't I watch Prime Video?",
        "The Prime Video app is not working",
        "I cannot play a Prime Video",
        "Prime Video playback is not working"
    ],

    "device_product": [
        "My Kindle is not working",
        "Alexa is not working",
        "My Fire Stick is not working",
        "I have a problem with my Amazon device",
        "The product does not work",
        "My device is broken",
        "How does this Amazon device work?",
        "My Kindle has a problem",
        "My Alexa has stopped working",
        "I have a problem with my Fire TV",
        "The product I bought is not working",
        "My Amazon device is not working",
        "I need help with my Kindle",
        "I need help with Alexa",
        "I have a question about an Amazon product"
    ],

    "promotion": [
        "How do I use this promotion?",
        "I have a problem with a promotional offer",
        "Why did my promotional discount disappear?",
        "How can I get the promotion?",
        "The promotion is not working",
        "I was promised a promotional discount",
        "Where is my promotional credit?",
        "How do I claim this offer?",
        "The discount is not showing",
        "I have an issue with an offer"
    ],

    "shipping_information": [
        "How much is shipping?",
        "What shipping options are available?",
        "Can you tell me about shipping?",
        "Do you offer free shipping?",
        "How long does shipping take?",
        "What are the shipping charges?",
        "Can I change the shipping address?",
        "I need information about shipping",
        "What delivery options do you have?",
        "How does shipping work?",
        "Is expedited shipping available?",
        "Can I change where my package is delivered?",
        "What are the shipping methods?",
        "Tell me about delivery options",
        "How much does delivery cost?"
    ],

    "other": [
        "Thanks for your help",
        "Okay thank you",
        "I appreciate your response",
        "Please help me",
        "I need assistance",
        "Can someone help me?",
        "That does not help",
        "I am still waiting for a response",
        "Please respond",
        "Hello Amazon",
        "I have a question",
        "Can you help?",
        "Thanks",
        "Any update?",
        "I need help"
    ]
}


class IntentClassifier:

    def __init__(self):
        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    max_features=30000
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced"
                )
            )
        ])

        self.train()

    def train(self):

        texts = []
        labels = []

        for intent, examples in TRAINING_EXAMPLES.items():
            for text in examples:
                texts.append(text)
                labels.append(intent)

        print(
            f"Training intent classifier on {len(texts)} examples..."
        )

        self.pipeline.fit(texts, labels)

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True
        )

        joblib.dump(
            self.pipeline,
            MODEL_PATH
        )

        print(
            f"Model saved to: {MODEL_PATH}"
        )

    def predict(self, text):

        probabilities = self.pipeline.predict_proba([text])[0]

        classes = self.pipeline.classes_

        best_index = probabilities.argmax()

        intent = classes[best_index]

        confidence = float(
            probabilities[best_index]
        )

        return intent, confidence


if __name__ == "__main__":

    classifier = IntentClassifier()

    test_messages = [
        "Where is my Amazon package?",
        "I want a refund",
        "I cannot log into my account",
        "Prime Video is not working",
        "Why was I charged twice?"
    ]

    print("\nTesting classifier:\n")

    for message in test_messages:

        intent, confidence = classifier.predict(message)

        print(
            f"{message}\n"
            f" -> {intent} ({confidence:.3f})\n"
        )
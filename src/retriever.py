import re
import pandas as pd

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "amazon_conversations.csv"


class Retriever:

    def __init__(self):

        print("Loading Amazon conversation data...")

        self.df = pd.read_csv(DATA_FILE)

        self.df["customer_text"] = (
            self.df["customer_text"]
            .fillna("")
            .astype(str)
        )

        self.df["amazon_response"] = (
            self.df["amazon_response"]
            .fillna("")
            .astype(str)
        )

        # Remove empty customer messages
        self.df = self.df[
            self.df["customer_text"].str.strip() != ""
        ].copy()

        # Normalize text
        self.df["normalized_text"] = (
            self.df["customer_text"]
            .apply(self.normalize_text)
        )

        # Remove exact duplicate customer messages
        self.df = self.df.drop_duplicates(
            subset=["normalized_text"]
        ).reset_index(drop=True)

        print(
            f"Loaded {len(self.df):,} unique conversations."
        )

        print("Building search index...")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(
            self.df["normalized_text"]
        )

        print("Search index ready.")

    @staticmethod
    def normalize_text(text):

        text = str(text).lower()

        # Remove URLs
        text = re.sub(
            r"https?://\S+",
            " ",
            text
        )

        # Remove Twitter handles
        text = re.sub(
            r"@\w+",
            " ",
            text
        )

        # Remove excessive whitespace
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    def search(self, query, top_k=3):

        normalized_query = self.normalize_text(query)

        if not normalized_query:
            return []

        query_vector = self.vectorizer.transform(
            [normalized_query]
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix
        ).flatten()

        # Exclude exact textual matches.
        exact_matches = (
            self.df["normalized_text"] == normalized_query
        )

        scores[exact_matches.values] = -1

        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            score = float(scores[index])

            # Ignore invalid results
            if score < 0:
                continue

            results.append({
                "customer_text":
                    self.df.iloc[index]["customer_text"],

                "response_text":
                    self.df.iloc[index]["amazon_response"],

                "similarity":
                    score
            })

        return results


if __name__ == "__main__":

    retriever = Retriever()

    while True:

        query = input("\nCustomer message: ")

        if query.lower() == "exit":
            break

        results = retriever.search(
            query,
            top_k=3
        )

        print("\nTop historical matches:")

        for i, result in enumerate(results, 1):

            print(f"\n--- Match {i} ---")

            print(
                "Customer:",
                result["customer_text"]
            )

            print(
                "AmazonHelp:",
                result["response_text"]
            )

            print(
                "Similarity:",
                round(
                    result["similarity"],
                    3
                )
            )
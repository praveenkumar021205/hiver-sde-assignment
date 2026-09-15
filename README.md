

# Hiver SDE Intern — AI Support Agent

## 1. Overview

This project builds an AI customer-support agent for **AmazonHelp** using the Customer Support on Twitter dataset.

The agent performs three main tasks:

1. Classifies an incoming customer message into a support intent.
2. Retrieves historically similar Amazon support conversations and uses them to ground a draft reply.
3. Decides whether the case can be automatically handled or should be escalated to a human.

The main goal was not only to build a working pipeline, but also to evaluate where the system succeeds and where it fails.

---

## 2. Dataset

The project uses the **Customer Support on Twitter** dataset.

The dataset contains customer-support conversations between customers and brands on Twitter.

Important fields include:

* `tweet_id`
* `author_id`
* `inbound`
* `created_at`
* `text`
* `response_tweet_id`
* `in_response_to_tweet_id`

### Brand selection

I selected **AmazonHelp** because it had the largest number of response examples in the dataset.

Processing produced:

* AmazonHelp responses: **169,840**
* Customer tweets found: **154,976**
* Joined Amazon conversations: **168,823**

The raw dataset is not included in the repository because of its large size.

---

# 3. System Architecture

```text
Customer Message
       |
       v
+--------------------+
| Intent Classifier  |
| TF-IDF + Logistic  |
| Regression         |
+--------------------+
       |
       v
Predicted Intent
       |
       +----------------------+
       |                      |
       v                      v
+----------------+    +------------------+
| TF-IDF         |    | Confidence       |
| Retriever      |    | / Intent         |
+----------------+    +------------------+
       |
       v
Historical Amazon
Support Response
       |
       v
+--------------------+
| Reply Generator    |
| Grounded/Fallback  |
+--------------------+
       |
       v
+--------------------+
| Escalation Logic   |
+--------------------+
       |
       v
AUTO_HANDLE / ESCALATE
```

---

# 4. Intent Taxonomy

The initial intent taxonomy was derived from recurring customer-support issues in the Amazon conversations.

The system currently uses:

* `delivery_tracking`
* `missing_delivery`
* `shipping_information`
* `order_issue`
* `return_refund`
* `cancellation`
* `payment_billing`
* `account_issue`
* `prime_video`
* `device_product`
* `promotion`
* `other`

The taxonomy intentionally remains small rather than attempting to reproduce every possible support category.

---

# 5. Intent Classifier

The classifier uses:

* TF-IDF features
* Unigrams and bigrams
* Logistic Regression
* Balanced class weights

Configuration:

```text
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    max_features=30000
)

LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)
```

The classifier was trained using a small manually constructed set of representative intent examples.

---

# 6. Retrieval

Historical Amazon support responses are indexed using TF-IDF similarity.

For an incoming customer message:

1. The message is converted into TF-IDF features.
2. Similar historical customer messages are retrieved.
3. The associated Amazon support response is used as historical grounding.
4. A similarity score determines whether the historical response is sufficiently similar.

A grounding threshold of approximately **0.45** is currently used for reply generation.

An important finding was that retrieval similarity alone is not sufficient to guarantee correctness.

A high similarity score can still correspond to the wrong intent.

---

# 7. Reply Generation

The reply generator follows a grounded-first strategy.

If a sufficiently similar historical response exists, the system includes that historical guidance in the draft.

If retrieval confidence is insufficient, the system produces a conservative intent-specific fallback response.

For example, a delivery issue receives delivery-oriented guidance rather than an invented order status.

The system also removes Twitter-specific elements such as:

* User mentions
* URLs
* Some Twitter formatting

before using historical responses as grounding.

---

# 8. Escalation Policy

The escalation layer uses deterministic rules.

Sensitive intents are currently escalated:

```text
payment_billing
account_issue
return_refund
```

Cases are also escalated when historical retrieval grounding is insufficient.

Otherwise, non-sensitive cases with sufficient historical grounding can be marked:

```text
AUTO_HANDLE
```

The purpose is to avoid automatically handling cases involving potentially sensitive account, payment, or refund information.

---

# 9. Evaluation

The evaluation uses **200 examples** from the golden dataset.

### Intent evaluation

| System                       |   Accuracy |   Macro F1 |
| ---------------------------- | ---------: | ---------: |
| Majority baseline            |     32.50% |      4.46% |
| Keyword baseline             | **44.50%** |     25.14% |
| TF-IDF + Logistic Regression |     40.50% | **30.53%** |

The machine-learning classifier does not have the highest raw accuracy, but it has the highest Macro F1 among the tested systems.

This matters because the golden set is imbalanced.

The `other` class contains **65 of 200 examples (32.5%)**, so a system that predicts `other` frequently can obtain a deceptively strong accuracy score.

---

# 10. Agent Evaluation

The complete agent evaluation produced:

```text
Golden examples:       200

Intent accuracy:       40.50%

Average confidence:    0.1372

Average retrieval:     0.4807

AUTO_HANDLE:            169
ESCALATE:                31

Escalation rate:        15.50%
```

The results are stored in:

```text
evaluation/agent_evaluation_200.csv
```

---

# 11. Offline Judge

An LLM-based judge was implemented in:

```text
evaluation/llm_judge.py
```

The intended judge evaluates:

* Relevance
* Grounding
* Helpfulness
* Safety
* Overall quality

However, the external LLM judge could not be executed because the API account used for testing had **no remaining API credits**.

An offline heuristic judge was therefore used during development.

Its development-time results were:

| Criterion   |  Average |
| ----------- | -------: |
| Relevance   | 3.18 / 5 |
| Grounding   | 3.49 / 5 |
| Helpfulness | 2.59 / 5 |
| Safety      | 5.00 / 5 |
| Overall     | 3.59 / 5 |

**These numbers are explicitly offline heuristic scores and are NOT presented as LLM-judge scores.**

---

# 12. Human Evaluation Limitation

A formal independent second-rater agreement study was not completed because of time constraints.

The 200-example golden set received AI-assisted labeling/review.

Therefore, the project does **not** claim an independently measured human-agreement score.

This is an important limitation because independently labeled evaluation data would provide stronger evidence of the quality of the intent taxonomy and agent outputs.

---

# 13. Top 5 Failure Modes

## Failure Mode 1 — Delivery messages classified as `other`

Some delivery complaints, especially multilingual or indirect messages, were classified as `other`.

Example:

> "Spanish waiting 4 months"

Expected:

```text
missing_delivery
```

Predicted:

```text
other
```

### Hypothesis

The TF-IDF classifier relies heavily on lexical overlap and struggles with short or multilingual messages.

---

## Failure Mode 2 — Shipping information vs missing delivery

Very short messages can be ambiguous.

Example:

> "Yes was supposed to be today."

Expected:

```text
shipping_information
```

Predicted:

```text
missing_delivery
```

### Hypothesis

The distinction between asking about expected delivery information and reporting a missing delivery is difficult when the message contains little context.

---

## Failure Mode 3 — Device/product questions classified as delivery issues

Example:

> "Echo Dot India services"

Expected:

```text
device_product
```

Predicted:

```text
delivery_tracking
```

The retriever also returned historically similar but irrelevant troubleshooting content.

### Hypothesis

Short product-related messages contain weak contextual signals, causing both classification and retrieval to over-weight generic words.

---

## Failure Mode 4 — Promotion vs refund/return overlap

Multi-intent messages can contain several concepts simultaneously.

For example, a message may mention:

* Promotion
* Defective product
* Replacement
* Refund

The current classifier must select only one intent.

### Hypothesis

A single-label intent taxonomy is insufficient for multi-intent customer requests.

A future version could use multi-label classification or a primary-intent-plus-secondary-intent representation.

---

## Failure Mode 5 — Short conversational messages

Messages such as:

> "Yes was supposed to be today."

contain very little information by themselves.

### Hypothesis

TF-IDF works well when useful keywords are present, but it lacks the semantic/contextual understanding required for short conversational follow-ups.

A future system could include conversation history instead of classifying each tweet independently.

---

# 14. What Is Misleading About My Headline Number?

The headline **40.5% intent accuracy** should not be interpreted as "the AI understands 40.5% of customer requests."

There are several reasons:

1. The evaluation set is imbalanced.
2. `other` represents 32.5% of the golden set.
3. Accuracy does not show performance equally across all intents.
4. Macro F1 is only **30.53%**, showing weaker performance across minority classes.
5. Retrieval quality and reply quality are separate from intent accuracy.
6. Some high-similarity retrievals can still be semantically wrong.
7. The golden-set labeling process was AI-assisted rather than independently double-annotated.

Therefore, the most useful interpretation is that this is an early baseline system whose main value is demonstrating the complete support-agent pipeline and exposing the areas requiring improvement.

---

# 15. Decision Log

## Decision 1 — Select AmazonHelp

AmazonHelp was selected because it had the largest number of response examples, providing more historical support data.

## Decision 2 — Use a small intent taxonomy

A small taxonomy was preferred over dozens of highly specific categories to make the first version easier to evaluate.

## Decision 3 — Use TF-IDF

TF-IDF provides a simple, interpretable baseline suitable for a take-home assignment and makes retrieval behavior easy to inspect.

## Decision 4 — Use Logistic Regression

Logistic Regression is fast, interpretable, and works well with sparse TF-IDF features.

## Decision 5 — Include bigrams

Bigrams help capture phrases such as:

```text
tracking number
refund request
account issue
delivery date
```

## Decision 6 — Use class balancing

The intent distribution is imbalanced, so `class_weight="balanced"` was used.

## Decision 7 — Ground replies in historical responses

Historical Amazon responses provide evidence of how similar cases were handled instead of generating completely unsupported responses.

## Decision 8 — Add a grounding threshold

Low-similarity historical matches can be misleading, so a threshold was introduced.

## Decision 9 — Escalate sensitive intents

Payment, account, and refund cases can involve sensitive information, so these are routed for human review.

## Decision 10 — Prefer conservative fallbacks

When grounding is weak, the system uses a generic response rather than inventing specific account/order information.

## Decision 11 — Keep raw data outside Git

The original Twitter dataset is hundreds of MB and should not be committed to the repository.

## Decision 12 — Evaluate against baselines

The majority and keyword baselines provide simple reference points for judging whether the learned classifier adds value.

## Decision 13 — Report Macro F1

Accuracy alone is misleading because the golden set is imbalanced.

## Decision 14 — Do not claim LLM-judge results

The API account had no remaining credits, so the offline heuristic judge is clearly labeled as such.

## Decision 15 — Explicitly report the human-evaluation limitation

Rather than inventing independent human-agreement results, the limitation is reported directly.

---

# 16. Running the Project

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Process the raw dataset:

```bash
python src/data_processing.py
```

Train the intent classifier:

```bash
python src/intent_classifier.py
```

Test retrieval:

```bash
python src/retriever.py
```

Run the support agent:

```bash
python src/agent.py
```

Evaluate the intent classifier:

```bash
python evaluation/evaluate_intents.py
```

Evaluate the baselines:

```bash
python evaluation/evaluate_baselines.py
```

Evaluate the complete agent:

```bash
python evaluation/evaluate_agent.py
```

Run the offline judge:

```bash
python evaluation/llm_judge.py
```

---

# 17. Repository Structure

```text
hiver-sde-assignment-final/
│
├── data/
│   ├── raw/
│   │   └── twcs.csv
│   │
│   ├── processed/
│   │   ├── amazon_conversations.csv
│   │   └── intent_model.joblib
│   │
│   └── golden/
│       └── amazon_golden.csv
│
├── evaluation/
│   ├── create_golden_set.py
│   ├── suggest_golden_labels.py
│   ├── prepare_golden.py
│   ├── evaluate_intents.py
│   ├── evaluate_baselines.py
│   ├── evaluate_agent.py
│   ├── llm_judge.py
│   ├── golden_template.csv
│   ├── golden_set_200.csv
│   └── agent_evaluation_200.csv
│
├── src/
│   ├── data_processing.py
│   ├── intent_classifier.py
│   ├── retriever.py
│   ├── reply_generator.py
│   ├── escalation.py
│   └── agent.py
│
├── tests/
│
├── report/
│
└── README.md
```

---

# 18. Limitations and Future Improvements

The current implementation is intentionally lightweight.

The highest-impact improvements would be:

1. Use conversation history rather than isolated tweets.
2. Replace TF-IDF classification with a stronger semantic model.
3. Improve multilingual support.
4. Use multi-label classification for multi-intent requests.
5. Add reranking to improve retrieval quality.
6. Use a stronger grounding verification step.
7. Collect an independently double-annotated golden set.
8. Run an actual independent LLM judge when API access is available.
9. Add confidence calibration.
10. Learn escalation thresholds from evaluation data instead of relying entirely on hand-written rules.

---

# 19. Conclusion

The project demonstrates an end-to-end AI support-agent pipeline:

```text
Customer message
       ↓
Intent classification
       ↓
Historical retrieval
       ↓
Grounded response generation
       ↓
Escalation decision
```

The evaluation shows that the system is functional but still has substantial room for improvement, particularly around short messages, multilingual requests, overlapping intents, and retrieval relevance.

The main value of the current system is not just the headline accuracy number, but the evaluation framework and failure analysis that identify where a production-quality support agent would need additional work.

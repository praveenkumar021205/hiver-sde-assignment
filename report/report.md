# Hiver SDE Intern Take-Home Report

## 1. Problem and brand selection
Selected **AmazonHelp** because it has the largest response volume in the supplied Customer Support on Twitter dataset, giving a broad historical resolution pool.

## 2. Data preparation
The reliable linkage is the response tweet's `in_response_to_tweet_id` pointing to the customer's `tweet_id`. We exclude tweets that are not directly linked to an AmazonHelp response.

## 3. Intent taxonomy
Initial taxonomy:
- delivery_tracking
- missing_delivery
- order_issue
- return_refund
- cancellation
- payment_billing
- prime_video
- device_product
- account_issue
- promotion
- shipping_information
- other

These labels must be refined against the hand-labelled golden set before reporting final results.

## 4. System
Customer message -> intent classifier -> historical TF-IDF retriever -> draft response -> escalation policy.

## 5. Evaluation
Use 150–250 hand-labelled examples.
Report accuracy and macro-F1. Compare against:
1. trivial majority-class baseline
2. TF-IDF + Logistic Regression baseline

For replies, manually score a subset on correctness, historical grounding, helpfulness, brand appropriateness, and unsupported claims. If an LLM judge is used, compare its labels/scores with human labels and report agreement.

## 6. Results
Fill with measured numbers only.

## 7. Top five failure modes
Fill with real examples from the evaluation set. For each:
- customer message
- predicted intent / reply
- expected behavior
- hypothesis

## 8. What is misleading about my headline number?
A single aggregate accuracy can hide poor performance on minority intents and can look better because some examples are easy or highly repetitive. Macro-F1, per-intent results, retrieval quality, and human review of replies are therefore necessary.

## 9. One more week
- expand and rebalance the golden set
- use stronger semantic embeddings
- improve multi-turn conversation reconstruction
- add policy/safety constraints
- improve LLM judge calibration
- test temporal generalization

## 10. Decision log
1. Selected AmazonHelp for response volume.
2. Used direct tweet-ID linkage rather than mention matching.
3. Started with a small interpretable intent taxonomy.
4. Used TF-IDF + Logistic Regression as the simple baseline.
5. Added historical retrieval to ground responses.
6. Added confidence thresholds for escalation.
7. Treat payment/account issues conservatively.
8. Use macro-F1 rather than accuracy alone.
9. Keep a hand-labelled golden set separate from training examples.
10. Report failure cases, not only headline scores.

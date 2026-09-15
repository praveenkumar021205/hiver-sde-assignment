def decide_escalation(intent, confidence, retrieval_score):

    # Sensitive cases should always receive human review.
    if intent in {
        "payment_billing",
        "account_issue",
        "return_refund"
    }:
        return (
            "ESCALATE",
            f"Human review recommended for sensitive {intent} case."
        )

    # No reliable historical grounding.
    if retrieval_score < 0.20:
        return (
            "ESCALATE",
            "No sufficiently similar historical resolution was found."
        )

    # Otherwise allow automated handling.
    return (
        "AUTO_HANDLE",
        "Non-sensitive intent has sufficient historical grounding "
        "for an automated draft."
    )
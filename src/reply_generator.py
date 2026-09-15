import re


GROUNDING_THRESHOLD = 0.45


def clean_historical_response(response):
    """
    Remove customer-specific information from a historical
    AmazonHelp response before using it as grounding evidence.
    """

    if not response:
        return ""

    text = str(response)

    # Remove Twitter handles
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove Twitter-style signatures such as ^KH
    text = re.sub(r'\^[A-Za-z]{1,4}\b', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def generate_reply(customer_message, intent, best_match):

    historical_response = ""
    retrieval_score = 0.0

    if best_match:

        historical_response = best_match.get(
            "response_text",
            ""
        )

        retrieval_score = float(
            best_match.get(
                "similarity",
                0.0
            )
        )

    historical_response = clean_historical_response(
        historical_response
    )

    # ---------------------------------------------------------
    # ONLY USE HISTORICAL RESPONSE WHEN MATCH IS STRONG ENOUGH
    # ---------------------------------------------------------

    strong_grounding = (
        historical_response
        and retrieval_score >= GROUNDING_THRESHOLD
    )

    # ---------------------------------------------------------
    # INTENT-SPECIFIC REPLIES
    # ---------------------------------------------------------

    fallback_replies = {

        "delivery_tracking":
            "I can help you check the latest tracking information "
            "for your Amazon order. Please check the tracking details "
            "in your order or provide the relevant order information.",

        "missing_delivery":
            "I understand that your order has not arrived as expected. "
            "Please check the latest delivery status and estimated "
            "delivery date in your Amazon order details.",

        "order_issue":
            "I understand that there is an issue with your order or "
            "the item you received. Please provide the relevant order "
            "details so the issue can be reviewed.",

        "return_refund":
            "I understand that you need help with a return or refund. "
            "Please check the return or refund status in your Amazon "
            "order details.",

        "cancellation":
            "I can help with your order cancellation request. "
            "Please provide the relevant order details so the request "
            "can be reviewed.",

        "payment_billing":
            "I understand your concern about the payment or charge. "
            "Please check the transaction details in your Amazon "
            "account. If the charge is still unclear, Amazon Support "
            "can review the transaction.",

        "account_issue":
            "I understand that you are having trouble with your "
            "Amazon account. Please follow the appropriate sign-in "
            "or account recovery steps.",

        "prime_video":
            "I understand that you are having trouble with Prime Video. "
            "Please check the app, device and account settings. "
            "If the issue continues, Amazon Support can help investigate it.",

        "device_product":
            "I understand that you are having an issue with an Amazon "
            "device or product. Please provide the product details "
            "and describe the issue so it can be reviewed.",

        "promotion":
            "I understand that you are having an issue with a promotion "
            "or offer. Please check the promotion terms and eligibility "
            "associated with the order.",

        "shipping_information":
            "I can help with shipping information. Please check the "
            "estimated delivery date and tracking details associated "
            "with your order.",

        "other":
            "I understand your concern. Please provide a few more "
            "details about the issue so it can be reviewed."
    }

    reply = fallback_replies.get(
        intent,
        fallback_replies["other"]
    )

    # ---------------------------------------------------------
    # ADD HISTORICAL GROUNDING ONLY FOR STRONG MATCHES
    # ---------------------------------------------------------

    if strong_grounding:

        reply += (
            "\n\nSimilar Amazon support guidance: "
            + historical_response
        )

    else:

        reply += (
            "\n\nNo sufficiently similar historical resolution "
            "was found, so this draft uses a general response "
            "for the predicted intent."
        )

    # ---------------------------------------------------------
    # HUMAN REVIEW FOR SENSITIVE CASES
    # ---------------------------------------------------------

    if intent in {
        "payment_billing",
        "account_issue",
        "return_refund"
    }:

        reply += (
            "\n\nHuman review is recommended before sending "
            "a final response because this case may involve "
            "account, payment, or refund information."
        )

    return reply


# -------------------------------------------------------------
# SIMPLE LOCAL TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    example_match = {
        "response_text":
            "@123456 I'm sorry about the unexpected charge. "
            "We can't access your account from Twitter. "
            "https://t.co/example ^KH",

        "similarity": 0.56
    }

    result = generate_reply(
        customer_message="Why was I charged twice?",
        intent="payment_billing",
        best_match=example_match
    )

    print("=" * 60)
    print("REPLY GENERATOR TEST")
    print("=" * 60)
    print(result)
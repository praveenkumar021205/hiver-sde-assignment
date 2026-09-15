from src.escalation import decide

def test_low_confidence_escalates():
    d, _ = decide("other", 0.2, 0.8)
    assert d == "ESCALATE"

def test_weak_retrieval_escalates():
    d, _ = decide("delivery_tracking", 0.9, 0.05)
    assert d == "ESCALATE"

def test_strong_case_auto_handles():
    d, _ = decide("delivery_tracking", 0.9, 0.5)
    assert d == "AUTO_HANDLE"

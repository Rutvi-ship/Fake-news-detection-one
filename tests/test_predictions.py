# tests/test_predictions.py
def test_real_like_has_higher_real_probability(artifacts):
    vectorizer, model = artifacts

    fake_like = "SHOCKING! Secret plot exposed!!! Click now to see the truth nobody wants you to know"
    real_like = "WASHINGTON (Reuters) - Officials announced a policy update regarding trade negotiations on Tuesday."

    X_fake = vectorizer.transform([fake_like])
    X_real = vectorizer.transform([real_like])

    p_real_fake_like = model.predict_proba(X_fake)[0, 1]  # P(class=1: real)
    p_real_real_like = model.predict_proba(X_real)[0, 1]

    assert p_real_real_like > p_real_fake_like

def test_threshold_decision(artifacts):
    vectorizer, model = artifacts
    text = "WASHINGTON (Reuters) - The central bank released its quarterly economic outlook."
    X = vectorizer.transform([text])
    proba_real = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]  # 0=fake, 1=real in sklearn with default 0.5 threshold
    # If proba_real >= 0.5 -> pred should be 1 (real); else 0 (fake)
    assert pred == (1 if proba_real >= 0.5 else 0)

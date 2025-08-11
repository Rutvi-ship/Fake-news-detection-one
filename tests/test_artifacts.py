# tests/test_artifacts.py
import numpy as np

def test_classes_order(artifacts):
    _, model = artifacts
    # Expect binary labels [0, 1] => 0=fake, 1=real
    assert list(model.classes_) == [0, 1]

def test_predict_proba_shape_and_bounds(artifacts):
    vectorizer, model = artifacts
    X = vectorizer.transform(["short sample text"])
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    # probabilities within [0,1] and rows sum to ~1
    assert np.all((proba >= 0) & (proba <= 1))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)

def test_batch_prediction_consistency(artifacts):
    vectorizer, model = artifacts
    texts = ["one text about government",
             "breaking shocking click now!!!",
             "economy and policy discussion in washington"]
    X = vectorizer.transform(texts)
    proba = model.predict_proba(X)
    assert proba.shape == (len(texts), 2)

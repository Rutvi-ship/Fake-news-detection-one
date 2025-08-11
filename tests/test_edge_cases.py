# tests/test_edge_cases.py
import numpy as np
import scipy.sparse as sp
import pytest

def test_empty_string_does_not_crash(artifacts):
    vectorizer, model = artifacts
    X = vectorizer.transform([""])
    assert isinstance(X, (sp.csr_matrix, sp.csc_matrix))
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)
    assert np.isfinite(proba).all()

def test_unicode_and_punctuation(artifacts):
    vectorizer, model = artifacts
    text = "Café prices ↑↑ after policy—analysis shows mixed results… 🇨🇦"
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)

def test_long_text(artifacts):
    vectorizer, model = artifacts
    long_text = "Washington report: " + ("economy policy market trade " * 500)
    X = vectorizer.transform([long_text])
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2)

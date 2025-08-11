# tests/conftest.py
import os
import joblib
import pytest

@pytest.fixture(scope="session")
def artifacts():
    vec_path = "vectorizer.joblib"
    mdl_path = "lr_model.joblib"
    assert os.path.exists(vec_path), f"Missing {vec_path}"
    assert os.path.exists(mdl_path), f"Missing {mdl_path}"
    vectorizer = joblib.load(vec_path)
    model = joblib.load(mdl_path)
    return vectorizer, model

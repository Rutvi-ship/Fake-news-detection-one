
import numpy as np
from scipy.sparse import csr_matrix

import streamlit as st
import joblib

#Loads the TF-IDF vectorizer 
#Loads the Logistic Regression model 

vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("lr_model.joblib")
 
def explain_single_doc(text, top_k=10):
    """
    Returns:
      proba_real, proba_fake, contributions (list of tuples: token, push_to_real)
    Notes:
      - push_to_real > 0 means the token pushes prediction toward 'Real'
      - push_to_real < 0 means it pushes toward 'Fake'
    """
    X = vectorizer.transform([text])
    if not isinstance(X, csr_matrix):
        X = csr_matrix(X)

    # probabilities (LogisticRegression supports predict_proba)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_fake = float(proba[0])
        proba_real = float(proba[1])
    else:
        # fallback: decision_function -> sigmoid
        z = float(model.decision_function(X)[0])
        proba_real = 1.0 / (1.0 + np.exp(-z))
        proba_fake = 1.0 - proba_real

    # token contributions toward class "Real" (coef for class 1)
    coef = model.coef_[0]                  # shape (n_features,)
    feat_names = vectorizer.get_feature_names_out()

    contribs = []
    for val, j in zip(X.data, X.indices):
        token = feat_names[j]
        push_real = float(val * coef[j])   # tf-idf value * weight
        contribs.append((token, push_real))

    # sort by absolute impact and keep top_k
    contribs = sorted(contribs, key=lambda t: abs(t[1]), reverse=True)[:top_k]
    return proba_real, proba_fake, contribs


#Displays a large title on the web app
st.title("Fake News Detector")

#Adds some instructional text 
st.write("Enter a News Article below to check whether it is Fake or Real. ")

#cteate multiline text box
inputn = st.text_area("News Article:","")

if st.button("Check News"):
    if inputn.strip():
        # --- credibility + transparency ---
        proba_real, proba_fake, contribs = explain_single_doc(inputn, top_k=10)

        # predicted label using default 0.5 threshold
        is_real = proba_real >= 0.5
        pred_label = "Real" if is_real else "Fake"

        # credibility score = probability of predicted class
        credibility = proba_real if is_real else proba_fake
        cred_pct = int(round(credibility * 100))

        # Show main result with credibility
        if is_real:
            st.success(f"The News is **Real**  •  Credibility: **{cred_pct}%**")
        else:
            st.error(f"The News is **Fake**  •  Credibility: **{cred_pct}%**")

        # Show raw probabilities for transparency
        st.write(f"**P(Real):** {proba_real:.2f}   |   **P(Fake):** {proba_fake:.2f}")

        # Simple transparency: just words
        if contribs:
            st.subheader("Transparency: Top Influential Tokens")
            pos = [f" {t}" for (t, w) in contribs if w > 0]
            neg = [f" {t}" for (t, w) in contribs if w < 0]

            if pos:
                st.markdown("**Push toward Real:**")
                st.write(", ".join(pos))
            if neg:
                st.markdown("**Push toward Fake:**")
                st.write(", ".join(neg))
        else:
            st.info("No informative tokens found (try a longer text).")

    else:
        st.warning("Please enter some text to analyze.")


import streamlit as st
import joblib
import numpy as np
from feature_extractor import extract_lexical_features, FEATURE_ORDER, vectorize

st.set_page_config(page_title="Live Phishing URL Detector", page_icon="🛡️")

st.title("🛡️ Live Phishing URL Detector")
st.write("Enter a URL. The model extracts lexical features and predicts if it's **Phishing** or **Legitimate**.")

@st.cache_resource
def load_model():
    try:
        bundle = joblib.load("artifacts/model.pkl")
        pipe = bundle["pipeline"]
        order = bundle["feature_order"]
        if order != FEATURE_ORDER:
            st.warning("Model feature order differs from code. Using model's order.")
            return pipe, order
        return pipe, FEATURE_ORDER
    except Exception:
        return None, FEATURE_ORDER

pipe, order = load_model()

url = st.text_input("URL", placeholder="e.g. https://secure-login.example.com/verify")
go = st.button("Check")

def rule_based_score(feats: dict) -> float:
    # A simple fallback if no ML model exists. Returns probability-ish score [0,1].
    score = 0.0
    score += 0.15 if feats["uses_shortener"] else 0.0
    score += 0.15 if feats["has_ip_in_domain"] else 0.0
    score += 0.10 if feats["suspicious_tld"] else 0.0
    score += 0.10 if feats["contains_login_keywords"] else 0.0
    score += min(0.10, 0.02 * max(0, feats["num_dots"] - 3))
    score += min(0.10, 0.02 * max(0, feats["num_hyphens"] - 2))
    score += min(0.10, 0.01 * max(0, feats["url_length"] - 75))
    score += min(0.10, 0.02 * max(0, feats["num_parameters"] - 2))
    return max(0.0, min(0.99, score))

if go and url.strip():
    feats = extract_lexical_features(url.strip())
    X = np.array([vectorize(feats)])

    st.subheader("Extracted Features")
    with st.expander("Show features"):
        st.json({k: float(v) if isinstance(v, (int, float)) else v for k, v in feats.items()})

    if pipe is not None:
        # reorder if needed
        if order != FEATURE_ORDER:
            # align: create a mapping from name->index in model order
            name_to_idx = {name:i for i,name in enumerate(FEATURE_ORDER)}
            X_ordered = np.array([[X[0][name_to_idx[name]] for name in order]])
        else:
            X_ordered = X

        proba = float(pipe.predict_proba(X_ordered)[0,1])
        pred = "Phishing" if proba >= 0.5 else "Legitimate"
        st.metric("Prediction", pred, delta=f"Phish Prob: {proba:.2%}")
    else:
        # fallback simple rules
        proba = rule_based_score(feats)
        pred = "Phishing" if proba >= 0.5 else "Legitimate"
        st.info("No trained model found at artifacts/model.pkl – using simple rule-based fallback.")
        st.metric("Prediction (Rule-Based)", pred, delta=f"Phish Score: {proba:.2%}")

    st.caption("Note: This tool uses **lexical** features only (from the text of the URL). For production, add SSL/WHOIS/page-content checks.")
else:
    st.write("👆 Paste a URL and click **Check**.")

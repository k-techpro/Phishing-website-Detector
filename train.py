
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from feature_extractor import extract_lexical_features, FEATURE_ORDER, vectorize

import argparse
import os
import json
from tqdm import tqdm

def load_dataset(csv_path: str):
    """
    Expecting a CSV with columns:
      - url: string
      - label: 1 for phishing, 0 for legitimate
    """
    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns.")
    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        feats = extract_lexical_features(str(row["url"]))
        rows.append(vectorize(feats))
        y.append(int(row["label"]))
    X = pd.DataFrame(rows, columns=FEATURE_ORDER)
    y = pd.Series(y, name="label")
    return X, y

def main(args):
    df = load_dataset(args.csv)
    X, y = build_feature_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=None))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred, digits=4))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception as e:
        print("AUC failed:", e)

    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "model.pkl")
    joblib.dump({
        "pipeline": pipe,
        "feature_order": FEATURE_ORDER
    }, model_path)
    print("Saved:", model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to training CSV with columns url,label")
    parser.add_argument("--outdir", default="artifacts", help="Where to save model.pkl")
    args = parser.parse_args()
    main(args)

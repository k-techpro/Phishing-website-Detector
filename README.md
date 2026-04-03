
# Live Phishing URL Detector (Lexical)

A minimal end-to-end project for detecting phishing URLs from user input. It provides:
- A **feature extractor** for URL lexical features
- A **training script** to fit a Logistic Regression model
- A **Streamlit** app for live predictions

## 1) Install
```bash
pip install -r requirements.txt
```

## 2) Train a model (optional but recommended)
Prepare a CSV with columns:
- `url` (string)
- `label` (1 = phishing, 0 = legitimate)

Example:
```csv
url,label
https://secure-login.example.com/verify,1
https://www.wikipedia.org,0
```
Then run:
```bash
python train.py --csv your_dataset.csv --outdir artifacts
```
This saves `artifacts/model.pkl`

## 3) Run the live app
```bash
streamlit run app.py
```
Open the local URL shown in the terminal and enter any URL to check.

## Notes
- The app uses only **lexical** features. You can extend it with SSL, WHOIS age, and page-content features.
- If `artifacts/model.pkl` is missing, the app falls back to a simple **rule-based** scorer so it still works live.
- To avoid the common "feature mismatch" error, both training and inference use the **same `FEATURE_ORDER`**.

- Created by **kombaiya** **k-techpro**

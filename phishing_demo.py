# phishing_demo.py
import os, re, socket, datetime, ipaddress
from urllib.parse import urlparse, urljoin

import pandas as pd
import joblib
import requests
import tldextract
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    import whois
except Exception:
    whois = None

DATA_FILE = "phishing.csv"
MODEL_FILE = "phishing_model.pkl"

# -------------------------
# Training function
# -------------------------
def train_and_save_model():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Put the CSV in this folder.")
        return None, None

    df = pd.read_csv(DATA_FILE)
    if "Result" not in df.columns:
        print("ERROR: dataset must have a 'Result' column (1 safe, -1 phishing).")
        return None, None

    X = df.drop("Result", axis=1)
    y = df["Result"]

    print("Training RandomForest (this may take a minute)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Trained. Test Accuracy: {acc*100:.2f}%")

    joblib.dump((model, X.columns.tolist()), MODEL_FILE)
    print(f"Saved model + feature order to {MODEL_FILE}")
    return model, X.columns.tolist()

# -------------------------
# Heuristic feature extractor
# -------------------------
def simple_feature_values(url):
    # Add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    host = parsed.netloc.split(':')[0].lower()
    full = url
    now = datetime.datetime.now()

    # 1. IP address in host?
    has_ip = -1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', host) else 1

    # 2. URL length
    if len(full) < 54:
        url_len = 1
    elif len(full) <= 75:
        url_len = 0
    else:
        url_len = -1

    # 3. '@' symbol
    has_at = -1 if '@' in full else 1

    # 4. '//' beyond protocol
    double_slash = -1 if full.count('//') > 1 else 1

    # 5. '-' in domain (prefix/suffix trick)
    prefix_suffix = -1 if '-' in host else 1

    # 6. Subdomain count
    dots = host.count('.')
    if dots <= 1:
        subdomain = 1
    elif dots == 2:
        subdomain = 0
    else:
        subdomain = -1

    # 7. HTTPS presence
    https_flag = 1 if parsed.scheme == 'https' else -1

    # 8. Shortening service
    short_pattern = r"bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co|tiny\.cc|is\.gd|cutt\.ly|tinyurl\.com|adf\.ly"
    short_service = -1 if re.search(short_pattern, full, re.I) else 1

    # 9. Domain age via WHOIS (if available)
    domain_age = -1
    if whois is not None:
        try:
            w = whois.whois(host)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation and isinstance(creation, datetime.datetime):
                age_days = (now - creation).days
                domain_age = 1 if age_days > 365 else -1
        except Exception:
            domain_age = -1

    # 10. DNS record existence
    try:
        socket.gethostbyname(host)
        dns_record = 1
    except Exception:
        dns_record = -1

    # 11..n : quick HTML checks (iframe, onmouseover, rightclick, popup, external anchors, forms)
    html = ""
    try:
        resp = requests.get(full, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        html = resp.text or ""
    except Exception:
        html = ""

    soup = BeautifulSoup(html, "html.parser") if html else None

    iframe_flag = -1 if (soup and soup.find("iframe")) else 1
    onmouseover_flag = -1 if re.search(r'onmouseover', html, re.I) else 1
    rightclick_flag = -1 if re.search(r'contextmenu|event.button', html, re.I) else 1
    popup_flag = -1 if re.search(r'window.open\(|popup', html, re.I) else 1

    # favicon domain mismatch
    favicon_flag = 1
    try:
        if soup:
            link = soup.find("link", rel=lambda r: r and "icon" in r.lower())
            if link and link.get("href"):
                fav_url = urljoin(full, link.get("href"))
                fav_host = urlparse(fav_url).netloc.split(':')[0]
                if fav_host and tldextract.extract(fav_host).registered_domain != tldextract.extract(host).registered_domain:
                    favicon_flag = -1
    except Exception:
        favicon_flag = 1

    # anchors external ratio
    anchor_flag = 1
    try:
        if soup:
            anchors = [a for a in soup.find_all("a", href=True) if a.get("href")]
            if anchors:
                external = 0
                for a in anchors:
                    href = a['href']
                    # ignore mailto / javascript
                    if href.startswith(("http://", "https://")):
                        a_host = urlparse(href).netloc.split(':')[0]
                        if tldextract.extract(a_host).registered_domain != tldextract.extract(host).registered_domain:
                            external += 1
                ratio = external / len(anchors)
                anchor_flag = -1 if ratio > 0.5 else 1
    except Exception:
        anchor_flag = 1

    # forms action pointing to external domain
    form_flag = 1
    try:
        if soup:
            forms = soup.find_all("form", action=True)
            for f in forms:
                act = f.get("action")
                if act and act.startswith(("http://", "https://")):
                    act_host = urlparse(act).netloc.split(':')[0]
                    if tldextract.extract(act_host).registered_domain != tldextract.extract(host).registered_domain:
                        form_flag = -1
                        break
    except Exception:
        form_flag = 1

    # port (explicit non-default port is suspicious)
    port_flag = 1
    if ':' in parsed.netloc:
        try:
            p = int(parsed.netloc.split(':')[-1])
            if p not in (80, 443):
                port_flag = -1
        except:
            port_flag = -1

    # Bundle computed features dictionary
    feats = {
        "having_ip": has_ip,
        "url_length": url_len,
        "shortening_service": short_service,
        "having_at_symbol": has_at,
        "double_slash_redirecting": double_slash,
        "prefix_suffix": prefix_suffix,
        "having_sub_domain": subdomain,
        "ssl_final_state": https_flag,
        "domain_registration_length": domain_age,
        "dns_record": dns_record,
        "favicon": favicon_flag,
        "links_in_tags": anchor_flag,
        "sfh": form_flag,
        "iframe": iframe_flag,
        "on_mouseover": onmouseover_flag,
        "right_click": rightclick_flag,
        "pop_up_window": popup_flag,
        "port": port_flag
    }
    return feats

# Map heuristics to dataset columns
def build_feature_vector(url, columns):
    heuristic = simple_feature_values(url)
    vector = []
    for col in columns:
        key = col.lower().replace(" ", "_")
        v = 0
        # lots of partial matches to cover common dataset column names
        if "ip" in key and "age" not in key:
            v = heuristic.get("having_ip", 0)
        elif "length" in key and "domain" not in key:
            v = heuristic.get("url_length", 0)
        elif "short" in key:
            v = heuristic.get("shortening_service", 0)
        elif "@" in key or "at" in key:
            v = heuristic.get("having_at_symbol", 0)
        elif "double" in key or "redirect" in key:
            v = heuristic.get("double_slash_redirecting", 0)
        elif "-" in key or "prefix" in key or "suffix" in key:
            v = heuristic.get("prefix_suffix", 0)
        elif "sub" in key and "domain" in key:
            v = heuristic.get("having_sub_domain", 0)
        elif "ssl" in key or "https" in key or "final_state" in key:
            v = heuristic.get("ssl_final_state", 0)
        elif "domain" in key and ("reg" in key or "age" in key or "registration" in key):
            v = heuristic.get("domain_registration_length", 0)
        elif "dns" in key:
            v = heuristic.get("dns_record", 0)
        elif "favicon" in key:
            v = heuristic.get("favicon", 0)
        elif "anchor" in key or "links" in key or "tags" in key:
            v = heuristic.get("links_in_tags", 0)
        elif "sfh" in key or "form" in key:
            v = heuristic.get("sfh", 0)
        elif "iframe" in key:
            v = heuristic.get("iframe", 0)
        elif "mouse" in key:
            v = heuristic.get("on_mouseover", 0)
        elif "right" in key:
            v = heuristic.get("right_click", 0)
        elif "pop" in key:
            v = heuristic.get("pop_up_window", 0)
        elif "port" in key:
            v = heuristic.get("port", 0)
        else:
            # default fallback (dataset had many features; unknown ones set to 0)
            v = 0
        vector.append(int(v))
    return vector

# -------------------------
# Main: train if needed, then interactive predict
# -------------------------
def main():
    if os.path.exists(MODEL_FILE):
        try:
            model, columns = joblib.load(MODEL_FILE)
            print(f"Loaded model from {MODEL_FILE}")
        except Exception:
            print("Failed to load model file. Re-training.")
            model, columns = train_and_save_model()
    else:
        model, columns = train_and_save_model()

    if model is None or columns is None:
        print("Model not available. Exiting.")
        return

    print("\nREADY. Paste a URL to check (type 'exit' to quit).")
    while True:
        url = input("URL> ").strip()
        if not url:
            continue
        if url.lower() in ("exit", "quit"):
            break
        try:
            fv = build_feature_vector(url, columns)
            pred = model.predict([fv])[0]
            label = "✅ SAFE" if pred == 1 else "🚨 PHISHING"
            print(f"-> Prediction: {label}\n")
        except Exception as e:
            print("Error during prediction:", e)
            print("Try a simpler URL or ensure you have internet access for WHOIS/HTML fetch.")
            continue

if __name__ == "__main__":
    main()
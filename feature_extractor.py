
import re
import math
from urllib.parse import urlparse, parse_qs

SHORTENERS = {
    "bit.ly","goo.gl","tinyurl.com","ow.ly","t.co","is.gd","buff.ly","adf.ly","shorturl.at",
    "cutt.ly","tiny.cc","rb.gy","s.id","v.gd","shrtco.de","rebrand.ly","lnkd.in","surl.li"
}

SUSPICIOUS_TLDS = {
    "tk","ml","ga","cf","gq","xyz","top","work","support","zip","cricket","click","country",
    "link","fit","rest","biz","info","pw","party"
}

LOGIN_KEYWORDS = {"login","signin","verify","update","secure","account","bank","webscr","confirm"}

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    counts = Counter(s)
    total = len(s)
    ent = 0.0
    for c in counts.values():
        p = c/total
        ent -= p * math.log2(p)
    return ent

def has_ip_address(host: str) -> bool:
    # IPv4
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host or ""):
        return True
    # IPv6 (very loose)
    if ":" in (host or "") and len(host.split(":")) >= 3:
        return True
    return False

def count_special(s: str, chars: str) -> int:
    return sum(s.count(ch) for ch in chars)

def get_tld(host: str) -> str:
    # naive TLD extraction
    if not host:
        return ""
    parts = host.lower().split(".")
    return parts[-1] if parts else ""

def uses_shortener(host: str) -> bool:
    return (host or "").lower() in SHORTENERS

def extract_lexical_features(url: str) -> dict:
    # robust parse
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except Exception:
        parsed = urlparse("http://" + (url or ""))

    scheme = parsed.scheme or ""
    host = (parsed.netloc or parsed.path or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    # Fix for cases like "example.com" ending up in path
    if host and "." not in host and "/" in path:
        # keep as-is
        pass

    # feature calculations
    url_str = f"{scheme}://{host}{path}"
    url_len = len(url)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)
    num_params = len(parse_qs(query))
    num_digits = sum(ch.isdigit() for ch in url)
    num_dots = url.count(".")
    num_hyphens = url.count("-")
    num_at = url.count("@")
    num_subdirs = path.count("/")
    has_https = 1 if scheme.lower() == "https" else 0
    has_at = 1 if "@" in url else 0
    has_ip = 1 if has_ip_address(host) else 0
    tld = get_tld(host)
    suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
    shortener = 1 if uses_shortener(host) else 0
    has_port = 1 if ":" in host and not host.endswith(":") else 0
    contains_login_kw = 1 if any(k in url.lower() for k in LOGIN_KEYWORDS) else 0
    digit_ratio = num_digits / max(1, url_len)
    host_entropy = shannon_entropy(host)
    path_entropy = shannon_entropy(path)

    features = {
        "url_length": url_len,
        "hostname_length": host_len,
        "path_length": path_len,
        "query_length": query_len,
        "num_parameters": num_params,
        "num_digits": num_digits,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_at": num_at,
        "num_subdirs": num_subdirs,
        "has_https": has_https,
        "has_at": has_at,
        "has_ip_in_domain": has_ip,
        "suspicious_tld": suspicious_tld,
        "uses_shortener": shortener,
        "has_port_in_host": has_port,
        "contains_login_keywords": contains_login_kw,
        "digit_ratio": digit_ratio,
        "hostname_entropy": host_entropy,
        "path_entropy": path_entropy,
    }

    return features

# Ordered list used by both training and inference to avoid mismatch
FEATURE_ORDER = [
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "num_parameters",
    "num_digits",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_subdirs",
    "has_https",
    "has_at",
    "has_ip_in_domain",
    "suspicious_tld",
    "uses_shortener",
    "has_port_in_host",
    "contains_login_keywords",
    "digit_ratio",
    "hostname_entropy",
    "path_entropy",
]

def vectorize(features: dict):
    # returns list in the agreed feature order
    return [features[name] for name in FEATURE_ORDER]

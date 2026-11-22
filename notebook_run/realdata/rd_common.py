"""
Shared helpers for the Part D real-dataset runners (Table VII of the manuscript).

Every runner follows the same rules:
  * it reads only files you have downloaded yourself under the dataset's own licence
    (PhysioNet credentialed access for MIMIC-CXR, RadGraph, VinDr-CXR, REFLACX and
    CorBenchX; Stanford AIMI registration for CheXpert);
  * it finds files by their official names anywhere under the directory you give it,
    prints the columns it found before using them, and stops with a clear message if
    a required file or column is missing;
  * it never treats a labeler output as ground truth for the cause of a disagreement:
    no dataset here labels that cause, so every check is a face-validity or
    assumption check, stated as such in the output.
"""
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ files
def find_file(root, *names):
    """First file under root whose basename matches one of names (case-insensitive;
    a name ending in .csv also matches .csv.gz)."""
    want = set()
    for n in names:
        want.add(n.lower())
        if n.lower().endswith(".csv"):
            want.add(n.lower() + ".gz")
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower() in want:
                return os.path.join(dp, fn)
    return None


def find_files(root, pattern):
    rx = re.compile(pattern, re.I)
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if rx.search(fn):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def read_table(path, **kw):
    df = pd.read_csv(path, **kw)
    print(f"  {os.path.basename(path)}: {len(df)} rows; columns = {list(df.columns)}")
    return df


def require(path, what):
    if path is None:
        sys.exit(f"{what} not found. Check the directory you set in the configuration cell.")
    return path


def col(df, *cands, required=True):
    """Column of df matching one of cands (case-insensitive, ignoring _ and spaces)."""
    norm = lambda s: re.sub(r"[\s_]+", "", str(s).lower())
    m = {norm(c): c for c in df.columns}
    for c in cands:
        if norm(c) in m:
            return m[norm(c)]
    if required:
        sys.exit(f"none of the columns {cands} found; columns are {list(df.columns)}")
    return None


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("wrote", path)


def save_md(lines, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)


def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.to_dict("records"):
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                 else ("--" if isinstance(v, (float, np.floating)) else str(v)) for v in r.values()]
        out.append("| " + " | ".join(cells) + " |")
    return out


# ------------------------------------------------------------------ report text
SECTION_RX = re.compile(r"^\s*(FINDINGS|IMPRESSION|CONCLUSION)\s*:", re.I | re.M)


def report_sections(text):
    """Findings + impression of a MIMIC-style free-text report (falls back to the whole text)."""
    text = "" if text is None else str(text)
    parts = {}
    hits = list(re.finditer(r"^\s*([A-Z][A-Z /&]+):", text, re.M))
    for i, h in enumerate(hits):
        name = h.group(1).strip().upper()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(text)
        parts[name] = text[h.end():end]
    body = " ".join(parts.get(k, "") for k in ("FINDINGS", "IMPRESSION", "CONCLUSION")).strip()
    if not body:
        body = text
    body = re.sub(r"_{2,}", " ", body)           # MIMIC de-identification placeholders
    return re.sub(r"\s+", " ", body).strip()


# Hedge cues for real reports (the synthetic harness uses its own template cues).
HEDGE_CUES = [
    r"\bpossibl[ey]\b", r"\bprobabl[ey]\b", r"\blikely\b", r"\bmay\b", r"\bmight\b", r"\bcould\b",
    r"\bcannot (?:be )?(?:excluded|exclude|ruled out|rule out)\b", r"\bnot (?:be )?excluded\b",
    r"\bsuggest(?:s|ive|ing)?\b", r"\bsuspicious\b", r"\bquestionabl[ey]\b", r"\bequivocal\b",
    r"\bindeterminate\b", r"\bconcerning for\b", r"\bversus\b", r"\bvs\.?\b", r"\bpresumabl[ey]\b",
    r"\bdifficult to (?:exclude|assess)\b", r"\bpotential(?:ly)?\b", r"\bperhaps\b",
]
HEDGE_RX = re.compile("|".join(HEDGE_CUES), re.I)
NEG_RX = re.compile(r"\b(no|without|negative for|free of|absence of|resolved|not seen|no evidence of)\b", re.I)

FINDING_TERMS = {
    "Atelectasis": ["atelecta"],
    "Cardiomegaly": ["cardiomegaly", "enlarged cardiac silhouette", "heart size is enlarged",
                     "enlarged heart", "cardiac enlargement"],
    "Consolidation": ["consolidat"],
    "Edema": ["edema", "vascular congestion"],
    "Pleural Effusion": ["effusion"],
    "Pneumothorax": ["pneumothora"],
    "Pneumonia": ["pneumonia"],
    "Lung Opacity": ["opacit", "airspace disease", "infiltrat"],
    "Lung Lesion": ["nodul", "mass", "lesion"],
    "Fracture": ["fracture"],
}
ANY_FINDING = sorted({t for v in FINDING_TERMS.values() for t in v})


def sentences(text):
    return [s for s in re.split(r"(?<=[.;])\s+|\n", str(text)) if s.strip()]


def lexical_v1(text, terms=None):
    """(hedge, mention) for one report, with the paper's View 1 definitions on real text:
    hedge   = 1 if a sentence that names a finding also carries a hedge cue;
    mention = 1 if a sentence names a finding without a negation cue (asserted present)."""
    terms = terms or ANY_FINDING
    hedge = mention = 0
    for s in sentences(text):
        low = s.lower()
        if not any(t in low for t in terms):
            continue
        if HEDGE_RX.search(low):
            hedge = 1
        elif not NEG_RX.search(low):
            mention = 1
    return float(hedge), float(mention)


# ------------------------------------------------------------------ images
def laplacian_var(img):
    lap = np.abs(np.gradient(np.gradient(img, axis=0), axis=0)
                 + np.gradient(np.gradient(img, axis=1), axis=1))
    return float(lap.var())


def load_gray(path, size=32):
    """Grayscale image in [0,1], resized to size x size. JPG/PNG via PIL, DICOM via pydicom."""
    from PIL import Image
    if path.lower().endswith((".dcm", ".dicom")):
        try:
            import pydicom
        except ImportError:
            sys.exit("pydicom is needed for DICOM images: pip install pydicom")
        ds = pydicom.dcmread(path)
        a = ds.pixel_array.astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
        a = (a - a.min()) / (a.max() - a.min() + 1e-8)
        im = Image.fromarray((a * 255).astype(np.uint8))
    else:
        im = Image.open(path).convert("L")
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def index_images(root, exts=(".jpg", ".jpeg", ".png", ".dcm", ".dicom")):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                idx.setdefault(os.path.splitext(fn)[0], os.path.join(dp, fn))
    return idx


# ------------------------------------------------------------------ statistics
def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return float("nan")
    return float(roc_auc_score(y[ok], s[ok]))


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    a = np.asarray(a); b = np.asarray(b)
    if len(np.unique(np.concatenate([a, b]))) < 2:
        return float("nan")
    return float(cohen_kappa_score(a, b))


def fleiss_kappa(counts):
    """counts: (n_items, n_categories) array of rater counts per item (same raters per item)."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(1)
    counts, n = counts[n >= 2], n[n >= 2]
    if len(counts) == 0:
        return float("nan")
    p_j = counts.sum(0) / counts.sum()
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P_i.mean(), (p_j ** 2).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def logit_or(y, X, names):
    """Odds ratios (95% Wald CI) from an unpenalized logistic regression; statsmodels if present."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    try:
        import statsmodels.api as sm
        m = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0)
        b, se, p = m.params[1:], m.bse[1:], m.pvalues[1:]
    except Exception:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1e6, max_iter=2000).fit(X, y)
        b = m.coef_[0]
        pr = m.predict_proba(X)[:, 1]
        Xd = np.hstack([np.ones((len(X), 1)), X])
        cov = np.linalg.pinv(Xd.T @ (Xd * (pr * (1 - pr))[:, None]))
        se = np.sqrt(np.diag(cov))[1:]
        from scipy.stats import norm
        p = 2 * norm.sf(np.abs(b / se))
    return pd.DataFrame({"term": names, "odds_ratio": np.exp(b), "ci_low": np.exp(b - 1.96 * se),
                         "ci_high": np.exp(b + 1.96 * se), "p": p})


def dawid_skene(L, n_iter=50, n_classes=2, smooth=1.0):
    """Dawid-Skene EM. L: (n_items, n_readers) int labels in {0..C-1}, -1 = not read.
    Returns posterior (n_items, C), class prior, confusion (n_readers, C, C)."""
    L = np.asarray(L)
    n, R = L.shape
    C = n_classes
    T = np.zeros((n, C))                      # init: majority vote
    for c in range(C):
        T[:, c] = (L == c).sum(1)
    T = (T + 1e-3) / (T + 1e-3).sum(1, keepdims=True)
    for _ in range(n_iter):
        prior = T.mean(0)
        conf = np.full((R, C, C), smooth)
        for r in range(R):
            m = L[:, r] >= 0
            for l in range(C):
                conf[r, :, l] += T[m][L[m, r] == l].sum(0)
        conf /= conf.sum(2, keepdims=True)
        logT = np.tile(np.log(prior + 1e-12), (n, 1))
        for r in range(R):
            m = L[:, r] >= 0
            logT[m] += np.log(conf[r][:, L[m, r]].T + 1e-12)
        logT -= logT.max(1, keepdims=True)
        T = np.exp(logT); T /= T.sum(1, keepdims=True)
    return T, prior, conf


# ------------------------------------------------------------------ CheXpert-style labels from text
# Used when the CheXbert model is not available.
CHEXPERT14 = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
              "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices", "No Finding"]
TERMS_EN = dict(FINDING_TERMS, **{
    "Enlarged Cardiomediastinum": ["mediastinal widening", "widened mediastinum", "enlarged cardiomediastinal",
                                   "mediastinum is widened", "cardiomediastinal silhouette is enlarged"],
    "Pleural Other": ["pleural thickening", "pleural plaque", "fibrothorax"],
    "Support Devices": ["tube", "catheter", "line", "pacemaker", "wire", "stent", "device", "port"],
})

def finding_status(text, terms):
    """CheXpert-style status of one finding in a report: 1 present, -1 uncertain, 0 negated, nan unmentioned.
    Aggregation as in the CheXpert labeler: present beats uncertain beats negated."""
    st = set()
    for s in sentences(text):
        low = s.lower()
        if not any(t in low for t in terms):
            continue
        if HEDGE_RX.search(low):
            st.add(-1)
        elif NEG_RX.search(low):
            st.add(0)
        else:
            st.add(1)
    if 1 in st:
        return 1.0
    if -1 in st:
        return -1.0
    if 0 in st:
        return 0.0
    return float("nan")


def lexical_labels(text):
    out = {f: finding_status(text, TERMS_EN[f]) for f in CHEXPERT14 if f in TERMS_EN}
    pos = [v for k, v in out.items() if k != "Support Devices" and v == 1]
    out["No Finding"] = float("nan") if pos else 1.0
    return out


def radgraph_status(label):
    """Map RadGraph (OBS-DP / OBS-U / OBS-DA) and RadGraph-XL ('Observation::definitely present', ...)
    labels to one of OBS-DP, OBS-U, OBS-DA, or None for anatomy."""
    l = str(label)
    if l.startswith("OBS-"):
        return l[:6] if l[:6] in ("OBS-DP", "OBS-DA") else ("OBS-U" if l.startswith("OBS-U") else None)
    if l.lower().startswith("observation"):
        if "uncertain" in l:
            return "OBS-U"
        if "absent" in l:
            return "OBS-DA"
        if "present" in l:
            return "OBS-DP"
    return None

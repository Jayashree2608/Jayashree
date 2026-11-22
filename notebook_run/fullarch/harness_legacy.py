"""
Reusable, parameterized version of the Part-A semi-synthetic identifiability
harness from chapter1inconsistencyrealdatakagglemimic.ipynb, refactored so it
can be run many times under different configurations (seed, view subset,
reader count, patient count, quality-confound strength, cross-fitting) for an
exhaustive ablation / sensitivity study.

Every default matches the original notebook exactly (same generative process,
same feature definitions, same EM procedure) so single-seed default-config
output here reproduces the numbers already reported in the manuscript.
"""
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              precision_recall_fscore_support)
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["faithful", "omission", "hedge", "style"]
K_CLASSES = len(CLASS_NAMES)
DEVICE = torch.device("cpu")

HEDGE_WORDS = ["may represent", "possibly", "cannot exclude", "likely represents",
               "questionable", "suggestive of", "not definitively", "probable"]
HEDGE_TEMPLATES = [
    "{f} may represent an early process; correlate clinically.",
    "Findings are suggestive of {f}, though not definitively established.",
    "Cannot exclude {f} on this study.",
]
FAITHFUL_TEMPLATES = ["{f} is present.", "Findings consistent with {f}."]
STYLE_TEMPLATES = ["Impression notable for {f}, as above.", "{f} noted; no other acute findings."]
NORMAL_TEMPLATES = ["No acute cardiopulmonary abnormality.", "Unremarkable study."]

IMG_SIZE = 32
MIN_CASES_PER_READER = 15


def make_image(has_finding, q_true, rng):
    base = rng.normal(0.4, 0.05, size=(IMG_SIZE, IMG_SIZE))
    if has_finding:
        cx, cy = rng.integers(10, IMG_SIZE - 10, size=2)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2)))
        base = base + 0.35 * blob
    blur_sigma = 0.3 + 3.0 * q_true
    img = gaussian_filter(base, sigma=blur_sigma)
    img = img + rng.normal(0, 0.03 + 0.05 * q_true, size=img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def laplacian_var(img):
    lap = np.abs(np.gradient(np.gradient(img, axis=0), axis=0)
                 + np.gradient(np.gradient(img, axis=1), axis=1))
    return float(lap.var())


def generate_data(seed, n_patients=400, visits_per_patient=2, n_readers=20, beta_q=2.2):
    """beta_q is the coefficient linking true image quality (q_true, 0=sharp..1=poor)
    to the hedge logit -- the concomitant-variable confound strength. Default 2.2
    matches the original notebook exactly."""
    rng = np.random.default_rng(seed)
    readers = pd.DataFrame({
        "reader_id": range(n_readers),
        "hedge_trait": rng.normal(0, 1, n_readers),
        "omission_trait": rng.normal(0, 1, n_readers),
        "style_trait": rng.normal(0, 1, n_readers),
    })
    rows = []
    for pid in range(n_patients):
        true_finding = int(rng.integers(0, 2))
        for visit in range(visits_per_patient):
            q_true = float(rng.uniform(0, 1))
            img = make_image(true_finding, q_true, rng)
            q_meas = laplacian_var(img)
            reader_id = int(rng.integers(0, n_readers))
            rd = readers.iloc[reader_id]

            logit_faithful = 0.0
            logit_omission = -1.0 + (0.8 * rd.omission_trait if true_finding else -5.0)
            logit_hedge = -0.5 + beta_q * q_true + 0.8 * rd.hedge_trait
            logit_style = -1.0 + 0.8 * rd.style_trait
            logits = np.array([logit_faithful, logit_omission, logit_hedge, logit_style])
            p = np.exp(logits - logits.max()); p = p / p.sum()
            z = int(rng.choice(K_CLASSES, p=p))

            if not true_finding:
                text = rng.choice(NORMAL_TEMPLATES)
            elif z == 1:
                text = rng.choice(NORMAL_TEMPLATES)
            elif z == 2:
                text = rng.choice(HEDGE_TEMPLATES).format(f="the opacity")
            elif z == 3:
                text = rng.choice(STYLE_TEMPLATES).format(f="an opacity")
            else:
                text = rng.choice(FAITHFUL_TEMPLATES).format(f="an opacity")

            rows.append(dict(patient_id=pid, visit=visit, true_finding=true_finding,
                              q_true=q_true, q_meas=q_meas, reader_id=reader_id,
                              z=z, text=text, img=img))
    df = pd.DataFrame(rows)
    return df, readers


def hedge_density(text):
    words = text.lower().split()
    n = max(len(words), 1)
    hits = sum(text.lower().count(h) for h in HEDGE_WORDS)
    return hits / n


def mentions_finding(text):
    return float("opacity" in text.lower())


class FindingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        z = self.net(x).flatten(1)
        return self.fc(z).squeeze(-1)


class ImgDataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.imgs[idx]).unsqueeze(0).float()
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def _train_cnn(train_imgs, train_labels, epochs, seed):
    torch.manual_seed(seed)
    ds = ImgDataset(train_imgs, train_labels)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    cnn = FindingCNN().to(DEVICE)
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    cnn.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(cnn(xb), yb)
            loss.backward()
            opt.step()
    return cnn


def _score_cnn(cnn, imgs):
    cnn.eval()
    with torch.no_grad():
        x = torch.from_numpy(imgs).unsqueeze(1).float().to(DEVICE)
        return torch.sigmoid(cnn(x)).cpu().numpy()


def build_view2_holdout_split(df, seed, epochs=8):
    """Original notebook behaviour: single 80/20 split, CNN trained on 80%,
    scored on everyone (train scores are in-sample / leaky, holdout scores are not)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, holdout_idx = idx[:split], idx[split:]
    imgs_all = np.stack(df["img"].values)
    labels_all = df["true_finding"].values.astype(np.float32)
    cnn = _train_cnn(imgs_all[train_idx], labels_all[train_idx], epochs, seed)
    view2_raw = _score_cnn(cnn, imgs_all)
    is_holdout = np.zeros(n, dtype=bool)
    is_holdout[holdout_idx] = True
    return view2_raw, is_holdout


def build_view2_crossfit(df, seed, epochs=8, n_folds=5):
    """K-fold out-of-fold scoring: every case is scored by a model that never saw it
    during training -- the leakage fix described in the manuscript's Section VIII-A."""
    n = len(df)
    imgs_all = np.stack(df["img"].values)
    labels_all = df["true_finding"].values.astype(np.float32)
    view2_raw = np.zeros(n, dtype=np.float32)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        cnn = _train_cnn(imgs_all[train_idx], labels_all[train_idx], epochs, seed * 100 + fold_i)
        view2_raw[test_idx] = _score_cnn(cnn, imgs_all[test_idx])
    is_holdout = np.ones(n, dtype=bool)  # every score is out-of-fold
    return view2_raw, is_holdout


def build_views(df, seed, views=frozenset({"v1", "v2", "v3"}), cross_fit=True,
                 epochs=8, n_folds=5, min_cases_per_reader=MIN_CASES_PER_READER,
                 v3_mode="observed"):
    """v3_mode:
      "observed" (default, label-free): each reader's leave-one-out mean of the OBSERVED
          hedge density and of the OBSERVED non-mention rate over that reader's other reports.
          Uses report text and the reader id only, exactly what the real-data pipeline has.
      "labels" (legacy, LEAKS): leave-one-out rate of the TRUE hedge / style class of the
          reader's other cases. Kept only to quantify the leak (results_legacy_labelV3/)."""
    feat_cols = []

    if "v1" in views:
        df["view1_hedge"] = df["text"].apply(hedge_density)
        df["view1_mention"] = df["text"].apply(mentions_finding)
        feat_cols += ["view1_hedge", "view1_mention"]

    if "v2" in views:
        if cross_fit:
            view2_raw, is_holdout = build_view2_crossfit(df, seed, epochs=epochs, n_folds=n_folds)
        else:
            view2_raw, is_holdout = build_view2_holdout_split(df, seed, epochs=epochs)
        df["view2_raw"] = view2_raw
        df["is_holdout"] = is_holdout
        q_lr = LinearRegression().fit(df[["q_meas"]], df["view2_raw"])
        df["view2_resid"] = df["view2_raw"] - q_lr.predict(df[["q_meas"]])
        feat_cols += ["view2_resid"]

    if "v3" in views:
        if v3_mode == "labels":
            df["hedge_flag"] = (df["z"] == 2).astype(int)
            df["style_flag"] = (df["z"] == 3).astype(int)
            out_a, out_b = "view3_hedge_loo", "view3_style_loo"
            flag_a, flag_b = "hedge_flag", "style_flag"
        else:
            df["obs_hedge"] = df["text"].apply(hedge_density)
            df["obs_nomention"] = 1.0 - df["text"].apply(mentions_finding)
            out_a, out_b = "view3_hedge_obs", "view3_nomention_obs"
            flag_a, flag_b = "obs_hedge", "obs_nomention"
        reader_counts = df.groupby("reader_id").size()
        valid_readers = set(reader_counts[reader_counts >= min_cases_per_reader].index)

        def loo_rate(flag_col, out_col):
            sums = df.groupby("reader_id")[flag_col].transform("sum")
            counts = df.groupby("reader_id")[flag_col].transform("count")
            df[out_col] = (sums - df[flag_col]) / (counts - 1).clip(lower=1)
            pooled = df[out_col].mean()
            df.loc[~df["reader_id"].isin(valid_readers), out_col] = pooled

        loo_rate(flag_a, out_a)
        loo_rate(flag_b, out_b)
        feat_cols += [out_a, out_b]

    return df, feat_cols


def gaussian_ll(x, mu, var):
    var = np.maximum(var, 1e-4)
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mu) ** 2) / var


def _run_em(X, Qs, K, seed, n_iters=40):
    resp, ll, _ = _run_em_ext(X, Qs, K, seed, n_iters)
    return resp, ll


def _run_em_ext(X, Qs, K, seed, n_iters=40):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    order = np.argsort(X[:, 0] + rng.normal(0, 1e-6, n))
    resp = np.zeros((n, K))
    for k in range(K):
        chunk = order[k::K]
        resp[chunk, k] = 1.0
    resp += 1e-3
    resp /= resp.sum(1, keepdims=True)

    mu = np.zeros((K, d)); var = np.ones((K, d))
    coefs = np.zeros((K, Qs.shape[1] + 1))
    ll_hist = []
    for it in range(n_iters):
        Nk = resp.sum(0) + 1e-8
        for k in range(K):
            mu[k] = (resp[:, k:k+1] * X).sum(0) / Nk[k]
            var[k] = (resp[:, k:k+1] * (X - mu[k]) ** 2).sum(0) / Nk[k]
        # concomitant-variable multinomial logistic prior P(k|Q), fit by weighted
        # gradient ascent (closed-form softmax regression via IRLS-lite)
        Qd = np.hstack([np.ones((n, 1)), Qs])
        W = np.zeros((Qd.shape[1], K))
        for _ in range(25):
            logits = Qd @ W
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits); p /= p.sum(1, keepdims=True)
            grad = Qd.T @ (resp - p) / n - 1e-3 * W
            W += 0.5 * grad
        logits = Qd @ W
        logits -= logits.max(1, keepdims=True)
        prior = np.exp(logits); prior /= prior.sum(1, keepdims=True)

        loglik = np.zeros((n, K))
        for k in range(K):
            loglik[:, k] = gaussian_ll(X, mu[k], var[k]).sum(1) + np.log(prior[:, k] + 1e-12)
        m = loglik.max(1, keepdims=True)
        resp = np.exp(loglik - m); resp /= resp.sum(1, keepdims=True)
        total_ll = (m[:, 0] + np.log(np.exp(loglik - m).sum(1))).sum()
        ll_hist.append(total_ll)
    return resp, ll_hist[-1], (mu.copy(), var.copy(), W.copy())


def fit_em(df, feat_cols, seed, K=K_CLASSES, n_restarts=6, n_iters=40):
    X = StandardScaler().fit_transform(df[feat_cols].values)
    Qs = StandardScaler().fit_transform(df[["q_meas"]].values)
    best = None
    for r in range(n_restarts):
        resp, ll = _run_em(X, Qs, K, seed=seed * 1000 + r, n_iters=n_iters)
        if best is None or ll > best[1]:
            best = (resp, ll)
    resp, ll = best
    pred_raw = resp.argmax(1)
    return pred_raw, resp, ll


# ------------------------------------------------------------------ held-out EM ---
def e_step(Xs, Qs_, params, K=K_CLASSES):
    """Responsibilities for NEW rows under fitted (mu, var, W)."""
    mu, var, W = params
    Qd = np.hstack([np.ones((len(Xs), 1)), Qs_])
    logits = Qd @ W
    logits -= logits.max(1, keepdims=True)
    prior = np.exp(logits); prior /= prior.sum(1, keepdims=True)
    loglik = np.zeros((len(Xs), K))
    for k in range(K):
        loglik[:, k] = gaussian_ll(Xs, mu[k], var[k]).sum(1) + np.log(prior[:, k] + 1e-12)
    m = loglik.max(1, keepdims=True)
    resp = np.exp(loglik - m)
    return resp / resp.sum(1, keepdims=True)


def anchor_mapping(resp, hedge, mention):
    """LABEL-FREE naming of EM components using observed text only.
    hedge <- component with the highest mean hedge density;
    omission <- among the rest, lowest mean mention-of-finding;
    faithful <- among the rest, the larger component (default state); style <- the remaining one.
    Returns {class_index: component_index}."""
    K = resp.shape[1]
    mass = resp.sum(0) + 1e-12
    mh = (resp * hedge[:, None]).sum(0) / mass
    mm = (resp * mention[:, None]).sum(0) / mass
    left = set(range(K))
    mapping = {}
    c = max(left, key=lambda k: mh[k]); mapping[2] = c; left.remove(c)
    c = min(left, key=lambda k: mm[k]); mapping[1] = c; left.remove(c)
    c = max(left, key=lambda k: mass[k]); mapping[0] = c; left.remove(c)
    mapping[3] = left.pop()
    return mapping


def oracle_mapping(true_z, pred_raw, K=K_CLASSES):
    cm = confusion_matrix(true_z, pred_raw, labels=range(K))
    row_ind, col_ind = linear_sum_assignment(-cm)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}  # class -> component


def apply_mapping(pred_raw, mapping):
    inv = {c: k for k, c in mapping.items()}
    return np.array([inv.get(int(p), int(p)) for p in pred_raw])


def score_pred(true_z, pred, K=K_CLASSES):
    prec, rec, f1, sup = precision_recall_fscore_support(true_z, pred, labels=range(K), zero_division=0)
    return dict(accuracy=float(accuracy_score(true_z, pred)), balanced_accuracy=float(rec.mean()),
                macro_f1=float(f1.mean()), recall=rec.tolist(), f1=f1.tolist(), precision=prec.tolist())


def heldout_eval(df, feat_cols, seed, frac_train=0.7, n_restarts=6, n_iters=40, K=K_CLASSES):
    """Patient-level split. EM (scalers, mixture, concomitant link) is fit on TRAIN rows only;
    held-out rows are scored by the E-step. Component naming is fixed on TRAIN only, either by
    the true labels (oracle, optimistic) or by the label-free anchor rule."""
    rng = np.random.default_rng(seed + 7919)
    pats = df["patient_id"].unique(); rng.shuffle(pats)
    tr_p = set(pats[:int(frac_train * len(pats))])
    tr = df["patient_id"].isin(tr_p).values
    sx, sq = StandardScaler().fit(df.loc[tr, feat_cols].values), StandardScaler().fit(df.loc[tr, ["q_meas"]].values)
    Xtr, Qtr = sx.transform(df.loc[tr, feat_cols].values), sq.transform(df.loc[tr, ["q_meas"]].values)
    Xte, Qte = sx.transform(df.loc[~tr, feat_cols].values), sq.transform(df.loc[~tr, ["q_meas"]].values)
    best = None
    for r in range(n_restarts):
        resp, ll, params = _run_em_ext(Xtr, Qtr, K, seed=seed * 1000 + r, n_iters=n_iters)
        if best is None or ll > best[1]:
            best = (resp, ll, params)
    resp_tr, _, params = best
    resp_te = e_step(Xte, Qte, params, K)
    pr_tr, pr_te = resp_tr.argmax(1), resp_te.argmax(1)
    z_tr, z_te = df.loc[tr, "z"].values, df.loc[~tr, "z"].values
    h = df["text"].apply(hedge_density).values
    m = df["text"].apply(mentions_finding).values
    out = {}
    om = oracle_mapping(z_tr, pr_tr, K)
    am = anchor_mapping(resp_tr, h[tr], m[tr])
    out["oracle"] = score_pred(z_te, apply_mapping(pr_te, om), K)
    out["anchor"] = score_pred(z_te, apply_mapping(pr_te, am), K)
    out["oracle_train"] = score_pred(z_tr, apply_mapping(pr_tr, om), K)
    out["anchor_train"] = score_pred(z_tr, apply_mapping(pr_tr, am), K)
    out["n_train"], out["n_test"] = int(tr.sum()), int((~tr).sum())
    return out


def align_and_score(df, pred_raw, resp, K=K_CLASSES):
    true_z = df["z"].values
    cm_raw = confusion_matrix(true_z, pred_raw, labels=range(K))
    row_ind, col_ind = linear_sum_assignment(-cm_raw)
    perm_map = dict(zip(col_ind, row_ind))
    pred = np.array([perm_map.get(p, p) for p in pred_raw])
    df = df.copy()
    df["pred_z"] = pred
    col_order = [next((c for c, r in perm_map.items() if r == k), k) for k in range(K)]
    resp_aligned = resp[:, col_order]

    overall_acc = accuracy_score(true_z, pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        true_z, pred, labels=range(K), zero_division=0)
    macro_f1 = f1.mean()

    df["q_bin"] = pd.qcut(df["q_meas"], 3, labels=["low_quality", "mid_quality", "high_quality"])
    strat_acc = {}
    for qb in ["low_quality", "mid_quality", "high_quality"]:
        sub = df[df["q_bin"] == qb]
        strat_acc[qb] = accuracy_score(sub["z"], sub["pred_z"]) if len(sub) else np.nan

    p_faithful = resp_aligned[:, 0]
    corr, pval = pearsonr(p_faithful, df["q_meas"].values)

    return dict(
        accuracy=overall_acc, macro_f1=macro_f1,
        per_class_precision=dict(zip(CLASS_NAMES, prec)),
        per_class_recall=dict(zip(CLASS_NAMES, rec)),
        per_class_f1=dict(zip(CLASS_NAMES, f1)),
        per_class_support=dict(zip(CLASS_NAMES, support.tolist())),
        quality_stratified_accuracy=strat_acc,
        corr_pfaithful_q=corr, corr_pfaithful_q_pval=pval,
        pred_z=df["pred_z"], resp_aligned=resp_aligned, df=df,
    )


def run_single(seed, views=frozenset({"v1", "v2", "v3"}), n_patients=400,
               n_readers=20, beta_q=2.2, cross_fit=True, cnn_epochs=8,
               em_restarts=6, em_iters=40, visits_per_patient=2, v3_mode="observed", heldout=True):
    t0 = time.time()
    df, readers = generate_data(seed, n_patients=n_patients,
                                 visits_per_patient=visits_per_patient,
                                 n_readers=n_readers, beta_q=beta_q)
    df, feat_cols = build_views(df, seed, views=views, cross_fit=cross_fit,
                                 epochs=cnn_epochs, v3_mode=v3_mode)
    pred_raw, resp, ll = fit_em(df, feat_cols, seed=seed, n_restarts=em_restarts,
                                 n_iters=em_iters)
    result = align_and_score(df, pred_raw, resp)
    # in-sample, LABEL-FREE component naming (no true labels used to name components)
    h = df["text"].apply(hedge_density).values
    m = df["text"].apply(mentions_finding).values
    am = anchor_mapping(resp, h, m)
    result["pred_anchor"] = apply_mapping(pred_raw, am)
    result["anchor_insample"] = score_pred(df["z"].values, result["pred_anchor"])
    result["oracle_insample"] = score_pred(df["z"].values, result["pred_z"].values)
    if heldout:
        result["heldout"] = heldout_eval(df, feat_cols, seed, n_restarts=em_restarts, n_iters=em_iters)
    result["loglik"] = ll
    result["n_cases"] = len(df)
    result["n_features"] = len(feat_cols)
    result["feat_cols"] = feat_cols
    result["elapsed_sec"] = time.time() - t0
    return result

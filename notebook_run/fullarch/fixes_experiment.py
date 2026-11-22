"""
Tests two candidate fixes for problems found in Section II of the manuscript,
on (a) the synthetic image+report data and (b) the small public real-data set.

  Problem 1 (Eq. 10/11): the alignment loss is minimised by copying the image
      embedding into t' (w_v -> 1), so the encoders need not align at all.
      Variants tested:
        F0  as specified
        F1  stop-gradient on v inside the Eq. (10) mixture
        F2  F1 + w_v bounded to [0, 0.5]
        F3  F0 + an anchor term: ordinary alignment loss on the UNcorrected t
        F4  F2 + F3
  Problem 2 (Eq. 6): no term supervises the classifier toward the taxonomy.
      Variants tested:
        C0  none (as specified)
        C1a distillation from the EM mixture's responsibilities (the
            manuscript's own identifiable model, views V1+V2, label-free),
            class order fixed by a LABEL-FREE anchor (faithful := component
            with the highest mention-minus-hedge score)
        C1o same, but class order fixed by oracle Hungarian alignment to the
            true cause (synthetic only; mirrors Remark 1; upper bound)

The EM teacher is harness.fit_em on harness.build_views(views={v1,v2}); the
reader-rate view V3 is NOT used because it is built from other cases' cause
labels. Everything else (model, optimiser, epochs, split) matches train.py /
real_data_run.py.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
import harness_legacy as harness  # earlier EM, as used for Table IV
from generator import generate_dataset, build_vocab, K_CLASSES, CLASS_NAMES  # noqa: E402
from model import FullArchitecture, contrastive_loss_per_sample  # noqa: E402
from train import build_tensors  # noqa: E402
import real_data_run as R  # noqa: E402

CONFIGS = {
    "F0C0": dict(detach_v=False, cap=None, anchor=0.0, distill=None),
    "F1C0": dict(detach_v=True, cap=None, anchor=0.0, distill=None),
    "F2C0": dict(detach_v=True, cap=0.5, anchor=0.0, distill=None),
    "F3C0": dict(detach_v=False, cap=None, anchor=1.0, distill=None),
    "F4C0": dict(detach_v=True, cap=0.5, anchor=1.0, distill=None),
    "F0C1o": dict(detach_v=False, cap=None, anchor=0.0, distill="oracle"),
    "F4C1a": dict(detach_v=True, cap=0.5, anchor=1.0, distill="anchor"),
    "F4C1o": dict(detach_v=True, cap=0.5, anchor=1.0, distill="oracle"),
    # no refinement at all (t' = t); the reference for "does refinement help raw alignment"
    "F5NR": dict(detach_v=False, cap=None, anchor=0.0, distill=None, refine_steps=0),
}
SYNTH_CONFIGS = ["F0C0", "F1C0", "F2C0", "F3C0", "F4C0", "F0C1o", "F4C1a", "F4C1o"]
REAL_CONFIGS = ["F0C0", "F3C0", "F4C0", "F4C1a", "F5NR"]

REAL_HEDGES = ["possible", "possibly", "may", "suspicious", "suggest", "suggests", "likely", "probable",
               "cannot", "questionable", "uncertain", "unclear", "concern"]
REAL_MENTION = ["pneumonia", "covid", "infiltrate", "infiltrates", "opacity", "opacities", "consolidation",
                "ground", "glass", "lesion", "lesions", "infiltration"]


# ---------------------------------------------------------------- teachers
def _permute(resp, pred_raw, z, mention, hedge, mode):
    if mode == "oracle":
        cm = confusion_matrix(z, pred_raw, labels=range(K_CLASSES))
        r, c = linear_sum_assignment(-cm)
        out = np.zeros_like(resp)
        for ri, ci in zip(r, c):
            out[:, ri] = resp[:, ci]
        return out
    score = []
    for k in range(K_CLASSES):
        w = resp[:, k]
        score.append((w * mention).sum() / w.sum() - (w * hedge).sum() / w.sum() if w.sum() > 0 else -9)
    return resp[:, np.argsort(-np.array(score))]


def synth_teacher(train_df, seed, mode):
    imgs = list(train_df.img_t.values)
    d = pd.DataFrame({"img": imgs, "true_finding": train_df.true_finding_t.values,
                      "text": train_df.report_t.values,
                      "q_meas": [harness.laplacian_var(i) for i in imgs]})
    d, feat = harness.build_views(d, seed, views=frozenset({"v1", "v2"}), cross_fit=True)
    pred_raw, resp, ll = harness.fit_em(d, feat, seed)
    z = train_df.z.values
    out = _permute(resp, pred_raw, z, d["view1_mention"].values, d["view1_hedge"].values, mode)
    return out


def _mention_default(text):
    return any(w in REAL_MENTION for w in R.tok(text))


mention_fn = _mention_default  # replaceable (iu_xray_run.py installs a negation-aware version)


def real_teacher(tr, seed):
    texts = tr.txt.values
    toks = [R.tok(t) for t in texts]
    hedge = np.array([sum(w in REAL_HEDGES for w in t) / max(len(t), 1) for t in toks])
    mention = np.array([float(mention_fn(t)) for t in texts])
    imgs = np.stack(tr.img.values).astype(np.float32)
    if "y_target" in tr.columns:  # datasets with a report-derived normal/abnormal flag
        y = tr.y_target.values.astype(np.float32)
    else:
        y = tr.finding.astype(str).str.startswith("Pneumonia").values.astype(np.float32)
    groups = tr.patientid.values
    v2 = np.zeros(len(tr), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for fi, (a, b) in enumerate(gkf.split(imgs, y, groups)):
        cnn = harness._train_cnn(imgs[a], y[a], 8, seed * 100 + fi)
        v2[b] = harness._score_cnn(cnn, imgs[b])
    q = np.array([harness.laplacian_var(i) for i in imgs])
    from sklearn.linear_model import LinearRegression
    resid = v2 - LinearRegression().fit(q[:, None], v2).predict(q[:, None])
    d = pd.DataFrame({"view1_hedge": hedge, "view1_mention": mention, "view2_resid": resid, "q_meas": q})
    if hasattr(harness, "BINARY_FEATS"):   # corrected EM: hedge indicator is binary (Bernoulli)
        d["view1_hedge"] = (d["view1_hedge"] > 0).astype(float)
    pred_raw, resp, ll = harness.fit_em(d, ["view1_hedge", "view1_mention", "view2_resid"], seed)
    return _permute(resp, pred_raw, None, mention, hedge, "anchor")


# ---------------------------------------------------------------- metrics
def cls_metrics(p_c, z):
    pred = p_c.argmax(1)
    cm = confusion_matrix(z, pred, labels=range(K_CLASSES))
    r, c = linear_sum_assignment(-cm)
    pm = dict(zip(c, r))
    pa = np.array([pm.get(p, p) for p in pred])
    rec = precision_recall_fscore_support(z, pa, labels=range(K_CLASSES), zero_division=0)
    _, rec_, f1, _ = rec
    _, rec_raw, _, _ = precision_recall_fscore_support(z, pred, labels=range(K_CLASSES), zero_division=0)
    return dict(bal_acc_aligned=float(rec_.mean()), macro_f1_aligned=float(f1.mean()),
                bal_acc_unaligned=float(rec_raw.mean()))


def cls_dist(p_c):
    pred = p_c.argmax(1)
    ent = float(-(p_c * np.log(np.clip(p_c, 1e-8, None))).sum(1).mean())
    return dict(pred_frac=[float((pred == k).mean()) for k in range(K_CLASSES)], mean_entropy=ent,
                n_classes_used=int(len(np.unique(pred))))


def align_auc(v, t, finding_i, mention_j):
    s = (F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).t()).numpy()
    m = (finding_i[:, None] == mention_j[None, :]).astype(int)
    return float(roc_auc_score(m.flatten(), s.flatten()))


# ---------------------------------------------------------------- training
def train_student(cfg, trT, teT, vocab_n, seed, kind, resp=None, max_len=20, epochs=25, pretrain=8,
                  bs=32, lam=1.0, mu=1.0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = FullArchitecture(vocab_size=vocab_n, max_len=max_len, detach_v=cfg["detach_v"], w_v_cap=cfg["cap"],
                         refine_steps=cfg.get("refine_steps", 2))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    n = trT["img_t"].shape[0]
    if resp is not None:
        trT = dict(trT)
        trT["resp"] = torch.from_numpy(resp.astype(np.float32))

    def nce(v, t, b):
        if kind == "real":
            return R.masked_infonce(v, t, b["pid"])
        return contrastive_loss_per_sample(v, t)

    hist = []
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(n)
        ls = []
        for s in range(0, n, bs):
            sel = torch.from_numpy(idx[s:s + bs])
            if len(sel) < 4:
                continue
            b = {k: v[sel] for k, v in trT.items()}
            out = m.encode_batch(b)
            tp = m.correct(out["v_t"], out["t_t"], out["p_c"])
            ell = nce(out["v_t"], tp, b)
            gamma = 1.0 - out["p_c"][:, 0]
            L = torch.mean((1.0 + lam * gamma.detach()) * ell)
            if cfg["anchor"] > 0:
                L = L + cfg["anchor"] * torch.mean(nce(out["v_t"], out["t_t"], b))
            if ep < pretrain:
                q = m.anticipation(out["v_t"])
                p = out["p_c"].detach().clamp_min(1e-8)
                L = L + (p * (p.log() - q.clamp_min(1e-8).log())).sum(-1).mean()
            if cfg["distill"] is not None:
                r = b["resp"].clamp_min(1e-8)
                pc = out["p_c"].clamp_min(1e-8)
                L = L + mu * (r * (r.log() - pc.log())).sum(-1).mean()
            opt.zero_grad()
            L.backward()
            opt.step()
            ls.append(L.item())
        hist.append(float(np.mean(ls)))
    m.eval()
    return m, hist


def probe(m, T):
    with torch.no_grad():
        o = m.encode_batch(T)
        tp = m.correct(o["v_t"], o["t_t"], o["p_c"])
        w_v, _ = m.reliability(o["v_t"], o["t_t"], o["p_c"])
        return o, tp, float(w_v.mean())


# ---------------------------------------------------------------- synthetic
def run_synth(seed, names, n_patients=400, teacher_cache={}):
    df = generate_dataset(seed=seed, n_patients=n_patients)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(0.7 * len(df))
    tr = df.iloc[idx[:cut]].reset_index(drop=True)
    te = df.iloc[idx[cut:]].reset_index(drop=True)
    vocab = build_vocab(list(tr["report_t"]) + [t for t in tr["report_prev"] if isinstance(t, str)]
                        + [t for t in tr["report_second"] if isinstance(t, str)])
    trT, teT = build_tensors(tr, vocab), build_tensors(te, vocab)
    finding_te = te.true_finding_t.values
    mention_te = np.array([float("opacity" in t) for t in te.report_t.values])
    teachers = {}
    for mode in ("anchor", "oracle"):
        teachers[mode] = synth_teacher(tr, seed, mode)
    z_tr = tr.z.values
    teacher_quality = {mode: cls_metrics(r, z_tr) for mode, r in teachers.items()}
    results = []
    for name in names:
        cfg = CONFIGS[name]
        resp = teachers[cfg["distill"]] if cfg["distill"] else None
        t0 = time.time()
        m, hist = train_student(cfg, trT, teT, len(vocab), seed, "synth", resp=resp)
        o, tp, wv = probe(m, teT)
        p = o["p_c"].numpy()
        with torch.no_grad():
            otr = m.encode_batch(trT)
        r = dict(config=name, seed=seed, n_train=len(tr), n_test=len(te), loss_last=hist[-1],
                 mean_w_v_step1=wv,
                 cos_v_text_raw=float(F.cosine_similarity(o["v_t"], o["t_t"], dim=-1).mean()),
                 cos_v_tprime=float(F.cosine_similarity(o["v_t"], tp, dim=-1).mean()),
                 align_auc_raw=align_auc(o["v_t"], o["t_t"], finding_te, mention_te),
                 align_auc_corrected=align_auc(o["v_t"], tp, finding_te, mention_te),
                 **cls_metrics(p, te.z.values), **cls_dist(p),
                 teacher_agreement_train=(float((otr["p_c"].argmax(1).numpy() == resp.argmax(1)).mean())
                                           if resp is not None else None),
                 teacher_quality_train=teacher_quality[cfg["distill"]] if cfg["distill"] else None,
                 seconds=time.time() - t0)
        print(f"[synth s{seed}] {name:6s} w_v={wv:.3f} cos(v,t)={r['cos_v_text_raw']:.3f} "
              f"auc_raw={r['align_auc_raw']:.3f} balacc={r['bal_acc_aligned']:.3f} "
              f"classes={r['n_classes_used']} ({r['seconds']:.0f}s)", flush=True)
        results.append(r)
    return results


# ---------------------------------------------------------------- real data
def run_real(seed, names, d, epochs=30, extra_fn=None):
    rng = np.random.default_rng(seed)
    pids = np.array(d.patientid.unique(), dtype=object)
    rng.shuffle(pids)
    cut = int(0.7 * len(pids))
    tr = d[d.patientid.isin(set(pids[:cut]))]
    te = d[d.patientid.isin(set(pids[cut:]))]
    cnt = {}
    for t in tr.txt:
        for w in R.tok(t):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {w: i + 2 for i, w in enumerate(sorted(w for w, c in cnt.items() if c >= 2))}
    vocab["<unk>"] = 1
    trT, teT = R.build(tr, vocab), R.build(te, vocab)
    teacher = real_teacher(tr, seed)
    chance = R.chance(teT["pid"], rng)
    results = []
    for name in names:
        cfg = CONFIGS[name]
        resp = teacher if cfg["distill"] else None
        t0 = time.time()
        m, hist = train_student(cfg, trT, teT, len(vocab) + 2, seed, "real", resp=resp,
                                max_len=R.MAX_LEN, epochs=epochs)
        o, tp, wv = probe(m, teT)
        p = o["p_c"].numpy()
        with torch.no_grad():
            otr = m.encode_batch(trT)
        r = dict(config=name, seed=seed, n_train=len(tr), n_test=len(te), loss_last=hist[-1],
                 mean_w_v_step1=wv,
                 cos_v_text_raw=float(F.cosine_similarity(o["v_t"], o["t_t"], dim=-1).mean()),
                 cos_v_tprime=float(F.cosine_similarity(o["v_t"], tp, dim=-1).mean()),
                 retrieval_raw=R.retrieval(o["v_t"], o["t_t"], teT["pid"]),
                 retrieval_corrected=R.retrieval(o["v_t"], tp, teT["pid"]),
                 chance=chance, **cls_dist(p),
                 teacher_agreement_train=(float((otr["p_c"].argmax(1).numpy() == resp.argmax(1)).mean())
                                           if resp is not None else None),
                 seconds=time.time() - t0)
        if extra_fn is not None:
            r.update(extra_fn(m, trT, teT, tr, te))
        print(f"[real s{seed}] {name:6s} w_v={wv:.3f} cos(v,t)={r['cos_v_text_raw']:.3f} "
              f"R@5raw={r['retrieval_raw']['R@5']:.3f} (chance {chance['R@5']:.3f}) "
              f"classes={r['n_classes_used']} ent={r['mean_entropy']:.2f} ({r['seconds']:.0f}s)", flush=True)
        results.append(r)
    return results


if __name__ == "__main__":
    which = sys.argv[1]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    os.makedirs("results", exist_ok=True)
    res = []
    if which == "synth":
        for s in seeds:
            res += run_synth(s, SYNTH_CONFIGS)
        json.dump(res, open("results/fixes_synthetic.json", "w"), indent=2)
    else:
        d = R.load(sys.argv[3] if len(sys.argv) > 3 else "/home/claude/work/real")
        for s in seeds:
            res += run_real(s, REAL_CONFIGS, d)
        json.dump(res, open("results/fixes_real.json", "w"), indent=2)

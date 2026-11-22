"""
Real-data smoke test of the full Section IV architecture (Algorithm 1 as
specified, Algorithm 2 inference) on a small PUBLIC chest X-ray + text set
that is reachable from the sandbox: the GitHub-hosted covid-chestxray-dataset
(Cohen et al., arXiv:2006.11988; images + metadata.csv with per-image
clinical notes).

THIS IS NOT THE IU X-RAY / MIMIC-CXR EVALUATION. Differences that matter:
  * text is case-report clinical notes, NOT radiology reports (no
    Findings/Impression structure, many notes are shared by all of a
    patient's images);
  * there is NO ground-truth inconsistency label, so the four-class taxonomy
    cannot be scored; only unsupervised/retrieval diagnostics are possible;
  * no second-reader report exists (the reader-reader axis is always missing,
    which exercises the missing-axis convention of Eq. 5);
  * a real "prior" exists for patients with >1 image (same patientid, smaller
    day offset), so the current-prior axis is real, not simulated.

Deviation from the synthetic run (disclosed): the in-batch InfoNCE masks
same-patient off-diagonal pairs, because one patient's images share
near-identical notes and would otherwise be false negatives.
"""
import json
import re
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from model import FullArchitecture, CLASS_NAMES, K_CLASSES

IMG = 64
MAX_LEN = 48
DATA = "../../real"  # metadata + images (not redistributed)


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def load(data_dir):
    d = pd.read_csv(f"{data_dir}/pairs.csv").reset_index(drop=True)
    d["patientid"] = d.patientid.astype(str)
    d["offset"] = pd.to_numeric(d.offset, errors="coerce").fillna(0)
    imgs = []
    for fn in d.filename:
        im = Image.open(f"{data_dir}/img/{fn}").convert("L").resize((IMG, IMG))
        a = np.asarray(im, dtype=np.float32) / 255.0
        imgs.append((a - a.mean()) / (a.std() + 1e-6))
    d["img"] = imgs
    # real prior: same patient, strictly smaller offset, nearest earlier one
    prior = []
    for i, r in d.iterrows():
        c = d[(d.patientid == r.patientid) & (d.offset < r.offset)]
        prior.append(int(c.sort_values("offset").index[-1]) if len(c) else -1)
    d["prior_idx"] = prior
    return d


def build(d, vocab):
    def ids(t):
        x = [vocab.get(w, 1) for w in tok(t)][:MAX_LEN]
        return x + [0] * (MAX_LEN - len(x))

    n = len(d)
    img = np.stack(d.img.values)[:, None].astype(np.float32)
    pidx = d.prior_idx.values
    # prior index is global; remap through a position lookup
    pos = {g: k for k, g in enumerate(d.index)}
    img_prev = np.zeros_like(img)
    rep_prev = np.zeros((n, MAX_LEN), dtype=np.int64)
    has_prior = np.zeros(n, dtype=np.float32)
    txt_by_global = dict(zip(d.index, d.txt))
    img_by_global = dict(zip(d.index, d.img))
    for k, g in enumerate(pidx):
        if g >= 0 and g in pos:
            img_prev[k, 0] = img_by_global[g]
            rep_prev[k] = ids(txt_by_global[g])
            has_prior[k] = 1.0
    return dict(
        img_t=torch.from_numpy(img), img_prev=torch.from_numpy(img_prev),
        report_t_ids=torch.tensor([ids(t) for t in d.txt], dtype=torch.long),
        report_prev_ids=torch.from_numpy(rep_prev),
        report_second_ids=torch.zeros(n, MAX_LEN, dtype=torch.long),
        has_prior=torch.from_numpy(has_prior),
        has_second=torch.zeros(n),
        pid=torch.tensor(pd.factorize(d.patientid.astype(str))[0]),
    )


def masked_infonce(v, t, pid, temp=0.1):
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    sim = v @ t.t() / temp
    same = (pid[:, None] == pid[None, :]) & ~torch.eye(len(pid), dtype=torch.bool)
    sim = sim.masked_fill(same, -1e4)
    tgt = torch.arange(len(v))
    return 0.5 * (F.cross_entropy(sim, tgt, reduction="none") +
                  F.cross_entropy(sim.t(), tgt, reduction="none"))


def retrieval(v, t, pid, ks=(1, 5, 10)):
    s = F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).t()
    pos = pid[:, None] == pid[None, :]
    order = s.argsort(dim=1, descending=True)
    ranked_pos = torch.gather(pos, 1, order)
    first = ranked_pos.float().argmax(1) + 1  # rank of best positive
    out = {f"R@{k}": float((first <= k).float().mean()) for k in ks}
    out["median_rank"] = float(first.float().median())
    return out


def chance(pid, rng, reps=200, ks=(1, 5, 10)):
    acc = {f"R@{k}": [] for k in ks}
    med = []
    for _ in range(reps):
        s = torch.from_numpy(rng.random((len(pid), len(pid))).astype(np.float32))
        pos = pid[:, None] == pid[None, :]
        order = s.argsort(dim=1, descending=True)
        first = torch.gather(pos, 1, order).float().argmax(1) + 1
        for k in ks:
            acc[f"R@{k}"].append(float((first <= k).float().mean()))
        med.append(float(first.float().median()))
    r = {k: float(np.mean(v)) for k, v in acc.items()}
    r["median_rank"] = float(np.mean(med))
    return r


def run(d, seed, epochs=30, pretrain=8, bs=32, lam=1.0, tau_r=0.7, correct=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pids = d.patientid.unique()
    pids = np.array(pids, dtype=object); rng.shuffle(pids)
    cut = int(0.7 * len(pids))
    tr_p, te_p = set(pids[:cut]), set(pids[cut:])
    tr = d[d.patientid.isin(tr_p)]
    te = d[d.patientid.isin(te_p)]
    cnt = {}
    for t in tr.txt:
        for w in tok(t):
            cnt[w] = cnt.get(w, 0) + 1
    vocab = {w: i + 2 for i, w in enumerate(sorted(w for w, c in cnt.items() if c >= 2))}
    vocab["<unk>"] = 1
    trT, teT = build(tr, vocab), build(te, vocab)

    m = FullArchitecture(vocab_size=len(vocab) + 2, max_len=MAX_LEN)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(tr)
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
            tp = m.correct(out["v_t"], out["t_t"], out["p_c"]) if correct else out["t_t"]
            ell = masked_infonce(out["v_t"], tp, b["pid"])
            gamma = 1.0 - out["p_c"][:, 0]
            L = torch.mean((1.0 + lam * gamma.detach()) * ell)
            if ep < pretrain:
                q = m.anticipation(out["v_t"])
                p = out["p_c"].detach().clamp_min(1e-8)
                L = L + (p * (p.log() - q.clamp_min(1e-8).log())).sum(-1).mean()
            opt.zero_grad()
            L.backward()
            opt.step()
            ls.append(L.item())
        hist.append(float(np.mean(ls)))

    m.eval()
    with torch.no_grad():
        o = m.encode_batch(teT)
        tp = m.correct(o["v_t"], o["t_t"], o["p_c"]) if correct else o["t_t"]
        ret_corr = retrieval(o["v_t"], tp, teT["pid"])
        ret_raw = retrieval(o["v_t"], o["t_t"], teT["pid"])
        p = o["p_c"]
        pred = p.argmax(1).numpy()
        conf = p.max(1).values.numpy()
        ent = float(-(p * p.clamp_min(1e-8).log()).sum(-1).mean())
        routes = []
        for c, h in zip(pred, conf):
            if c == 0:
                routes.append("auto_corrected")
            elif c == 1 and h >= tau_r:
                routes.append("flag_for_radiologist")
            elif h >= tau_r:
                routes.append("auto_corrected")
            else:
                routes.append("flag_for_radiologist")
        nan = bool(torch.isnan(o["p_c"]).any() or torch.isnan(tp).any())
        # trajectory of the reliability weights across refinement steps (Eq. 9-10)
        wv_traj, cur = [], o["t_t"]
        for _ in range(m.refine_steps):
            w_v, w_t = m.reliability(o["v_t"], cur, o["p_c"])
            wv_traj.append(float(w_v.mean()))
            cur = w_v * o["v_t"] + w_t * cur
        cos_vt_raw = float(F.cosine_similarity(o["v_t"], o["t_t"], dim=-1).mean())
        cos_vt_corr = float(F.cosine_similarity(o["v_t"], tp, dim=-1).mean())
    return dict(
        seed=seed, correct=correct, n_train=len(tr), n_test=len(te),
        n_train_patients=len(tr_p), n_test_patients=len(te_p),
        frac_test_with_prior=float(teT["has_prior"].mean()),
        vocab=len(vocab), loss_first=hist[0], loss_last=hist[-1],
        retrieval_corrected=ret_corr, retrieval_uncorrected=ret_raw,
        chance=chance(teT["pid"], rng),
        pred_class_frac={CLASS_NAMES[k]: float((pred == k).mean()) for k in range(K_CLASSES)},
        mean_p_c={CLASS_NAMES[k]: float(p[:, k].mean()) for k in range(K_CLASSES)},
        mean_entropy_nats=ent, max_entropy_nats=float(np.log(K_CLASSES)),
        route_frac=pd.Series(routes).value_counts(normalize=True).to_dict(),
        any_nan=nan, mean_w_v_per_step=wv_traj,
        mean_cos_v_t_raw=cos_vt_raw, mean_cos_v_tprime=cos_vt_corr,
    )


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA
    t0 = time.time()
    d = load(data_dir)
    print(f"{len(d)} image-text pairs, {d.patientid.nunique()} patients, "
          f"{(d.prior_idx >= 0).mean():.1%} with a real prior")
    res = []
    for corr in (True, False):
        for seed in range(3):
            r = run(d, seed, correct=corr)
            print(f"correct={corr} seed={seed} R@5={r['retrieval_corrected']['R@5']:.3f} "
                  f"(chance {r['chance']['R@5']:.3f}) medrank={r['retrieval_corrected']['median_rank']:.0f} "
                  f"pred={ {k: round(v,2) for k,v in r['pred_class_frac'].items()} } nan={r['any_nan']}")
            res.append(r)
    json.dump(res, open("results/real_data_results.json", "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s")

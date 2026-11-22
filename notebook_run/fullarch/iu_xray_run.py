"""
Run the Section II architecture (and the proposed stabilization of Section II-G)
on the Indiana University chest X-ray collection (Open-I), as distributed on
Kaggle:  https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university

USAGE (Kaggle notebook, or any machine with the dataset downloaded):

    cd full_arch
    python3 iu_xray_run.py /kaggle/input/chest-xrays-indiana-university            # default run
    python3 iu_xray_run.py <data_dir> --seeds 0,1,2 --max-cases 1500 --epochs 30   # smaller / faster
    python3 iu_xray_run.py <data_dir> --configs F5NR,F0C0,F3C0,F4C0,F4C1a

    # any other dataset that has images + free-text reports: write a CSV with the
    # columns  filename, patientid, txt  (and optional y_target = 0/1 abnormal flag)
    python3 iu_xray_run.py <data_dir> --mode pairs --pairs-csv pairs.csv --img-dir <dir>

Output: results/iu_xray_results.json and results/iu_xray_summary.md (tables).

WHAT THIS DOES AND DOES NOT TEST
  * Images + real radiology reports (findings + impression), one frontal image
    per study, split at study level (IU provides no patient id beyond the study
    uid; one study per patient is the norm in this collection).
  * No longitudinal prior and no second reader exist in IU, so the current-prior
    and reader-reader axes are always missing (exercises the missing-axis
    convention of the consistency graph).
  * There is NO inconsistency label, so the four-class taxonomy cannot be scored.
    Reported instead: raw-embedding retrieval (exact image<->report pair among the
    test set, chance = k/N), w_v and cosines (Eq. 10 shortcut diagnostics),
    classifier collapse (classes used, entropy), teacher agreement, and a
    linear-probe AUC for a REPORT-DERIVED normal/abnormal flag on the raw image
    and raw text embeddings (sanity check that the encoders learned something).
  * CPU-only, d=32, small CNN and Transformer, 64x64 images: a software and
    diagnostic test, not a performance claim. Expect roughly 5-20 minutes per
    configuration per seed on 4 CPU cores for ~3,000 cases.

SCHEMA ASSUMPTIONS (discovered at runtime, printed before use)
  Reports CSV   : has columns 'findings' and 'impression' (any case); uses
                  'uid' as study id; 'problems' / 'mesh' (if present) for the
                  normal/abnormal flag.
  Projections   : has 'uid', 'filename', 'projection' (frontal rows are kept).
  Images        : found anywhere under <data_dir> by basename.
  If your copy differs, use --mode pairs.
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import fixes_experiment as FX
import real_data_run as R

NEG = re.compile(r"\b(no|without|negative for|free of|absence of|resolved|not)\b", re.I)
FINDING_WORDS = {
    "opacity", "opacities", "consolidation", "infiltrate", "infiltrates", "effusion", "effusions",
    "pneumothorax", "cardiomegaly", "edema", "nodule", "nodules", "mass", "atelectasis", "pneumonia",
    "fracture", "emphysema", "scarring", "granuloma", "granulomas", "calcification", "calcifications",
    "enlarged", "thickening", "hernia", "lesion", "lesions", "airspace",
}


def negation_aware_mention(text):
    """True if some sentence names a finding without a negation cue in the same sentence."""
    for s in re.split(r"[.;\n]", text.lower()):
        if any(w in FINDING_WORDS for w in R.tok(s)) and not NEG.search(s):
            return True
    return False


# ---------------------------------------------------------------- loading
def _index_images(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                idx.setdefault(fn, os.path.join(dp, fn))
    return idx


def _find_image(name, idx):
    if isinstance(name, str) and os.path.isabs(name) and os.path.exists(name):
        return name                      # full path given (pairs mode): no basename lookup, no collisions
    for cand in (name, name + ".png", os.path.splitext(name)[0] + ".png", os.path.basename(name),
                 os.path.basename(name) + ".png"):
        if cand in idx:
            return idx[cand]
    return None


def _discover_csvs(root):
    rep = proj = None
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.lower().endswith(".csv"):
                continue
            f = os.path.join(dp, fn)
            try:
                cols = {c.lower() for c in pd.read_csv(f, nrows=0).columns}
            except Exception:
                continue
            if {"findings", "impression"} <= cols:
                rep = f
            if {"filename", "projection"} <= cols:
                proj = f
    return rep, proj


def _clean(s):
    s = "" if pd.isna(s) else str(s)
    s = re.sub(r"\bx{2,}\b", " ", s, flags=re.I)  # IU de-identification placeholders
    return re.sub(r"\s+", " ", s).strip()


def load_iu(root):
    rep_f, proj_f = _discover_csvs(root)
    print(f"reports csv    : {rep_f}\nprojections csv: {proj_f}")
    if rep_f is None or proj_f is None:
        sys.exit("Could not find the IU reports/projections CSVs under the data dir; "
                 "use --mode pairs with your own pairs.csv (see the module docstring).")
    rep = pd.read_csv(rep_f)
    proj = pd.read_csv(proj_f)
    rep.columns = [c.lower() for c in rep.columns]
    proj.columns = [c.lower() for c in proj.columns]
    print("report columns :", list(rep.columns))
    print("projection cols:", list(proj.columns))
    proj = proj[proj.projection.astype(str).str.lower().str.contains("front")]
    proj = proj.drop_duplicates("uid", keep="first")
    d = proj.merge(rep, on="uid", how="inner")
    d["txt"] = (d.findings.map(_clean) + " " + d.impression.map(_clean)).str.strip()
    d = d[d.txt.map(lambda t: len(R.tok(t))) >= 5].copy()
    if "problems" in d.columns:
        d["y_target"] = (d.problems.astype(str).str.strip().str.lower() != "normal").astype(int)
    elif "mesh" in d.columns:
        d["y_target"] = (~d.mesh.astype(str).str.lower().str.contains("normal")).astype(int)
    else:
        d["y_target"] = d.txt.str.lower().str.contains(r"no acute").map(lambda b: 0 if b else 1)
    d["patientid"] = d.uid.astype(str)
    d["offset"] = 0.0
    d["filename"] = d.filename.astype(str)
    return d.reset_index(drop=True)


def load_pairs(pairs_csv, img_dir):
    d = pd.read_csv(pairs_csv)
    for c in ("filename", "patientid", "txt"):
        if c not in d.columns:
            sys.exit(f"pairs csv needs a '{c}' column")
    d["patientid"] = d.patientid.astype(str)
    d["offset"] = 0.0
    d["txt"] = d.txt.map(_clean)
    if "y_target" not in d.columns:
        d["y_target"] = 0
        print("WARNING: no y_target column; the CNN-probe view of the EM teacher will be uninformative.")
    d["__root"] = img_dir
    return d


def attach_images(d, root, size, max_cases, seed=0):
    idx = _index_images(root)
    print(f"images indexed : {len(idx)} under {root}")
    paths = [_find_image(f, idx) for f in d.filename]
    d = d.assign(path=paths)
    miss = d.path.isna().sum()
    if miss:
        print(f"dropping {miss} rows whose image file was not found")
    d = d[d.path.notna()].reset_index(drop=True)
    if max_cases and len(d) > max_cases:
        d = d.sample(max_cases, random_state=seed).reset_index(drop=True)
    imgs = []
    for p in d.path:
        im = Image.open(p).convert("L").resize((size, size))
        a = np.asarray(im, dtype=np.float32) / 255.0
        imgs.append((a - a.mean()) / (a.std() + 1e-6))
    d["img"] = imgs
    d["prior_idx"] = -1  # IU has no longitudinal linkage
    return d


# ---------------------------------------------------------------- extra metrics
def make_extra_fn():
    def extra(m, trT, teT, tr, te):
        with torch.no_grad():
            otr = m.encode_batch(trT)
            ote = m.encode_batch(teT)
        out = {}
        ytr, yte = tr.y_target.values, te.y_target.values
        if len(set(ytr)) > 1 and len(set(yte)) > 1:
            for name, key in (("probe_auc_abnormal_from_image", "v_t"), ("probe_auc_abnormal_from_text", "t_t")):
                clf = LogisticRegression(max_iter=2000).fit(otr[key].numpy(), ytr)
                out[name] = float(roc_auc_score(yte, clf.predict_proba(ote[key].numpy())[:, 1]))
        out["test_abnormal_fraction"] = float(yte.mean())
        return out
    return extra


# ---------------------------------------------------------------- summary
def summarize(res, path):
    df = pd.DataFrame([{
        "config": r["config"], "seed": r["seed"],
        "w_v": (np.nan if FX.CONFIGS[r["config"]].get("refine_steps", 2) == 0 else r["mean_w_v_step1"]),
        "cos_v_rawtext": r["cos_v_text_raw"],
        "R@1": r["retrieval_raw"]["R@1"], "R@5": r["retrieval_raw"]["R@5"], "R@10": r["retrieval_raw"]["R@10"],
        "chance_R@5": r["chance"]["R@5"], "classes_used": r["n_classes_used"], "entropy": r["mean_entropy"],
        "teacher_agree": r["teacher_agreement_train"],
        "probe_auc_img": r.get("probe_auc_abnormal_from_image"),
        "probe_auc_txt": r.get("probe_auc_abnormal_from_text"),
    } for r in res])
    g = df.groupby("config", sort=False)
    ms = lambda c: g[c].apply(lambda s: "---" if s.isna().all() else f"{s.mean():.3f} ± {s.std(ddof=0):.3f}")
    t = pd.DataFrame({c: ms(c) for c in ["w_v", "cos_v_rawtext", "R@1", "R@5", "R@10", "chance_R@5", "entropy",
                                           "teacher_agree", "probe_auc_img", "probe_auc_txt"]})
    t["classes_used"] = g["classes_used"].apply(lambda s: str(list(map(int, s))))
    md = "### IU X-ray run (mean ± s.d. over seeds)\n\n" + t.to_markdown() + "\n"
    open(path, "w").write(md)
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--mode", choices=["iu", "pairs"], default="iu")
    ap.add_argument("--pairs-csv")
    ap.add_argument("--img-dir")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--configs", default="F5NR,F0C0,F3C0,F4C0,F4C1a")
    ap.add_argument("--max-cases", type=int, default=0, help="subsample for speed (0 = all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=96, help="report tokens kept")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    R.IMG = a.img_size
    R.MAX_LEN = a.max_len
    FX.mention_fn = negation_aware_mention

    t0 = time.time()
    d = load_iu(a.data_dir) if a.mode == "iu" else load_pairs(a.pairs_csv, a.img_dir or a.data_dir)
    d = attach_images(d, a.data_dir if a.mode == "iu" else (a.img_dir or a.data_dir), a.img_size, a.max_cases)
    print(f"{len(d)} image-report pairs, {d.patientid.nunique()} patients/studies, "
          f"abnormal fraction {d.y_target.mean():.2f}, median report length "
          f"{int(np.median([len(R.tok(t)) for t in d.txt]))} tokens (kept: {a.max_len})")

    os.makedirs("results", exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    names = a.configs.split(",")
    for n in names:
        if n not in FX.CONFIGS:
            sys.exit(f"unknown config {n}; choose from {list(FX.CONFIGS)}")
    res = []
    for s in seeds:
        res += FX.run_real(s, names, d, epochs=a.epochs, extra_fn=make_extra_fn())
        json.dump(res, open("results/iu_xray_results.json", "w"), indent=2)  # incremental save
    summarize(res, "results/iu_xray_summary.md")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

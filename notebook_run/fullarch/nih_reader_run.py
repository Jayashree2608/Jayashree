"""
Reader-disagreement test on NIH ChestX-ray14 with the expert labels released by
Google (Majkowska et al., Radiology 2019; Nabulsi et al., Sci. Rep. 2021).
This is the open substitute for the planned VinDr-CXR reader-reader validation.

QUESTION TESTED
  Can the image encoder of the architecture (the input of the anticipation head
  of Eq. 7, which sees the image only) predict, before any report exists, where
  independent radiologists will disagree or hedge?

  Targets (per image, from the individual reader reads; YES / NO / HEDGE):
    disagree_any  : for at least one finding, readers' reads are not all identical
    hedge_any     : at least one reader answered HEDGE for some finding
    disagree_<f>  : readers split on finding f (opacity, nodule, pneumothorax, fracture)
  Reported as test AUC (chance 0.5), mean +/- s.d. over seeds, next to two
  reference predictors that need no image model:
    adjudicated-count : number of findings the adjudicated label calls YES
                        (is disagreement just "the finding is present"?)
    quality proxy     : Laplacian variance of the image (the paper's V2 quality proxy)
  An image model that does not beat these has learned nothing about disagreement
  beyond finding presence or acquisition quality.

WHAT THIS DOES NOT TEST
  ChestX-ray14 has no radiology reports, so the reader-reader axis on report
  embeddings, the four-class taxonomy (omission and style need report text) and the
  consistency graph cannot be tested here. This checks only the image-side
  claim behind the anticipation head, using structured reader labels.

DATA (not redistributable; obtain yourself)
  1. Expert labels: request via the Google form linked from
     https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest
     (section "Additional labels"); use  four_findings_expert_labels/  (4,374 images,
     individual_readers.csv + validation_labels.csv + test_labels.csv).
     The 810-image all-findings set (five readers, 14 findings) also works: pass
     --labels-dir all_findings_expert_labels (schema is discovered at runtime).
  2. The NIH ChestX-ray14 images (files like 00000001_000.png), anywhere under --images-dir
     (e.g. the Kaggle input directory of a ChestX-ray14 copy).

USAGE
    python3 nih_reader_run.py --labels-dir <.../four_findings_expert_labels> --images-dir <nih images root>
    options: --seeds 0,1,2  --epochs 25  --img-size 96  --max-images 0  --threads 0

SCHEMA ASSUMPTIONS (the column names of the released CSVs were not visible when this
was written; everything below is discovered at runtime and printed)
  individual_readers.csv : one row per reader per image; an image-id column (name
      contains 'image'), a reader column (name contains 'reader'), and one column
      per finding holding YES / NO / HEDGE.
  *_labels.csv           : image-id column plus the adjudicated finding columns (YES/NO).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import ImageEncoder  # noqa: E402
import harness_current as harness  # corrected EM (Part A)

VALUES = {"YES", "NO", "HEDGE"}


# ---------------------------------------------------------------- labels
def _find(root, name):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower() == name.lower():
                return os.path.join(dp, fn)
    return None


def _id_col(df):
    for c in df.columns:
        if "image" in c.lower() and "id" in c.lower():
            return c
    for c in df.columns:
        if "image" in c.lower() or "file" in c.lower() or "id" == c.lower():
            return c
    return df.columns[0]


def load_reads(labels_dir):
    f = _find(labels_dir, "individual_readers.csv") or _find(labels_dir, "test_individual_readers.csv")
    if f is None:
        sys.exit(f"individual_readers.csv (or test_individual_readers.csv) not found under {labels_dir}")
    df = pd.read_csv(f)
    print(f"reads file     : {f}\ncolumns        : {list(df.columns)}")
    idc = _id_col(df)
    rc = next((c for c in df.columns if "reader" in c.lower()), None)
    if rc is None:
        sys.exit("no reader-id column found (name containing 'reader')")
    findings = [c for c in df.columns if c not in (idc, rc)
                and set(df[c].dropna().astype(str).str.upper().unique()) <= VALUES
                and df[c].notna().any()]
    if not findings:
        sys.exit("no finding columns with YES/NO/HEDGE values found")
    print(f"image id col   : {idc}\nreader col     : {rc}\nfinding cols   : {findings}")
    df = df.rename(columns={idc: "image"})
    for c in findings:
        df[c] = df[c].astype(str).str.upper().where(df[c].notna(), None)
    return df, findings, rc


def load_adjudicated(labels_dir, findings):
    parts = []
    for nm in ("validation_labels.csv", "test_labels.csv"):
        f = _find(labels_dir, nm)
        if f:
            parts.append(pd.read_csv(f))
    if not parts:
        return None
    a = pd.concat(parts, ignore_index=True)
    idc = _id_col(a)
    a = a.rename(columns={idc: "image"})
    cols = [c for c in a.columns if c in findings or c.lower() in {x.lower() for x in findings}]
    if not cols:  # adjudicated columns may be prefixed/suffixed; take YES/NO columns that are not metadata
        cols = [c for c in a.columns if c != "image"
                and set(a[c].dropna().astype(str).str.upper().unique()) <= {"YES", "NO"} and a[c].notna().any()]
    print(f"adjudicated cols: {cols}")
    a["adjudicated_pos"] = sum((a[c].astype(str).str.upper() == "YES").astype(int) for c in cols) if cols else 0
    return a[["image", "adjudicated_pos"]].drop_duplicates("image")


def build_targets(reads, findings, rc="reader"):
    rows = []
    for img, g in reads.groupby("image"):
        r = {"image": img}
        dis_any = hed_any = 0
        for f in findings:
            vals = g[f].dropna().tolist()
            vals = [v for v in vals if v in VALUES]
            dis = int(len(set(vals)) > 1) if len(vals) >= 2 else 0
            r[f"disagree_{f}"] = dis
            dis_any |= dis
            hed_any |= int("HEDGE" in vals)
        r["disagree_any"] = dis_any
        r["hedge_any"] = hed_any
        r["n_readers"] = int(g[rc].nunique()) if rc in g.columns else len(g)
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- images
def index_images(root):
    idx = {}
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                idx.setdefault(fn, os.path.join(dp, fn))
    return idx


def attach_images(t, root, size, max_images, seed=0):
    idx = index_images(root)
    print(f"images indexed : {len(idx)} under {root}")
    paths = []
    for name in t.image.astype(str):
        stem = os.path.splitext(name)[0]
        paths.append(idx.get(name) or idx.get(stem + ".png") or idx.get(stem + ".jpg"))
    t = t.assign(path=paths)
    miss = int(t.path.isna().sum())
    if miss:
        print(f"dropping {miss} images whose file was not found")
    t = t[t.path.notna()].reset_index(drop=True)
    if max_images and len(t) > max_images:
        t = t.sample(max_images, random_state=seed).reset_index(drop=True)
    arr, lap = [], []
    for p in t.path:
        im = Image.open(p).convert("L").resize((size, size))
        a = np.asarray(im, dtype=np.float32) / 255.0
        lap.append(harness.laplacian_var(a))
        arr.append((a - a.mean()) / (a.std() + 1e-6))
    t["quality_proxy"] = lap
    t["patient"] = t.image.astype(str).str.split("_").str[0]
    return t, np.stack(arr)[:, None].astype(np.float32)


# ---------------------------------------------------------------- model
class ReaderHeads(nn.Module):
    """Image encoder of the architecture (model.ImageEncoder) + one linear head per target."""

    def __init__(self, n_targets, d=32):
        super().__init__()
        self.enc = ImageEncoder(d)
        self.head = nn.Linear(d, n_targets)

    def forward(self, x):
        return self.head(self.enc(x))


def auc(y, s):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def run_seed(seed, t, X, targets, epochs, bs=32):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pats = np.array(t.patient.unique(), dtype=object)
    rng.shuffle(pats)
    cut = int(0.7 * len(pats))
    tr_m = t.patient.isin(set(pats[:cut])).values
    te_m = ~tr_m
    Y = t[targets].values.astype(np.float32)
    Xtr, Ytr = torch.from_numpy(X[tr_m]), torch.from_numpy(Y[tr_m])
    Xte = torch.from_numpy(X[te_m])
    # weight positives so rare disagreement is not ignored
    pos = Ytr.mean(0).clamp(0.02, 0.98)
    pw = ((1 - pos) / pos)
    m = ReaderHeads(len(targets))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    n = len(Xtr)
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            sel = torch.from_numpy(idx[s:s + bs])
            if len(sel) < 4:
                continue
            loss = F.binary_cross_entropy_with_logits(m(Xtr[sel]), Ytr[sel], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        S = torch.sigmoid(m(Xte)).numpy()
    te = t[te_m]
    res = {"seed": seed, "n_train": int(tr_m.sum()), "n_test": int(te_m.sum())}
    for j, tg in enumerate(targets):
        y = te[tg].values
        res[tg] = dict(
            prevalence_test=float(y.mean()),
            auc_image_model=auc(y, S[:, j]),
            auc_quality_proxy=auc(y, te.quality_proxy.values),
            auc_adjudicated_count=(auc(y, te.adjudicated_pos.values) if "adjudicated_pos" in te else float("nan")),
        )
    return res


def summarize(results, targets, path):
    rows = []
    for tg in targets:
        def agg(key):
            v = np.array([r[tg][key] for r in results], dtype=float)
            v = v[~np.isnan(v)]
            return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "---"
        prev = np.mean([r[tg]["prevalence_test"] for r in results])
        rows.append({"target": tg, "test prevalence": f"{prev:.3f}", "AUC image model": agg("auc_image_model"),
                     "AUC quality proxy": agg("auc_quality_proxy"),
                     "AUC adjudicated count": agg("auc_adjudicated_count")})
    md = ("### NIH ChestX-ray14 reader-disagreement test (test AUC, mean ± s.d. over seeds; chance 0.5)\n\n"
          + pd.DataFrame(rows).to_markdown(index=False) + "\n")
    open(path, "w").write(md)
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    t0 = time.time()

    reads, findings, rc = load_reads(a.labels_dir)
    t = build_targets(reads, findings, rc)
    adj = load_adjudicated(a.labels_dir, findings)
    if adj is not None:
        t = t.merge(adj, on="image", how="left")
    t, X = attach_images(t, a.images_dir, a.img_size, a.max_images)
    targets = ["disagree_any", "hedge_any"] + [f"disagree_{f}" for f in findings]
    targets = [c for c in targets if t[c].nunique() > 1]
    print(f"{len(t)} images, {t.patient.nunique()} patients; targets (prevalence):")
    for c in targets:
        print(f"  {c:32s} {t[c].mean():.3f}")

    os.makedirs("results", exist_ok=True)
    res = []
    for s in [int(x) for x in a.seeds.split(",")]:
        r = run_seed(s, t, X, targets, a.epochs)
        res.append(r)
        print(f"[seed {s}] " + "  ".join(f"{tg}: img {r[tg]['auc_image_model']:.3f} / q {r[tg]['auc_quality_proxy']:.3f}"
                                         for tg in targets[:2]), flush=True)
        json.dump(res, open("results/nih_reader_results.json", "w"), indent=2)
    summarize(res, targets, "results/nih_reader_summary.md")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

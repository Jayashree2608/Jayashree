"""
D3 - VinDr-CXR (Nguyen et al., 2022; PhysioNet "vindr-cxr" 1.0.0, credentialed).

Role in Table VII: the reader-reader axis. Each training image was labelled independently by
three of 17 radiologists (rad_id R1..R17), so reader disagreement is observed directly. The
labels are structured findings, not free text, so the four-cause taxonomy cannot be scored;
what is checked is the reader side of the model:

  (a) How much do readers disagree? Per finding: prevalence, non-unanimous rate, Fleiss kappa.
  (b) Are reader traits stable? View 3 of the EM assumes a reader's tendency is a fixed trait.
      Dawid-Skene EM per finding gives each reader's sensitivity and specificity; a split-half
      test (each reader's images split at random into two halves) gives the correlation of a
      reader's positive-call rates across halves, over findings. A trait that is real should
      replicate across halves.
  (c) Do readers agree on where? For local labels called by >= 2 readers on the same image,
      the mean best-match IoU of their boxes.
  (d) Optional, needs the DICOM images: can a cross-fitted image CNN (the input side of the
      anticipation head) predict which images readers will disagree on, compared with two
      reference predictors that need no image model (number of findings the Dawid-Skene
      posterior calls present; image sharpness)?

FILES under --vindr-dir: annotations/image_labels_train.csv (image_id, rad_ID, 28 label columns),
annotations/annotations_train.csv (image_id, rad_id, class_name, x_min, y_min, x_max, y_max);
images train/<image_id>.dicom (optional). The Kaggle competition copy (train.csv with image_id,
class_name, rad_id, boxes) is accepted for (a)-(c) when image_labels_train.csv is absent.

USAGE  python3 vindr_run.py --vindr-dir <dir> [--with-images] [--max-images 3000] [--seeds 0,1,2]
"""
import argparse
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C


def load_labels(root):
    print("VinDr-CXR files:")
    f = C.find_file(root, "image_labels_train.csv")
    boxes_f = C.find_file(root, "annotations_train.csv") or C.find_file(root, "train.csv")
    boxes = C.read_table(boxes_f) if boxes_f else None
    if f:
        lab = C.read_table(f)
        idc, rc = C.col(lab, "image_id"), C.col(lab, "rad_id", "rad_ID")
        finds = [c for c in lab.columns if c not in (idc, rc)]
        lab = lab.rename(columns={idc: "image_id", rc: "rad_id"})
        lab[finds] = (lab[finds].fillna(0).astype(float) > 0).astype(int)
    elif boxes is not None:
        print("  image_labels_train.csv not found: image-level labels derived from box annotations")
        b = boxes.rename(columns={C.col(boxes, "image_id"): "image_id", C.col(boxes, "rad_id", "rad_ID"): "rad_id",
                                  C.col(boxes, "class_name"): "class_name"})
        lab = pd.crosstab([b.image_id, b.rad_id], b.class_name).clip(upper=1).reset_index()
        finds = [c for c in lab.columns if c not in ("image_id", "rad_id")]
    else:
        sys.exit("neither image_labels_train.csv nor annotations_train.csv/train.csv found")
    finds = [f for f in finds if f.strip().lower() != "no finding"]   # derived label, not a finding
    nr = lab.groupby("image_id").rad_id.nunique()
    print(f"  images {lab.image_id.nunique()}, readers {lab.rad_id.nunique()}, reads per image: "
          f"{nr.value_counts().sort_index().to_dict()}")
    return lab, finds, boxes


def agreement(lab, finds):
    rows = []
    g = lab.groupby("image_id")
    for f in finds:
        pos = g[f].sum(); n = g[f].count()
        k = C.fleiss_kappa(np.column_stack([pos, n - pos]))
        rows.append(dict(finding=f, prevalence_any_reader=float((pos > 0).mean()),
                         prevalence_majority=float((pos > n / 2).mean()),
                         non_unanimous=float(((pos > 0) & (pos < n)).mean()), fleiss_kappa=k))
    return pd.DataFrame(rows).sort_values("prevalence_any_reader", ascending=False)


def reader_traits(lab, finds, min_reads=200, seed=0):
    readers = sorted(lab.rad_id.unique())
    ri = {r: i for i, r in enumerate(readers)}
    imgs = sorted(lab.image_id.unique()); ii = {m: i for i, m in enumerate(imgs)}
    sens, spec = {}, {}
    disagree = {}
    for f in finds:
        L = -np.ones((len(imgs), len(readers)), dtype=int)
        L[lab.image_id.map(ii).values, lab.rad_id.map(ri).values] = lab[f].values
        if L.max() < 1 or (L == 1).sum() < 20:
            continue
        T, prior, conf = C.dawid_skene(L)
        sens[f] = conf[:, 1, 1]; spec[f] = conf[:, 0, 0]
        disagree[f] = T[:, 1]
    nreads = lab.groupby("rad_id").size().reindex(readers).values
    keep = nreads >= min_reads
    S = pd.DataFrame(sens, index=readers)[keep]; P = pd.DataFrame(spec, index=readers)[keep]
    # split-half stability of positive-call rates
    rng = np.random.default_rng(seed)
    lab = lab.copy(); lab["half"] = rng.integers(0, 2, len(lab))
    rates = lab.groupby(["rad_id", "half"])[finds].mean()
    rs = []
    for f in finds:
        r = rates[f].unstack("half").dropna()
        r = r[r.index.isin(S.index)]
        if len(r) >= 4 and r.std().min() > 0:
            rs.append(np.corrcoef(r[0], r[1])[0, 1])
    res = dict(n_readers_kept=int(keep.sum()), min_reads=min_reads,
               split_half_r_median=float(np.median(rs)) if rs else np.nan, split_half_n_findings=len(rs),
               sens_range_median_over_findings=float((S.max() - S.min()).median()) if len(S) else np.nan,
               spec_range_median_over_findings=float((P.max() - P.min()).median()) if len(P) else np.nan)
    post = pd.DataFrame(disagree, index=imgs)
    return res, S, P, post


def box_iou(boxes):
    if boxes is None:
        return pd.DataFrame()
    b = boxes.rename(columns={C.col(boxes, "rad_id", "rad_ID"): "rad_id"}).dropna(subset=["x_min"])
    b = b[b.class_name != "No finding"]

    def iou(p, q):
        ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = ix * iy
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / u if u > 0 else 0.0
    rows = []
    for (img, cls), g in b.groupby(["image_id", "class_name"]):
        per = {r: g2[["x_min", "y_min", "x_max", "y_max"]].values for r, g2 in g.groupby("rad_id")}
        if len(per) < 2:
            continue
        vals = [np.mean([max(iou(p, q) for q in per[r2]) for p in per[r1]]) for r1, r2 in combinations(per, 2)]
        rows.append((cls, float(np.mean(vals))))
    d = pd.DataFrame(rows, columns=["class_name", "iou"])
    return d.groupby("class_name").iou.agg(["count", "mean", "median"]).reset_index().sort_values("count", ascending=False)


def image_probe(lab, finds, post, root, seeds, max_images, epochs, harness):
    idx = C.index_images(root, exts=(".dicom", ".dcm", ".png", ".jpg"))
    g = lab.groupby("image_id")
    pos, n = g[finds].sum(), g[finds].count()
    y = ((pos > 0) & (pos < n)).any(axis=1).astype(int)
    ids = [m for m in y.index if m in idx][:max_images]
    print(f"  images found: {len(ids)}")
    if len(ids) < 100:
        print("  fewer than 100 images: (d) skipped"); return None
    imgs = np.stack([C.load_gray(idx[m], 32) for m in ids]); yy = y.loc[ids].values.astype(np.float32)
    q = np.array([C.laplacian_var(x) for x in imgs])
    ds_count = (post.loc[ids] > 0.5).sum(axis=1).values
    from sklearn.model_selection import StratifiedKFold
    out = []
    for s in seeds:
        sc = np.zeros(len(ids), dtype=np.float32)
        for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(imgs, yy)):
            sc[te] = harness._score_cnn(harness._train_cnn(imgs[tr], yy[tr], epochs, s * 100 + i), imgs[te])
        out.append(C.auc(yy, sc))
    return dict(n=len(ids), disagree_rate=float(yy.mean()), auc_image_cnn_mean=float(np.mean(out)),
                auc_image_cnn_sd=float(np.std(out)), auc_ds_positive_count=C.auc(yy, ds_count),
                auc_sharpness=max(C.auc(yy, q), C.auc(yy, -q)), seeds=list(seeds))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr-dir", required=True)
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--min-reads", type=int, default=200)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lab, finds, boxes = load_labels(a.vindr_dir)
    A = agreement(lab, finds)
    T, S, P, post = reader_traits(lab, finds, a.min_reads)
    B = box_iou(boxes)
    res = dict(n_images=int(lab.image_id.nunique()), n_readers=int(lab.rad_id.nunique()), reader_traits=T)
    if a.with_images:
        import harness_current as H
        res["image_probe"] = image_probe(lab, finds, post, a.vindr_dir, [int(s) for s in a.seeds.split(",")],
                                         a.max_images, a.epochs, H)
    A.to_csv(os.path.join(a.out, "vindr_agreement.csv"), index=False)
    S.to_csv(os.path.join(a.out, "vindr_reader_sensitivity.csv")); P.to_csv(os.path.join(a.out, "vindr_reader_specificity.csv"))
    C.save_json(res, os.path.join(a.out, "vindr_results.json"))
    L = [f"# D3 VinDr-CXR: reader-reader axis ({res['n_images']} images, {res['n_readers']} readers)", "",
         "**(a) Agreement per finding** (three independent reads per image)", ""]
    L += C.md_table(A)
    L += ["", "**(b) Are reader traits stable?**", "",
          f"* readers with >= {T['min_reads']} reads: {T['n_readers_kept']}",
          f"* split-half correlation of per-reader positive-call rate, median over {T['split_half_n_findings']} findings: "
          f"{T['split_half_r_median']:.3f}",
          f"* range of per-reader sensitivity (Dawid-Skene), median over findings: {T['sens_range_median_over_findings']:.3f}; "
          f"specificity: {T['spec_range_median_over_findings']:.3f}"]
    if len(B):
        L += ["", "**(c) Localisation agreement** (mean best-match box IoU between readers who both called the finding)", ""]
        L += C.md_table(B)
    if res.get("image_probe"):
        p = res["image_probe"]
        L += ["", "**(d) Does the image predict reader disagreement?** (AUC, 5-fold cross-fitted)", "",
              f"* n = {p['n']}, disagreement rate {p['disagree_rate']:.3f}",
              f"* image CNN {p['auc_image_cnn_mean']:.3f} +/- {p['auc_image_cnn_sd']:.3f}; "
              f"reference: Dawid-Skene positive count {p['auc_ds_positive_count']:.3f}; sharpness {p['auc_sharpness']:.3f}"]
    C.save_md(L, os.path.join(a.out, "vindr_summary.md"))
    return res


if __name__ == "__main__":
    main()

"""
E2 - Reader-reader axis without PhysioNet: NIH ChestX-ray14 with the Google expert labels
(Majkowska et al., Radiology 2019). Free request form on the Google Cloud NIH chest X-ray page;
the same files Part C2 uses (four_findings_expert_labels/individual_readers.csv, or the
all_findings_expert_labels set). Each image has independent reads by three radiologists
answering YES / NO / HEDGE per finding.

Open replacement for VinDr-CXR in Table VII. It runs the VinDr analysis (vindr_run) on these reads
and adds the one thing VinDr cannot give: an explicit HEDGE answer per reader. That allows View 3's
assumption to be tested directly: is a reader's hedging rate a stable trait? (split-half test of
each reader's hedge rate, and between-reader spread). Part C2 already tests the image side
(does the image predict disagreement); this step needs no images.

USAGE  python3 nih_traits_run.py --labels-dir <four_findings_expert_labels> [--min-reads 100]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C
import vindr_run as VD

VALUES = {"YES", "NO", "HEDGE"}


def load(root):
    f = C.find_file(root, "individual_readers.csv") or C.find_file(root, "test_individual_readers.csv")
    if f is None:
        sys.exit("individual_readers.csv not found under " + root)
    print("NIH expert reads:")
    d = C.read_table(f)
    idc = next((c for c in d.columns if "image" in c.lower()), d.columns[0])
    rc = next((c for c in d.columns if "reader" in c.lower()), None)
    if rc is None:
        sys.exit("no reader column (name containing 'reader')")
    finds = [c for c in d.columns if c not in (idc, rc) and d[c].notna().any()
             and set(d[c].dropna().astype(str).str.upper().unique()) <= VALUES]
    print(f"  image col {idc}; reader col {rc}; findings {finds}")
    d = d.rename(columns={idc: "image_id", rc: "rad_id"})
    for c in finds:
        d[c] = d[c].astype(str).str.upper().where(d[c].notna(), None)
    return d, finds


def hedge_traits(d, finds, min_reads, seed=0):
    rng = np.random.default_rng(seed)
    h = d[["image_id", "rad_id"]].copy()
    h["hedge_any"] = (d[finds] == "HEDGE").any(axis=1).astype(int)
    for f in finds:
        h[f] = (d[f] == "HEDGE").astype(int)
    n = h.groupby("rad_id").size()
    keep = n[n >= min_reads].index
    h = h[h.rad_id.isin(keep)].copy()
    h["half"] = rng.integers(0, 2, len(h))
    r = h.groupby(["rad_id", "half"]).hedge_any.mean().unstack("half").dropna()
    rate = h.groupby("rad_id").hedge_any.mean()
    # does the reader explain hedging beyond the image? logistic with image fixed effects is too
    # large; use the share of hedges that come from images where another reader did not hedge
    g = h.groupby("image_id").hedge_any
    others = (g.transform("sum") - h.hedge_any) / (g.transform("count") - 1).clip(lower=1)
    lone = float(((h.hedge_any == 1) & (others == 0)).sum() / max(h.hedge_any.sum(), 1))
    return dict(n_readers_kept=int(len(keep)), min_reads=min_reads,
                hedge_rate_min=float(rate.min()) if len(rate) else np.nan,
                hedge_rate_median=float(rate.median()) if len(rate) else np.nan,
                hedge_rate_max=float(rate.max()) if len(rate) else np.nan,
                split_half_r=float(np.corrcoef(r[0], r[1])[0, 1]) if len(r) >= 3 else np.nan,
                share_of_hedges_where_no_other_reader_hedged=lone), rate


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--min-reads", type=int, default=100)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d, finds = load(a.labels_dir)
    pos = d[["image_id", "rad_id"]].copy()
    for f in finds:
        pos[f] = (d[f] == "YES").astype(int)
    A = VD.agreement(pos, finds)
    T, S, P, post = VD.reader_traits(pos, finds, a.min_reads)
    H, rate = hedge_traits(d, finds, a.min_reads)
    res = dict(n_images=int(d.image_id.nunique()), n_readers=int(d.rad_id.nunique()), reader_traits_positive=T,
               hedge_traits=H)
    A.to_csv(os.path.join(a.out, "nih_traits_agreement.csv"), index=False)
    rate.rename("hedge_rate").to_csv(os.path.join(a.out, "nih_reader_hedge_rates.csv"))
    C.save_json(res, os.path.join(a.out, "nih_traits_results.json"))
    L = [f"# E2 NIH ChestX-ray14 expert reads: reader-reader axis ({res['n_images']} images, {res['n_readers']} readers)", "",
         "**(a) Agreement per finding** (YES vs NO/HEDGE)", ""] + C.md_table(A)
    L += ["", "**(b) Are reader traits stable?** (positive calls, Dawid-Skene)", "",
          f"* readers with >= {T['min_reads']} reads: {T['n_readers_kept']}; split-half r of positive-call rate, "
          f"median over findings: {T['split_half_r_median']:.3f}; sensitivity range {T['sens_range_median_over_findings']:.3f}, "
          f"specificity range {T['spec_range_median_over_findings']:.3f}",
          "", "**(c) Is hedging a reader trait?** (the assumption behind View 3)", "",
          f"* per-reader rate of any HEDGE answer: min {H['hedge_rate_min']:.3f}, median {H['hedge_rate_median']:.3f}, "
          f"max {H['hedge_rate_max']:.3f} ({H['n_readers_kept']} readers)",
          f"* split-half correlation of the hedge rate: {H['split_half_r']:.3f}",
          f"* share of hedges on images where no other reader hedged: {H['share_of_hedges_where_no_other_reader_hedged']:.3f}"]
    C.save_md(L, os.path.join(a.out, "nih_traits_summary.md"))
    return res


if __name__ == "__main__":
    main()

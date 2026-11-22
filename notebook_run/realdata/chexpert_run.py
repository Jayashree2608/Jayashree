"""
D5 - CheXpert (Irvin et al., 2019; Stanford AIMI, research-use registration).

Role in Table VII: secondary, cross-institution benchmark with uncertainty labels. CheXpert's
public release has images and labeler outputs (1 / 0 / -1 uncertain / blank) but no reports;
CheXpert Plus (2024) adds the reports. Checks:

  (a) Concomitant link at a second institution: is the uncertain (-1) rate higher on AP than on
      PA films, per finding and pooled with finding fixed effects (same model as D2-a)? If a
      MIMIC result (mimic_results.json) exists in --out, both are printed side by side.
  (b) Optional, needs images: does a cross-fitted image CNN predict that the labeler will mark
      the target finding uncertain (among studies that mention it), beyond AP/PA and sharpness?
      This is the image-side claim behind the anticipation head, on uncertainty instead of
      reader disagreement.
  (c) Optional, needs a CheXpert Plus report table: agreement of the lexical View 1 hedge flag
      with the labeler's uncertain label, the same check as D1 at another institution.

FILES under --chexpert-dir: train.csv and valid.csv (CheXpert-v1.0 or -v1.0-small: Path, Sex, Age,
Frontal/Lateral, AP/PA, 14 findings); images at the relative Path (optional); CheXpert Plus
report CSV with section_findings / section_impression and path_to_image (optional).

USAGE  python3 chexpert_run.py --chexpert-dir <dir> [--with-images] [--target "Pleural Effusion"]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C
from mimic_run import concomitant

FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
            "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices"]


def load(root):
    print("CheXpert files:")
    f = C.require(C.find_file(root, "train.csv"), "CheXpert train.csv")
    d = C.read_table(f)
    pc, fl, ap = C.col(d, "Path"), C.col(d, "Frontal/Lateral"), C.col(d, "AP/PA")
    d = d[d[fl] == "Frontal"].copy()
    d["ViewPosition"] = d[ap]
    d["subject_id"] = d[pc].str.extract(r"(patient\d+)")[0]
    d["study"] = d[pc].str.extract(r"(patient\d+/study\d+)")[0]
    d = d.drop_duplicates("study")
    fnd = [x for x in FINDINGS if x in d.columns]
    print(f"  frontal studies {len(d)}, patients {d.subject_id.nunique()}, AP/PA counts {d.ViewPosition.value_counts().to_dict()}")
    return d[d.ViewPosition.isin(["AP", "PA"])], fnd, os.path.dirname(f)


def resolve(root, rel):
    for base in (root, os.path.dirname(root), os.path.dirname(os.path.dirname(root))):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def image_probe(d, target, root, seeds, n_max, epochs, harness):
    m = d[d[target].notna()].copy()
    m["path_abs"] = [resolve(root, p) for p in m.Path]
    m = m[m.path_abs.notna()].head(n_max)
    print(f"  images found for {len(m)} studies mentioning {target}")
    if len(m) < 100:
        print("  fewer than 100: (b) skipped"); return None
    imgs = np.stack([C.load_gray(p, 32) for p in m.path_abs]); y = (m[target] == -1).astype(np.float32).values
    q = np.array([C.laplacian_var(x) for x in imgs]); apf = (m.ViewPosition == "AP").astype(float).values
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    a_cnn, a_ref = [], []
    for s in seeds:
        sc = np.zeros(len(m)); ref = np.zeros(len(m))
        for i, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(imgs, y)):
            sc[te] = harness._score_cnn(harness._train_cnn(imgs[tr], y[tr], epochs, s * 100 + i), imgs[te])
            Z = np.column_stack([apf, (q - q.mean()) / (q.std() + 1e-8)])
            ref[te] = LogisticRegression().fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        a_cnn.append(C.auc(y, sc)); a_ref.append(C.auc(y, ref))
    return dict(n=len(m), uncertain_rate=float(y.mean()), auc_image_cnn_mean=float(np.mean(a_cnn)),
                auc_image_cnn_sd=float(np.std(a_cnn)), auc_ap_plus_sharpness=float(np.mean(a_ref)))


def plus_reports(root, d, target):
    f = None
    for p in C.find_files(root, r"\.csv(\.gz)?$"):
        try:
            cols = {c.lower() for c in pd.read_csv(p, nrows=0).columns}
        except Exception:
            continue
        if {"section_impression", "path_to_image"} <= cols or {"section_findings", "path_to_image"} <= cols:
            f = p; break
    if f is None:
        return None
    r = C.read_table(f)
    r["study"] = r[C.col(r, "path_to_image")].str.extract(r"(patient\d+/study\d+)")[0]
    txt = (r.get("section_findings", "").fillna("").astype(str) + " " + r.get("section_impression", "").fillna("").astype(str))
    r["txt"] = txt.str.strip()
    r = r.drop_duplicates("study").merge(d[["study", target]], on="study")
    r = r[r[target].notna()]
    terms = C.FINDING_TERMS.get(target)
    lex = np.array([C.lexical_v1(t, terms) for t in r.txt])
    y = (r[target] == -1).astype(int).values
    return dict(n=len(r), labeler_uncertain_rate=float(y.mean()), lexical_hedge_rate=float(lex[:, 0].mean()),
                kappa_hedge=C.cohen_kappa(y, lex[:, 0]),
                kappa_mention_vs_positive=C.cohen_kappa((r[target] == 1).astype(int).values, lex[:, 1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--chexpert-dir", required=True)
    ap.add_argument("--target", default="Pleural Effusion")
    ap.add_argument("--with-images", action="store_true")
    ap.add_argument("--max-images", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d, fnd, root = load(a.chexpert_dir)
    per, pool = concomitant(d, fnd)
    res = dict(n_studies=len(d), n_patients=int(d.subject_id.nunique()), concomitant_pooled=pool)
    if a.with_images:
        import harness_current as H
        res["image_probe"] = image_probe(d, a.target, root, [int(s) for s in a.seeds.split(",")], a.max_images, a.epochs, H)
    res["plus_reports"] = plus_reports(a.chexpert_dir, d, a.target)
    per.to_csv(os.path.join(a.out, "chexpert_concomitant.csv"), index=False)
    C.save_json(res, os.path.join(a.out, "chexpert_results.json"))
    L = [f"# D5 CheXpert ({len(d)} frontal studies, {res['n_patients']} patients)", "",
         "**(a) Uncertain label on AP vs PA**, odds ratio among studies that mention the finding", ""]
    rows = []
    if pool:
        rows.append(dict(dataset="CheXpert", **{k: pool[k] for k in ("odds_ratio", "ci_low", "ci_high", "p", "rate_AP", "rate_PA", "n")}))
    mj = os.path.join(a.out, "mimic_results.json")
    if os.path.exists(mj):
        mp = json.load(open(mj)).get("concomitant_pooled")
        if mp:
            rows.append(dict(dataset="MIMIC-CXR", **{k: mp[k] for k in ("odds_ratio", "ci_low", "ci_high", "p", "rate_AP", "rate_PA", "n")}))
    if rows:
        L += C.md_table(pd.DataFrame(rows)) + [""]
    if len(per):
        L += C.md_table(per[["finding", "n_mentioned", "uncertain_rate", "odds_ratio", "ci_low", "ci_high", "p"]])
    if res.get("image_probe"):
        p = res["image_probe"]
        L += ["", f"**(b) Does the image predict an uncertain label for {a.target}?** (AUC, 5-fold)", "",
              f"* n = {p['n']}, uncertain rate {p['uncertain_rate']:.3f}; image CNN {p['auc_image_cnn_mean']:.3f} "
              f"+/- {p['auc_image_cnn_sd']:.3f}; reference AP/PA + sharpness {p['auc_ap_plus_sharpness']:.3f}"]
    if res.get("plus_reports"):
        p = res["plus_reports"]
        L += ["", "**(c) CheXpert Plus reports: lexical View 1 vs labeler**", "",
              f"* n = {p['n']}; kappa hedge vs uncertain {p['kappa_hedge']:.3f}; kappa mention vs positive {p['kappa_mention_vs_positive']:.3f}"]
    C.save_md(L, os.path.join(a.out, "chexpert_summary.md"))
    return res


if __name__ == "__main__":
    main()

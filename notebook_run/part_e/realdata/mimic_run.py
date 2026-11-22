"""
D2 - MIMIC-CXR (Johnson et al., 2019) with MIMIC-CXR-JPG; PhysioNet credentialed access.

Role in Table VII: primary source for the image-report and current-prior axes. This runner
executes the real-data protocol of Section III-C (formerly "implemented, not executed"):

  (a) Concomitant link, labels only. Does the CheXpert labeler's "uncertain" (-1) rate depend
      on acquisition (AP vs PA, and image sharpness when images are given)? The model
      assumes the class prior depends on quality (Eq. 17); this checks that assumption on
      real reports. Logistic regression per finding and pooled with finding fixed effects.
  (b) Current-prior axis, labels only. Consecutive studies of the same patient (ordered by
      StudyDate/StudyTime): how often does a finding flip between present and absent, and
      how often is an uncertain label at t resolved to a definite one at t+1? A hedge that is
      later resolved is consistent with genuine image ambiguity at time t.
  (c) The latent-class EM of Part A on real views (needs images; optional RadGraph and
      provider list), with label-free naming. No cause labels exist, so only face-validity
      checks are reported: does the component named "hedge" concentrate on studies the
      CheXpert labeler marked uncertain for the target finding; is the class mix stable
      across quality terciles; how overconfident are the posteriors.
  (d) Writes pairs.csv (filename, patientid, txt, y_target) so the architecture runner of
      Part C1 can be run on MIMIC with --mode pairs.

FILES (MIMIC-CXR-JPG 2.0.0/2.1.0 and MIMIC-CXR 2.0.0/2.1.0, any layout under the dirs given)
  mimic-cxr-2.0.0-metadata.csv(.gz)  dicom_id, subject_id, study_id, ViewPosition, StudyDate, StudyTime
  mimic-cxr-2.0.0-chexpert.csv(.gz)  subject_id, study_id, 14 findings in {1, 0, -1, blank}
  files/pXX/pXXXXXXXX/sXXXXXXXX/<dicom_id>.jpg    (images, optional: needed for (c))
  files/pXX/pXXXXXXXX/sXXXXXXXX.txt               (reports from mimic-cxr-reports.zip)
  cxr-provider-list.csv(.gz)  (MIMIC-CXR 2.1.0, optional: reader axis View 3)
  radgraph_v1_mimic.csv       (written by D1 from MIMIC-CXR_graphs.json, optional)

USAGE
  python3 mimic_run.py --jpg-dir <mimic-cxr-jpg> --reports-dir <mimic-cxr> [--provider-csv f]
                       [--radgraph-features f] [--target "Pleural Effusion"] [--n-cases 4000]
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C

FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices"]
CLASS_NAMES = ["faithful", "omission", "hedge", "style"]


# ------------------------------------------------------------------ loading
def load_tables(jpg_dir):
    print("MIMIC-CXR-JPG tables:")
    meta = C.read_table(C.require(C.find_file(jpg_dir, "mimic-cxr-2.0.0-metadata.csv"), "metadata csv"))
    chx = C.read_table(C.require(C.find_file(jpg_dir, "mimic-cxr-2.0.0-chexpert.csv"), "chexpert csv"))
    for c in ("dicom_id", "subject_id", "study_id", "ViewPosition"):
        C.col(meta, c)
    fnd = [f for f in FINDINGS if f in chx.columns]
    print(f"  findings present: {fnd}")
    front = meta[meta.ViewPosition.isin(["PA", "AP"])].copy()
    if "StudyDate" in front.columns:
        t = front.get("StudyTime", pd.Series(0, index=front.index)).fillna(0).astype(float)
        front["order"] = front.StudyDate.astype(float) * 1e6 + t
    else:
        front["order"] = front.study_id
    front = front.sort_values(["study_id", "ViewPosition"], ascending=[True, False])  # PA before AP
    st = front.drop_duplicates("study_id")[["subject_id", "study_id", "dicom_id", "ViewPosition", "order"]]
    st = st.merge(chx[["study_id"] + fnd], on="study_id", how="inner")
    print(f"  frontal studies with CheXpert labels: {len(st)} ({st.subject_id.nunique()} patients)")
    return st, fnd


def attach_reports(st, reports_dir):
    idx = {}
    for p in C.find_files(reports_dir, r"^s\d+\.txt$"):
        idx[int(os.path.basename(p)[1:-4])] = p
    print(f"  report files found: {len(idx)}")
    st = st[st.study_id.isin(idx)].copy()
    st["txt"] = [C.report_sections(open(idx[s], errors="ignore").read()) for s in st.study_id]
    return st[st.txt.str.len() > 0]


def attach_provider(st, path):
    if not path or not os.path.exists(path):
        print("  provider list not given: View 3 (reader) will be omitted")
        st["reader_id"] = np.nan
        return st
    p = C.read_table(path)
    sid = C.col(p, "study_id")
    pc = next((c for c in p.columns if re.search(r"provider|attending|author|reader", c, re.I)), None)
    if pc is None:
        print("  no provider column found: View 3 omitted"); st["reader_id"] = np.nan; return st
    print(f"  reader column: {pc}")
    return st.merge(p[[sid, pc]].rename(columns={sid: "study_id", pc: "reader_id"}).drop_duplicates("study_id"),
                    on="study_id", how="left")


def attach_images(st, jpg_dir, size=32):
    idx = C.index_images(jpg_dir, exts=(".jpg", ".jpeg", ".png"))
    keep = st.dicom_id.isin(idx)
    print(f"  images found for {keep.sum()} / {len(st)} sampled studies")
    st = st[keep].copy()
    st["img"] = [C.load_gray(idx[d], size) for d in st.dicom_id]
    st["q_meas"] = st.img.map(C.laplacian_var)
    st["img_path"] = [idx[d] for d in st.dicom_id]
    return st


# ------------------------------------------------------------------ (a) concomitant link
def concomitant(st, fnd, quality=None):
    rows, pooled = [], []
    for f in fnd:
        if f == "Support Devices":
            continue
        m = st[f].notna()                       # finding mentioned at all
        y = (st.loc[m, f] == -1).astype(int).values
        if m.sum() < 30 or y.sum() < 5 or y.sum() == len(y):
            continue
        X = [(st.loc[m, "ViewPosition"] == "AP").astype(float).values]
        names = ["AP_vs_PA"]
        if quality is not None:
            q = quality[m.values]
            X.append((q - q.mean()) / (q.std() + 1e-8)); names.append("sharpness_z")
        o = C.logit_or(y, np.column_stack(X), names)
        o.insert(0, "finding", f); o.insert(1, "n_mentioned", int(m.sum())); o.insert(2, "uncertain_rate", y.mean())
        rows.append(o)
        pooled.append(pd.DataFrame({"y": y, "ap": X[0], "finding": f}))
    per = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    pool = None
    if pooled:
        P = pd.concat(pooled, ignore_index=True)
        D = pd.get_dummies(P.finding, drop_first=True).astype(float)
        o = C.logit_or(P.y.values, np.column_stack([P.ap.values, D.values]), ["AP_vs_PA"] + list(D.columns))
        pool = o.iloc[0].to_dict()
        pool["n"] = len(P)
        pool["rate_AP"] = P.y[P.ap == 1].mean(); pool["rate_PA"] = P.y[P.ap == 0].mean()
    return per, pool


# ------------------------------------------------------------------ (b) current-prior axis
def longitudinal(st, fnd):
    s = st.sort_values(["subject_id", "order"])
    rows = []
    for f in fnd:
        a = s[f].values; sub = s.subject_id.values
        same = sub[1:] == sub[:-1]
        prev, cur = a[:-1][same], a[1:][same]
        both_def = np.isin(prev, [0, 1]) & np.isin(cur, [0, 1])
        unc = prev == -1
        rows.append(dict(finding=f, n_pairs=int(same.sum()),
                         flip_rate_definite=float((prev[both_def] != cur[both_def]).mean()) if both_def.any() else np.nan,
                         n_uncertain_at_t=int(unc.sum()),
                         resolved_present=float((cur[unc] == 1).mean()) if unc.any() else np.nan,
                         resolved_absent_or_unmentioned=float(((cur[unc] == 0) | pd.isna(cur[unc])).mean()) if unc.any() else np.nan,
                         still_uncertain=float((cur[unc] == -1).mean()) if unc.any() else np.nan))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ (c) EM on real views
def em_views(st, target, seed, rg_path, cnn_epochs, harness):
    df = st.copy().reset_index(drop=True)
    if rg_path and os.path.exists(rg_path):
        rg = pd.read_csv(rg_path, dtype={"study_id": str})
        df = df.assign(study_id=df.study_id.astype(str)).merge(rg, on="study_id", how="left")
        ok = df.rg_hedge.notna()
        print(f"  View 1 from RadGraph inference for {ok.sum()} / {len(df)} studies (lexical for the rest)")
    else:
        ok = pd.Series(False, index=df.index)
        print("  View 1: lexical proxy (no RadGraph features given)")
    terms = C.TERMS_EN.get(target)
    lex = np.array([C.lexical_v1(t, terms) for t in df.txt])
    df["view1_hedge"] = np.where(ok, df.get("rg_hedge", 0), lex[:, 0])
    df["view1_mention"] = np.where(ok, df.get("rg_mention", 0), lex[:, 1])
    feat = ["view1_hedge", "view1_mention"]
    # View 2: cross-fitted CNN for the target finding, residualised on quality
    df["true_finding"] = (df[target] == 1).astype(np.float32)   # labeler label; used only to train V2
    v2, _ = harness.build_view2_crossfit(df, seed, epochs=cnn_epochs)
    from sklearn.linear_model import LinearRegression
    df["view2_raw"] = v2
    df["view2_resid"] = v2 - LinearRegression().fit(df[["q_meas"]], v2).predict(df[["q_meas"]])
    feat.append("view2_resid")
    # View 3: provider leave-one-out rates of observed hedge and non-mention
    if df.reader_id.notna().sum() > 0:
        cnt = df.groupby("reader_id").reader_id.transform("count")
        valid = df.reader_id.notna() & (cnt >= harness.MIN_CASES_PER_READER)
        for src, out in (("view1_hedge", "view3_hedge_obs"), ("view1_mention", "view3_nomention_obs")):
            flag = df[src] if src == "view1_hedge" else 1 - df[src]
            s_ = flag.groupby(df.reader_id).transform("sum"); c_ = flag.groupby(df.reader_id).transform("count")
            df[out] = (s_ - flag) / (c_ - 1).clip(lower=1)
            df.loc[~valid, out] = df.loc[valid, out].mean() if valid.any() else flag.mean()
        feat += ["view3_hedge_obs", "view3_nomention_obs"]
        print(f"  View 3: {valid.sum()} studies with a provider having >= {harness.MIN_CASES_PER_READER} studies")
    return df, feat


def em_face_validity(df, feat, target, seed, harness, n_restarts):
    pred_raw, resp, ll = harness.fit_em(df, feat, seed, K=4, n_restarts=n_restarts)
    mapping = harness.anchor_mapping(resp, df.view1_hedge.values, df.view1_mention.values)
    inv = {v: k for k, v in mapping.items()}
    named = np.array([inv[k] for k in pred_raw])
    post = resp[:, [mapping[c] for c in range(4)]]          # columns in CLASS_NAMES order
    unc = (df[target] == -1).values
    q3 = pd.qcut(df.q_meas.rank(method="first"), 3, labels=["low", "mid", "high"])
    mix = pd.crosstab(q3, pd.Categorical([CLASS_NAMES[k] for k in named], CLASS_NAMES), normalize="index")
    res = dict(
        n=len(df), features=feat, log_lik=float(ll),
        class_share={CLASS_NAMES[k]: float((named == k).mean()) for k in range(4)},
        labeler_uncertain_rate=float(unc.mean()),
        p_uncertain_given_named_hedge=float(unc[named == 2].mean()) if (named == 2).any() else np.nan,
        p_uncertain_given_other=float(unc[named != 2].mean()) if (named != 2).any() else np.nan,
        auc_hedge_posterior_for_labeler_uncertain=C.auc(unc, post[:, 2]),
        auc_omission_posterior_for_labeler_positive_unmentioned=C.auc(
            ((df[target] == 1) & (df.view1_mention == 0)).values, post[:, 1]),
        frac_conf_gt_099=float((resp.max(1) > 0.99).mean()),
        mix_by_quality_tercile=mix.round(3).to_dict(),
        corr_faithful_posterior_quality=float(np.corrcoef(post[:, 0], df.q_meas)[0, 1]),
    )
    return res, named, post


def analyse(st, fnd, target, out, prefix, title, load_images=None, seeds="0,1,2", n_cases=4000,
            radgraph_features=None, cnn_epochs=8, n_restarts=6, pairs_basename=True, notes=None):
    """Steps (a)-(d) on any study table with the columns subject_id, study_id, ViewPosition, order, txt,
    reader_id (may be NaN) and CheXpert-style finding columns (1 / 0 / -1 / NaN).
    load_images(samp) must return samp restricted to studies with an image, with img, q_meas, img_path."""
    import harness_current as H
    os.makedirs(out, exist_ok=True)
    res = {"n_studies_with_report": len(st), "n_patients": int(st.subject_id.nunique()), "target": target}
    if notes:
        res["notes"] = notes
    print("(a) concomitant link, all studies, labels only")
    per, pool = concomitant(st, fnd)
    res["concomitant_pooled"] = pool
    print("(b) current-prior axis")
    lon = longitudinal(st, fnd)
    em = []
    if load_images is not None:
        rng = np.random.default_rng(0)
        prank = dict(zip(st.subject_id.unique(), rng.permutation(st.subject_id.nunique())))
        samp = st.assign(_r=st.subject_id.map(prank)).sort_values(["_r", "order"]).drop(columns="_r")
        samp = samp.head(n_cases * 2)                 # whole patients, random order
        samp = load_images(samp).head(n_cases)
        if len(samp) >= 100:
            print("(a') concomitant link with image sharpness, image sample")
            per_q, pool_q = concomitant(samp, fnd, quality=samp.q_meas.values)
            res["concomitant_pooled_image_sample"] = pool_q
            per_q.to_csv(os.path.join(out, f"{prefix}_concomitant_with_sharpness.csv"), index=False)
            print("(c) latent-class EM on real views")
            for s in [int(x) for x in str(seeds).split(",")]:
                df, feat = em_views(samp, target, s, radgraph_features, cnn_epochs, H)
                r, named, post = em_face_validity(df, feat, target, s, H, n_restarts)
                r["seed"] = s; em.append(r)
                print(f"  seed {s}: shares {r['class_share']}; P(-1|hedge)={r['p_uncertain_given_named_hedge']:.3f} "
                      f"vs {r['p_uncertain_given_other']:.3f}; AUC {r['auc_hedge_posterior_for_labeler_uncertain']:.3f}")
            res["em"] = em
            pairs = samp[["img_path", "subject_id", "txt"]].rename(columns={"img_path": "filename", "subject_id": "patientid"})
            if pairs_basename:
                pairs["filename"] = pairs.filename.map(os.path.basename)  # runner finds images by basename under --img-dir
            pairs["y_target"] = (samp[target] == 1).astype(int).values
            pairs.to_csv(os.path.join(out, f"{prefix}_pairs.csv"), index=False)
            print(f"(d) wrote {os.path.join(out, prefix + '_pairs.csv')} for the Part C1 architecture runner (--mode pairs)")
        else:
            print("fewer than 100 studies with images: (a') and (c) skipped")
    per.to_csv(os.path.join(out, f"{prefix}_concomitant.csv"), index=False)
    lon.to_csv(os.path.join(out, f"{prefix}_longitudinal.csv"), index=False)
    C.save_json(res, os.path.join(out, f"{prefix}_results.json"))

    L = [f"# {title} ({res['n_studies_with_report']} frontal studies, {res['n_patients']} patients)", ""]
    if notes:
        L += [f"> {n}" for n in notes] + [""]
    L += [
         "**(a) Is 'uncertain' more frequent on AP (portable) films?** Odds ratio, uncertain vs definite label, "
         "among studies that mention the finding.", ""]
    if pool:
        L += [f"* pooled with finding fixed effects: OR {pool['odds_ratio']:.2f} "
              f"[{pool['ci_low']:.2f}, {pool['ci_high']:.2f}], p = {pool['p']:.2g}; "
              f"uncertain rate AP {pool['rate_AP']:.3f} vs PA {pool['rate_PA']:.3f}; n = {pool['n']}", ""]
    if len(per):
        L += C.md_table(per[["finding", "n_mentioned", "uncertain_rate", "odds_ratio", "ci_low", "ci_high", "p"]])
    L += ["", "**(b) Current-prior axis** (consecutive frontal studies of the same patient)", ""]
    L += C.md_table(lon)
    if em:
        E = pd.DataFrame([{**{"seed": r["seed"]}, **{f"share_{k}": v for k, v in r["class_share"].items()},
                           "P(-1 | hedge)": r["p_uncertain_given_named_hedge"], "P(-1 | other)": r["p_uncertain_given_other"],
                           "AUC hedge post.": r["auc_hedge_posterior_for_labeler_uncertain"],
                           "conf>0.99": r["frac_conf_gt_099"], "r(faithful, quality)": r["corr_faithful_posterior_quality"]}
                          for r in em])
        L += ["", f"**(c) Latent-class EM on real views, target = {target}** (face validity only; "
              f"features: {', '.join(em[0]['features'])})", ""]
        L += C.md_table(E)
    C.save_md(L, os.path.join(out, f"{prefix}_summary.md"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpg-dir", required=True)
    ap.add_argument("--reports-dir", default=None, help="MIMIC-CXR reports root (default: --jpg-dir)")
    ap.add_argument("--provider-csv", default=None)
    ap.add_argument("--radgraph-features", default=None)
    ap.add_argument("--target", default="Pleural Effusion")
    ap.add_argument("--n-cases", type=int, default=4000, help="studies sampled for the image-based EM (c)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--cnn-epochs", type=int, default=8)
    ap.add_argument("--n-restarts", type=int, default=6)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    st, fnd = load_tables(a.jpg_dir)
    st = attach_reports(st, a.reports_dir or a.jpg_dir)
    if a.target not in fnd:
        sys.exit(f"target {a.target} not among {fnd}")
    st = attach_provider(st, a.provider_csv)
    loader = None if a.no_images else (lambda samp: attach_images(samp, a.jpg_dir))
    return analyse(st, fnd, a.target, a.out, "mimic", "D2 MIMIC-CXR", loader, a.seeds, a.n_cases,
                   a.radgraph_features, a.cnn_epochs, a.n_restarts)


if __name__ == "__main__":
    main()

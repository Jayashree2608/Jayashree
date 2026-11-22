"""
D4 - REFLACX (Bigolin Lanfredi et al., 2022; PhysioNet "reflacx-xray-localization" 1.0.0,
credentialed; images are MIMIC-CXR DICOMs and are not needed here).

Role in Table VII: gaze-based validation of contradiction localisation. REFLACX gives, for each
reading, the dictated report, the radiologist's own certainty (1-5) for every label, ellipses
around each abnormality and the eye-tracking fixations. In phases 1 and 2 five radiologists
read the same images. Three checks follow:

  (a) Reader-reader agreement with certainty. Each label of each reading is put in one of
      three states: absent (certainty 0 or 1, "not selected"/"unlikely"), hedged (2-4,
      "less likely"/"possibly"/"probably") or definite (5, "consistent with"). Fleiss kappa
      across the readers of the same image, per label.
  (b) Is the lexical View 1 a good hedge detector? For every reading, the lexical hedge flag
      computed on the dictated transcription is compared with the reader's own certainty
      ("any label in 2-4"). This is the only check in the paper where the hedge reference
      comes from the reader, not from an NLP labeler.
  (c) Why are findings omitted? For an image where >= 2 readers call a label definite or
      probable (certainty >= 4), the peers' ellipses for that label define the region. For
      every reader of the image the fraction of fixation time inside that region (divided by
      the region's share of the chest box, so 1 = uniform viewing) is compared between
      readers who reported the finding and readers who omitted it (certainty 0-1). If those
      who omitted it looked as long as those who reported it, the omission was a decision
      error (looked, not reported); if much less, a search error (never looked). The paper's
      taxonomy treats both as "omission"; this measures which dominates.
  (d) Phase 1 has a reader-reported "Quality issue" label: is hedging more frequent when the
      reader flags quality (the concomitant link, Eq. 17, with a reader-reported covariate)?

FILES under --reflacx-dir: main_data/metadata_phase_{1,2,3}.csv; main_data/<id>/fixations.csv,
anomaly_location_ellipses.csv, chest_bounding_box.csv, transcription.txt.

USAGE  python3 reflacx_run.py --reflacx-dir <dir>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C

META_COLS = {"id", "split", "eye_tracking_data_discarded", "image", "dicom_id", "subject_id",
             "image_size_x", "image_size_y", "phase"}
BOOL_LABELS = {"support devices", "quality issue", "other"}


def state(c):
    c = float(c) if pd.notna(c) else 0.0
    return 0 if c <= 1 else (1 if c <= 4 else 2)          # absent / hedged / definite


def load(root):
    print("REFLACX files:")
    parts = []
    for ph in (1, 2, 3):
        f = C.find_file(root, f"metadata_phase_{ph}.csv")
        if f:
            m = C.read_table(f); m["phase"] = ph; parts.append(m)
    if not parts:
        sys.exit("no metadata_phase_*.csv found under " + root)
    meta = pd.concat(parts, ignore_index=True)
    labels = [c for c in meta.columns if c not in META_COLS and c.lower() not in BOOL_LABELS
              and pd.to_numeric(meta[c], errors="coerce").dropna().between(0, 5).all()
              and pd.to_numeric(meta[c], errors="coerce").notna().any()]
    for c in labels:      # NaN = label not used in that phase (phase 1 and phases 2-3 have different lists)
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    print(f"  readings {len(meta)}, images {meta.dicom_id.nunique()}, certainty labels: {labels}")
    folders = {}
    for dp, dns, _ in os.walk(root):
        for d in dns:
            folders.setdefault(d, os.path.join(dp, d))
    meta["folder"] = meta["id"].map(folders)
    print(f"  reading folders found: {meta.folder.notna().sum()}")
    return meta, labels


def agreement(meta, labels):
    multi = meta.groupby("dicom_id").filter(lambda g: len(g) >= 2)
    rows = []
    for l in labels:
        counts = []
        for _, g in multi.groupby("dicom_id"):
            if g[l].isna().any():
                continue
            s = g[l].map(state).values
            counts.append([(s == k).sum() for k in range(3)])
        counts = np.array(counts)
        if counts.size == 0 or counts[:, 1:].sum() == 0:
            continue
        rows.append(dict(label=l, images=len(counts), reads=int(counts.sum()),
                         hedged_share_of_positive=float(counts[:, 1].sum() / max(counts[:, 1:].sum(), 1)),
                         fleiss_kappa_3state=C.fleiss_kappa(counts),
                         fleiss_kappa_present=C.fleiss_kappa(np.column_stack([counts[:, 0], counts[:, 1:].sum(1)]))))
    return pd.DataFrame(rows), multi.dicom_id.nunique()


def lexical_vs_certainty(meta, labels):
    rows = []
    for _, r in meta.iterrows():
        if not isinstance(r.folder, str):
            continue
        t = os.path.join(r.folder, "transcription.txt")
        if not os.path.exists(t):
            continue
        txt = open(t, errors="ignore").read()
        h, m = C.lexical_v1(txt)
        cert = r[labels].astype(float).values          # NaN compares False
        rows.append(dict(lex_hedge=h, lex_mention=m, reader_hedge=int(((cert >= 2) & (cert <= 4)).any()),
                         reader_definite=int((cert == 5).any())))
    d = pd.DataFrame(rows)
    if d.empty:
        return None
    y, p = d.reader_hedge.values, d.lex_hedge.values
    tp = ((y == 1) & (p == 1)).sum()
    return dict(n=len(d), reader_hedge_rate=float(y.mean()), lexical_hedge_rate=float(p.mean()),
                precision=float(tp / max(p.sum(), 1)), recall=float(tp / max(y.sum(), 1)), kappa=C.cohen_kappa(y, p),
                mention_vs_definite_kappa=C.cohen_kappa(d.reader_definite.values, d.lex_mention.values))


def _inside(x, y, e):
    cx, cy = (e.xmin + e.xmax) / 2, (e.ymin + e.ymax) / 2
    a, b = max((e.xmax - e.xmin) / 2, 1), max((e.ymax - e.ymin) / 2, 1)
    return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1


def gaze(meta, labels, min_peers=2):
    m = meta[(meta.eye_tracking_data_discarded.astype(str).str.lower() != "true") & meta.folder.notna()]
    cache = {}

    def files(fid):
        if fid not in cache:
            fx = pd.read_csv(os.path.join(fid, "fixations.csv"))
            el = pd.read_csv(os.path.join(fid, "anomaly_location_ellipses.csv"))
            cb = pd.read_csv(os.path.join(fid, "chest_bounding_box.csv"))
            cache[fid] = (fx, el, cb)
        return cache[fid]
    rows = []
    for dicom, g in m.groupby("dicom_id"):
        if len(g) < 3:
            continue
        for l in labels:
            callers = g[g[l] >= 4]
            if len(callers) < min_peers:
                continue
            for _, r in g.iterrows():
                peers = callers[callers["id"] != r["id"]]
                if len(peers) < min_peers:
                    continue
                ells = []
                for _, p in peers.iterrows():
                    _, el, _ = files(p.folder)
                    lc = next((c for c in el.columns if c.lower() == l.lower()), None)
                    if lc is not None:
                        ells.append(el[el[lc].astype(str).str.lower().isin(["true", "1"])])
                ells = pd.concat(ells) if ells else pd.DataFrame()
                if ells.empty:
                    continue
                fx, _, cb = files(r.folder)
                x, y = fx.average_x_position.values, fx.average_y_position.values
                dur = (fx.timestamp_end_fixation - fx.timestamp_start_fixation).clip(lower=0).values
                inside = np.zeros(len(fx), dtype=bool)
                for _, e in ells.iterrows():
                    inside |= _inside(x, y, e)
                # area share of the region inside the chest box (Monte Carlo)
                b = cb.iloc[0]
                rng = np.random.default_rng(0)
                px = rng.uniform(b.xmin, b.xmax, 4000); py = rng.uniform(b.ymin, b.ymax, 4000)
                ins = np.zeros(4000, dtype=bool)
                for _, e in ells.iterrows():
                    ins |= _inside(px, py, e)
                share = max(ins.mean(), 1e-3)
                dwell = dur[inside].sum() / max(dur.sum(), 1e-9)
                st = state(r[l])
                rows.append(dict(dicom_id=dicom, label=l, status=["omitted", "hedged", "reported"][st],
                                 dwell_fraction=dwell, enrichment=dwell / share, any_fixation=float(inside.any())))
    d = pd.DataFrame(rows)
    if d.empty:
        return None, None
    summ = d.groupby("status").agg(n=("enrichment", "size"), dwell_fraction_median=("dwell_fraction", "median"),
                                   enrichment_median=("enrichment", "median"), enrichment_mean=("enrichment", "mean"),
                                   any_fixation_in_region=("any_fixation", "mean"))
    o, rep = d[d.status == "omitted"].enrichment, d[d.status == "reported"].enrichment
    test = dict(n_omitted=len(o), n_reported=len(rep),
                p_mannwhitney=float(mannwhitneyu(o, rep).pvalue) if len(o) > 2 and len(rep) > 2 else np.nan,
                ratio_median_enrichment_omitted_to_reported=float(o.median() / rep.median()) if len(o) and rep.median() > 0 else np.nan,
                frac_omitted_never_fixated=float((d[d.status == "omitted"].any_fixation == 0).mean()) if len(o) else np.nan)
    return summ, test


def quality_link(meta, labels):
    qc = next((c for c in meta.columns if c.lower() == "quality issue"), None)
    if qc is None:
        return None
    m = meta[meta[qc].notna()]
    q = m[qc].astype(str).str.lower().isin(["true", "1"]).astype(float).values
    cert = m[labels].astype(float).values
    y = ((cert >= 2) & (cert <= 4)).any(1).astype(int)
    if q.sum() < 3 or y.sum() in (0, len(y)):
        return dict(n=len(m), n_quality_issue=int(q.sum()), note="too few quality flags for a model")
    o = C.logit_or(y, q[:, None], ["quality_issue"]).iloc[0].to_dict()
    o.update(n=len(m), n_quality_issue=int(q.sum()), hedge_rate_q=float(y[q == 1].mean()), hedge_rate_noq=float(y[q == 0].mean()))
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reflacx-dir", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    meta, labels = load(a.reflacx_dir)
    A, n_multi = agreement(meta, labels)
    lv = lexical_vs_certainty(meta, labels)
    G, gt = gaze(meta, labels)
    q = quality_link(meta, labels)
    res = dict(n_readings=len(meta), n_images=int(meta.dicom_id.nunique()), n_multi_read_images=int(n_multi),
               lexical_vs_reader_certainty=lv, gaze_test=gt, quality_link=q)
    A.to_csv(os.path.join(a.out, "reflacx_agreement.csv"), index=False)
    if G is not None:
        G.to_csv(os.path.join(a.out, "reflacx_gaze_by_status.csv"))
    C.save_json(res, os.path.join(a.out, "reflacx_results.json"))
    L = [f"# D4 REFLACX ({res['n_readings']} readings, {res['n_images']} images, {n_multi} read by several radiologists)", "",
         "**(a) Reader-reader agreement with certainty** (absent 0-1 / hedged 2-4 / definite 5)", ""]
    L += C.md_table(A) if len(A) else ["(no multi-read images)"]
    if lv:
        L += ["", "**(b) Lexical hedge flag on the dictation vs the reader's own certainty**", "",
              f"* readings {lv['n']}; reader hedge rate {lv['reader_hedge_rate']:.3f}, lexical {lv['lexical_hedge_rate']:.3f}",
              f"* precision {lv['precision']:.3f}, recall {lv['recall']:.3f}, kappa {lv['kappa']:.3f}; "
              f"mention vs any definite label: kappa {lv['mention_vs_definite_kappa']:.3f}"]
    if gt:
        L += ["", "**(c) Gaze in the peers' region: omitted vs reported** (enrichment 1 = uniform viewing)", ""]
        Gf = G.copy(); Gf.insert(0, "status", Gf.index)
        L += C.md_table(Gf.reset_index(drop=True))
        L += ["", f"* Mann-Whitney p (enrichment, omitted vs reported) = {gt['p_mannwhitney']:.3g}; "
              f"median ratio {gt['ratio_median_enrichment_omitted_to_reported']:.3f}; "
              f"omissions with no fixation in the region: {gt['frac_omitted_never_fixated']:.3f}"]
    if q:
        L += ["", "**(d) Reader-flagged quality issue and hedging** (phase 1)", "",
              "* " + ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}" for k, v in q.items())]
    C.save_md(L, os.path.join(a.out, "reflacx_summary.md"))
    return res


if __name__ == "__main__":
    main()

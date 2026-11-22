"""
E1 - ReXGradient-160K (Zhang et al., 2025, arXiv:2505.00228): 160,000 chest X-ray studies, 273,004
images, 109,487 patients from three US health systems, with Indication / Comparison / Findings /
Impression sections and several studies per patient. Access: sign the licence on Hugging Face
(huggingface.co/datasets/rajpurkarlab/ReXGradient-160K), no PhysioNet credentialing. Images are
distributed as deid_png.part00..09 (cat deid_png.part* > deid_png.tar; tar -xf deid_png.tar).

Open replacement for MIMIC-CXR in Table VII (image-report and current-prior axes). It runs the
same analysis as D2 (mimic_run.analyse): (a) concomitant link AP vs PA, (b) current-prior axis,
(c) the EM of Part A on real views, (d) pairs for the Section II architecture. Labels come from
E0 (CheXbert or the rule-based labeler) because ReXGradient ships none; there is no reader id,
so View 3 is omitted.

SCHEMA. The public dataset card does not list the metadata columns, so they are discovered: every
CSV / JSON / JSONL under --rex-dir is read and the first table with a patient column and a
findings or impression column is used. Candidates (case-insensitive), overridable with --col-*:
  patient : PatientID, patient_id        study : AccessionNumber, StudyInstanceUid, study_id, id
  date    : StudyDate, study_date        view  : ViewPosition, view_position, view
  findings: Findings   impression: Impression   images: ImagePath(s), image_path(s), images
If the view is not a column, a JSON whose name contains "view_position" mapping image -> view is used.

USAGE  python3 rexgradient_run.py --rex-dir <repo> --images-dir <extracted deid_png> [--no-images]
         [--target "Pleural Effusion"] [--labeler auto|chexbert|lexical] [--max-label 20000]
"""
import argparse
import ast
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C
import mimic_run as MM
import label_reports as LR

CANDS = dict(patient=["PatientID", "patient_id", "patientid", "subject_id"],
             study=["AccessionNumber", "StudyInstanceUid", "StudyInstanceUID", "study_id", "studyid", "id"],
             date=["StudyDate", "study_date", "date"],
             view=["ViewPosition", "view_position", "view", "Projection"],
             findings=["Findings", "findings", "section_findings"],
             impression=["Impression", "impression", "section_impression"],
             images=["ImagePath", "ImagePaths", "image_path", "image_paths", "images", "ImageFile", "png_path"])


def _read(p):
    if p.endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    if p.endswith(".json"):
        d = json.load(open(p))
        if isinstance(d, dict):
            vals = list(d.values())
            if vals and all(isinstance(v, dict) for v in vals):
                return pd.DataFrame([dict(v, _key=k) for k, v in d.items()])
            return pd.DataFrame({"_key": list(d.keys()), "_value": vals})
        return pd.json_normalize(d)
    return pd.read_csv(p)


def discover(root, over):
    files = [p for p in C.find_files(root, r"\.(csv|json|jsonl)$") if "bbox" not in os.path.basename(p).lower()]
    tables, view_map = [], None
    for p in files:
        try:
            t = _read(p)
        except Exception as e:
            print(f"  could not read {p}: {e}"); continue
        print(f"  {os.path.relpath(p, root)}: {len(t)} rows; columns = {list(t.columns)[:20]}")
        if "view_position" in os.path.basename(p).lower():
            view_map = t
        cols = {k: (over.get(k) or C.col(t, *v, required=False)) for k, v in CANDS.items()}
        if cols["patient"] and (cols["findings"] or cols["impression"]):
            t = t.rename(columns={v: k for k, v in cols.items() if v})
            t["_file"] = os.path.basename(p)
            tables.append(t)
    if not tables:
        sys.exit("no metadata table with a patient column and a Findings/Impression column was found; "
                 "pass --col-patient / --col-findings / --col-impression with the names printed above")
    d = pd.concat(tables, ignore_index=True)
    print(f"  using {len(tables)} table(s), {len(d)} rows; mapped columns: "
          f"{[k for k in CANDS if k in d.columns]}")
    return d, view_map


def _first_image(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(v)
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return list(ast.literal_eval(v))
        except Exception:
            pass
    return [v] if isinstance(v, str) else []


def norm_view(v):
    v = str(v).upper().replace("-", "").replace(" ", "")
    if v.startswith("PA") or v == "POSTEROANTERIOR":
        return "PA"
    if v.startswith("AP") or "ANTEROPOSTERIOR" in v or "SUPINE" in v or "PORTABLE" in v:
        return "AP"
    return None


def build_table(d, view_map):
    if "study" not in d.columns:
        d["study"] = np.arange(len(d))
        print("  no study column found: one row = one study")
    if "images" in d.columns:
        d["image_list"] = d["images"].map(_first_image)
    else:
        d["image_list"] = [[] for _ in range(len(d))]
    if "view" not in d.columns and view_map is not None:
        vm = view_map
        if {"_key", "_value"} <= set(vm.columns):
            m = dict(zip(vm._key.astype(str), vm._value))
        else:
            kc = C.col(vm, "_key", "image_path", "ImagePath", "path", required=False)
            vc = C.col(vm, "ViewPosition", "view_position", "view", required=False)
            m = dict(zip(vm[kc].astype(str), vm[vc])) if kc and vc else {}
        bn = {os.path.basename(k): v for k, v in m.items()}
        d["view"] = [next((m.get(str(i), bn.get(os.path.basename(str(i)))) for i in L
                           if (m.get(str(i)) or bn.get(os.path.basename(str(i))))), None) for L in d.image_list]
        print(f"  views taken from the view-position file for {d.view.notna().sum()} studies")
    txt = (d.get("findings", pd.Series("", index=d.index)).fillna("").astype(str) + " " +
           d.get("impression", pd.Series("", index=d.index)).fillna("").astype(str)).str.strip()
    st = pd.DataFrame(dict(subject_id=d["patient"].astype(str), study_id=d["study"].astype(str), txt=txt,
                           ViewPosition=d["view"].map(norm_view) if "view" in d.columns else None,
                           order=pd.to_datetime(d["date"], errors="coerce").astype("int64") if "date" in d.columns else np.arange(len(d)),
                           img_rel=[L[0] if L else None for L in d.image_list]))
    st = st[st.txt.str.len() > 0].drop_duplicates("study_id")
    st["reader_id"] = np.nan
    print(f"  studies {len(st)}, patients {st.subject_id.nunique()}, views {st.ViewPosition.value_counts(dropna=False).to_dict()}")
    return st


def image_loader(images_dir):
    idx = None

    def load(samp):
        nonlocal idx
        paths = []
        for r in samp.img_rel:
            p = None
            if isinstance(r, str):
                cand = r if os.path.isabs(r) else os.path.join(images_dir, r)
                if os.path.exists(cand):
                    p = cand
                else:
                    if idx is None:
                        idx = C.index_images(images_dir, exts=(".png", ".jpg"))
                    p = idx.get(os.path.splitext(os.path.basename(r))[0])
            paths.append(p)
        s = samp.assign(img_path=paths)
        s = s[s.img_path.notna()].copy()
        print(f"  images found for {len(s)} / {len(samp)} sampled studies")
        s["img"] = [C.load_gray(p, 32) for p in s.img_path]
        s["q_meas"] = s.img.map(C.laplacian_var)
        return s
    return load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rex-dir", required=True)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--target", default="Pleural Effusion")
    ap.add_argument("--labeler", default="auto")
    ap.add_argument("--radgraph", default="auto")
    ap.add_argument("--max-label", type=int, default=20000)
    ap.add_argument("--n-cases", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", default="results")
    for k in CANDS:
        ap.add_argument(f"--col-{k}", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    print("ReXGradient-160K metadata:")
    d, vm = discover(a.rex_dir, {k: getattr(a, f"col_{k}") for k in CANDS})
    st = build_table(d, vm)
    st = st[st.ViewPosition.isin(["AP", "PA"])] if st.ViewPosition.notna().any() else st.assign(ViewPosition="PA")
    if st.ViewPosition.nunique() < 2:
        print("  WARNING: no AP/PA information: the concomitant test (a) is uninformative")
    rng = np.random.default_rng(0)                    # whole patients, random order, for the labeler budget
    prank = dict(zip(st.subject_id.unique(), rng.permutation(st.subject_id.nunique())))
    st = st.assign(_r=st.subject_id.map(prank)).sort_values(["_r", "order"]).drop(columns="_r")
    if a.max_label:
        st = st.head(a.max_label)
    rep = os.path.join(a.out, "rexgradient_reports.csv")
    st[["study_id", "txt"]].to_csv(rep, index=False)
    LR.main(["--reports", rep, "--prefix", "rexgradient", "--labeler", a.labeler, "--radgraph", a.radgraph,
             "--max-reports", "0", "--out", a.out])
    lab = pd.read_csv(os.path.join(a.out, "rexgradient_labels.csv"), dtype={"study_id": str})
    fnd = [f for f in MM.FINDINGS if f in lab.columns]
    st = st.merge(lab, on="study_id", how="inner")
    rgf = os.path.join(a.out, "rexgradient_radgraph_v1.csv")
    loader = None if (a.no_images or not a.images_dir) else image_loader(a.images_dir)
    src = lab.label_source.iloc[0] if len(lab) else "?"
    notes = []
    if not str(src).startswith("CheXbert"):
        notes.append("Labels come from the rule-based labeler. Check (a) and (b) are valid, but check (c) is circular "
                     "unless View 1 comes from RadGraph: the hedge component and the 'uncertain' label are then read off "
                     "the same rules. Install f1chexbert (and radgraph) and re-run for an independent check.")
    return MM.analyse(st, fnd, a.target, a.out, "rexgradient", f"E1 ReXGradient-160K, labels: {src}", loader,
                      a.seeds, a.n_cases, rgf if os.path.exists(rgf) else None, pairs_basename=False, notes=notes)


if __name__ == "__main__":
    main()

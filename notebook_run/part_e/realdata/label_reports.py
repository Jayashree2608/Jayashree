"""
E0 - Label free-text reports the way MIMIC-CXR ships them, for datasets that come without labels
(ReXGradient-160K, IU/Open-I).

MIMIC-CXR comes with CheXpert-labeler outputs (1 / 0 / -1 uncertain / blank) and RadGraph
inference. The open alternatives have reports only, so this step produces the same two things:

  * CheXpert-style labels for 14 findings, from the open CheXbert model (pip install f1chexbert;
    weights are downloaded from Hugging Face on first use) or, if it is not available, from the
    rule-based labeler in rd_common.lexical_labels.
  * View 1 features (rg_hedge, rg_mention, rg_negated) from the open RadGraph model
    (pip install radgraph; RadGraph or RadGraph-XL weights from Hugging Face), if available.

Which labeler produced each column is written to the output, and the summary says so. CheXbert
and RadGraph are models, not expert annotations: results built on them are face-validity checks.
When both CheXbert and the rule labeler run, their agreement per finding is reported.

INPUT  a CSV with a study id column and a report text column (names given with --id-col/--text-col)
OUTPUT <out>/<prefix>_labels.csv : study id, 14 finding columns, label_source
       <out>/<prefix>_radgraph_v1.csv : study_id, rg_hedge, rg_mention, rg_negated (if RadGraph ran)

USAGE  python3 label_reports.py --reports reports.csv --id-col study_id --text-col txt --prefix rexgradient
         [--labeler auto|chexbert|lexical] [--radgraph auto|on|off] [--max-reports 20000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C


def chexbert_labeler():
    try:
        from f1chexbert import F1CheXbert
        m = F1CheXbert(device="cpu")
    except Exception as e:                       # package missing, or weights not downloadable
        print(f"  CheXbert not available ({type(e).__name__}: {str(e)[:120]}); "
              "install with `pip install f1chexbert` on a machine that can reach huggingface.co")
        return None
    names = m.target_names

    def lab(text):
        v = m.get_label(str(text)[:4000], mode="classification")
        return {n: (np.nan if x == "" else float(x)) for n, x in zip(names, v)}
    return lab


def radgraph_model(kind):
    try:
        from radgraph import RadGraph
        return RadGraph(model_type=kind)
    except Exception as e:
        print(f"  RadGraph not available ({type(e).__name__}: {str(e)[:120]}); "
              "install with `pip install radgraph` on a machine that can reach huggingface.co")
        return None


def radgraph_features(rg, texts, batch=16):
    rows = []
    for i in range(0, len(texts), batch):
        out = rg(list(texts[i:i + batch]))
        for k in sorted(out, key=lambda x: int(x)):
            ents = out[k].get("entities", {})
            st = [C.radgraph_status(e.get("label")) for e in ents.values()
                  if any(t in str(e.get("tokens", "")).lower() for t in C.ANY_FINDING)]
            rows.append((float("OBS-U" in st), float("OBS-DP" in st), float("OBS-DA" in st)))
    return np.array(rows) if rows else np.zeros((0, 3))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--id-col", default="study_id")
    ap.add_argument("--text-col", default="txt")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--labeler", choices=["auto", "chexbert", "lexical"], default="auto")
    ap.add_argument("--radgraph", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--radgraph-model", default="radgraph-xl")
    ap.add_argument("--max-reports", type=int, default=20000, help="model labelers are slow on CPU (0 = all)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.reports)
    d = d[[a.id_col, a.text_col]].dropna().drop_duplicates(a.id_col).rename(columns={a.id_col: "study_id", a.text_col: "txt"})
    print(f"E0 labelling {len(d)} reports")
    t0 = time.time()
    lex = pd.DataFrame([C.lexical_labels(t) for t in d.txt], index=d.index)
    lab, src = lex.copy(), "rule-based lexical labeler"
    res = {"n": len(d)}
    if a.labeler in ("auto", "chexbert"):
        cb = chexbert_labeler()
        if cb is None and a.labeler == "chexbert":
            sys.exit("CheXbert requested but not available")
        if cb is not None:
            sub = d if not a.max_reports else d.head(a.max_reports)
            print(f"  CheXbert on {len(sub)} reports ...")
            cbl = pd.DataFrame([cb(t) for t in sub.txt], index=sub.index)
            lab = lab.astype(float)
            lab.loc[sub.index, cbl.columns] = cbl.values
            src = "CheXbert" if len(sub) == len(d) else f"CheXbert ({len(sub)} reports), rule-based for the rest"
            agree = {}
            for f in cbl.columns:
                if f in lex.columns:
                    x = cbl[f].fillna(9).values; y = lex.loc[sub.index, f].fillna(9).values
                    agree[f] = dict(exact=float((x == y).mean()), kappa=C.cohen_kappa(x, y),
                                    uncertain_rate_chexbert=float((x == -1).mean()), uncertain_rate_rule=float((y == -1).mean()))
            res["chexbert_vs_rule"] = agree
    lab["label_source"] = src
    out = pd.concat([d[["study_id"]], lab], axis=1)
    out.to_csv(os.path.join(a.out, f"{a.prefix}_labels.csv"), index=False)
    res["label_source"] = src
    res["uncertain_rate"] = {f: float((lab[f] == -1).mean()) for f in C.CHEXPERT14 if f in lab}
    if a.radgraph != "off":
        rg = radgraph_model(a.radgraph_model)
        if rg is None and a.radgraph == "on":
            sys.exit("RadGraph requested but not available")
        if rg is not None:
            sub = d if not a.max_reports else d.head(a.max_reports)
            print(f"  RadGraph ({a.radgraph_model}) on {len(sub)} reports ...")
            f = radgraph_features(rg, sub.txt.tolist())
            pd.DataFrame({"study_id": sub.study_id.values, "rg_hedge": f[:, 0], "rg_mention": f[:, 1],
                          "rg_negated": f[:, 2]}).to_csv(os.path.join(a.out, f"{a.prefix}_radgraph_v1.csv"), index=False)
            res["radgraph_reports"] = len(sub)
    res["seconds"] = round(time.time() - t0, 1)
    C.save_json(res, os.path.join(a.out, f"{a.prefix}_labels_info.json"))
    print(f"  labels from: {src}; {res['seconds']} s")
    return res


if __name__ == "__main__":
    main()

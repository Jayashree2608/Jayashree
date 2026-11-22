"""
D1 - RadGraph (Jain et al., 2021; PhysioNet "radgraph" 1.0.0, credentialed).

Role in Table VII: entity-relation supervision for the report parser. In this paper the
parser is never trained on RadGraph, and View 1 of the EM is a lexical proxy for RadGraph's
observation labels. RadGraph is therefore used here to measure two things the paper
assumes but has not checked:

  (a) How reliably is "hedge" itself annotated?  On test.json (two independent annotators
      per report) we report agreement on the observation labels OBS-DP (definitely present),
      OBS-U (uncertain) and OBS-DA (definitely absent), per entity span and per report.
      This is a ceiling for any text view that tries to detect hedging.
  (b) How good is the lexical View 1 used by the harness and by Parts B/C?  For every
      observation entity that names a finding, the lexical rule classifies its sentence as
      hedged / asserted / negated; we compare with the RadGraph label (3x3 confusion, kappa),
      and compare report-level hedge and mention flags with RadGraph's.
  (c) If MIMIC-CXR_graphs.json (RadGraph-model inference over all MIMIC-CXR reports) is
      present, it writes per-study View 1 features (radgraph_v1_mimic.csv) that the MIMIC
      runner (D2) uses in place of the lexical proxy.

FILES (any layout under --radgraph-dir): train.json, dev.json, test.json, and optionally
MIMIC-CXR_graphs.json. Each entry: {"text": tokenised report, "entities": {id: {"tokens",
"label", "start_ix", "end_ix", "relations"}}}; in test.json the annotations sit under
"labeler_1" and "labeler_2". The loader accepts either layout and prints what it found.

USAGE  python3 radgraph_run.py --radgraph-dir <dir> [--out results]
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

LABELS = ["OBS-DP", "OBS-U", "OBS-DA"]
LEX = {"OBS-DP": "asserted", "OBS-U": "hedged", "OBS-DA": "negated"}


def _load(path):
    with open(path) as f:
        d = json.load(f)
    print(f"  {os.path.basename(path)}: {len(d)} reports")
    return d


def annotations(entry):
    """List of entity dicts per annotator for one report entry."""
    keys = sorted(k for k in entry if re.fullmatch(r"labeler_?\d+", str(k)))
    if keys:
        return {k: entry[k].get("entities", {}) for k in keys}
    return {"labeler_1": entry.get("entities", {})}


def obs_entities(ents):
    out = {}
    for e in ents.values():
        lab = e.get("label", "")
        if lab in LABELS:
            out[(int(e.get("start_ix", -1)), int(e.get("end_ix", -1)))] = (lab, str(e.get("tokens", "")))
    return out


def sentence_of(tokens, i):
    lo = i
    while lo > 0 and tokens[lo - 1] not in (".", ";"):
        lo -= 1
    hi = i
    while hi < len(tokens) and tokens[hi] not in (".", ";"):
        hi += 1
    return " ".join(tokens[lo:hi + 1])


def lexical_label(sentence):
    low = sentence.lower()
    if C.HEDGE_RX.search(low):
        return "hedged"
    if C.NEG_RX.search(low):
        return "negated"
    return "asserted"


def names_finding(tok):
    t = tok.lower()
    return any(term in t for term in C.ANY_FINDING)


def agreement(test):
    rows_ent, rows_rep = [], []
    for rid, entry in test.items():
        ann = annotations(entry)
        if len(ann) < 2:
            continue
        a, b = [obs_entities(ann[k]) for k in sorted(ann)[:2]]
        for span in set(a) | set(b):
            rows_ent.append((a.get(span, ("none",))[0], b.get(span, ("none",))[0]))
        rows_rep.append((int(any(v[0] == "OBS-U" for v in a.values())),
                         int(any(v[0] == "OBS-U" for v in b.values()))))
    if not rows_ent:
        print("  test.json has no second annotator: agreement (a) skipped")
        return None
    e = pd.DataFrame(rows_ent, columns=["a", "b"])
    r = pd.DataFrame(rows_rep, columns=["a", "b"])
    both = e[(e.a != "none") & (e.b != "none")]
    res = {
        "n_reports": len(r), "n_obs_spans_union": len(e), "n_obs_spans_both": len(both),
        "span_match_rate": len(both) / len(e),
        "kappa_label_on_matched_spans": C.cohen_kappa(both.a, both.b),
        "kappa_label_union_incl_missing": C.cohen_kappa(e.a, e.b),
        "kappa_uncertain_vs_not_matched": C.cohen_kappa(both.a == "OBS-U", both.b == "OBS-U"),
        "report_hedge_rate_a": r.a.mean(), "report_hedge_rate_b": r.b.mean(),
        "kappa_report_any_uncertain": C.cohen_kappa(r.a, r.b),
        "confusion_matched": pd.crosstab(both.a, both.b).to_dict(),
    }
    return res


def lexical_vs_radgraph(sets):
    rows, reps = [], []
    for split, d in sets.items():
        for rid, entry in d.items():
            ents = annotations(entry)
            ents = ents[sorted(ents)[0]]
            toks = str(entry.get("text", "")).split()
            obs = obs_entities(ents)
            rg_h = rg_m = 0
            for (s, e_), (lab, tok) in obs.items():
                if not names_finding(tok) or s < 0 or s >= len(toks):
                    continue
                rows.append((split, lab, LEX[lab], lexical_label(sentence_of(toks, s))))
                rg_h |= lab == "OBS-U"; rg_m |= lab == "OBS-DP"
            lh, lm = C.lexical_v1(" ".join(toks))
            reps.append((split, int(rg_h), int(rg_m), int(lh), int(lm)))
    e = pd.DataFrame(rows, columns=["split", "radgraph", "radgraph_as_lex", "lexical"])
    r = pd.DataFrame(reps, columns=["split", "rg_hedge", "rg_mention", "lex_hedge", "lex_mention"])
    if e.empty:
        sys.exit("no observation entities naming a finding were found; check the JSON layout printed above")
    conf = pd.crosstab(e.radgraph_as_lex, e.lexical)

    def prf(y, p):
        tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
        return dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), kappa=C.cohen_kappa(y, p),
                    rate_radgraph=float(np.mean(y)), rate_lexical=float(np.mean(p)))
    return {
        "n_entities": len(e), "n_reports": len(r),
        "entity_accuracy": float((e.radgraph_as_lex == e.lexical).mean()),
        "entity_kappa": C.cohen_kappa(e.radgraph_as_lex, e.lexical),
        "entity_confusion_rows_radgraph": conf.to_dict(),
        "report_hedge": prf(r.rg_hedge.values, r.lex_hedge.values),
        "report_mention": prf(r.rg_mention.values, r.lex_mention.values),
    }


def mimic_features(path, out_csv):
    d = _load(path)
    rows = []
    for key, entry in d.items():
        m = re.search(r"s(\d{6,})", str(key))
        if not m:
            continue
        ents = annotations(entry)
        obs = obs_entities(ents[sorted(ents)[0]])
        fin = [lab for (lab, tok) in obs.values() if names_finding(tok)]
        rows.append((int(m.group(1)), float("OBS-U" in fin), float("OBS-DP" in fin), float("OBS-DA" in fin)))
    f = pd.DataFrame(rows, columns=["study_id", "rg_hedge", "rg_mention", "rg_negated"]).drop_duplicates("study_id")
    f.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}: {len(f)} studies; hedge rate {f.rg_hedge.mean():.3f}, mention rate {f.rg_mention.mean():.3f}")
    return len(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--radgraph-dir", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    print("RadGraph files:")
    sets = {}
    for nm in ("train.json", "dev.json", "test.json"):
        p = C.find_file(a.radgraph_dir, nm)
        if p:
            sets[nm[:-5]] = _load(p)
    if not sets:
        sys.exit("none of train.json / dev.json / test.json found under " + a.radgraph_dir)
    res = {"files": list(sets)}
    if "test" in sets:
        res["inter_annotator"] = agreement(sets["test"])
    res["lexical_vs_radgraph"] = lexical_vs_radgraph(sets)
    g = C.find_file(a.radgraph_dir, "MIMIC-CXR_graphs.json")
    if g:
        res["mimic_graphs_studies"] = mimic_features(g, os.path.join(a.out, "radgraph_v1_mimic.csv"))
    C.save_json(res, os.path.join(a.out, "radgraph_results.json"))

    L = ["# D1 RadGraph: is the text view reliable?", ""]
    ia = res.get("inter_annotator")
    if ia:
        L += ["**(a) Two annotators, test set** (ceiling for any hedge detector)", "",
              f"* reports: {ia['n_reports']}; observation spans (union): {ia['n_obs_spans_union']}; "
              f"span match rate {ia['span_match_rate']:.3f}",
              f"* kappa, OBS-DP/U/DA label on matched spans: {ia['kappa_label_on_matched_spans']:.3f}",
              f"* kappa, uncertain vs not, matched spans: {ia['kappa_uncertain_vs_not_matched']:.3f}",
              f"* kappa, report has any OBS-U: {ia['kappa_report_any_uncertain']:.3f} "
              f"(rates {ia['report_hedge_rate_a']:.3f} / {ia['report_hedge_rate_b']:.3f})", ""]
    lv = res["lexical_vs_radgraph"]
    L += ["**(b) Lexical View 1 vs RadGraph labels** (finding-naming observation entities)", "",
          f"* entities: {lv['n_entities']}; accuracy {lv['entity_accuracy']:.3f}; kappa {lv['entity_kappa']:.3f}", ""]
    t = pd.DataFrame([dict(flag=k, **lv[k]) for k in ("report_hedge", "report_mention")])
    L += C.md_table(t)
    L += ["", "Confusion (rows: RadGraph label, columns: lexical rule):", ""]
    cm = pd.DataFrame(lv["entity_confusion_rows_radgraph"]).fillna(0).astype(int)
    cm.insert(0, "radgraph", cm.index)
    L += C.md_table(cm.reset_index(drop=True))
    if "mimic_graphs_studies" in res:
        L += ["", f"(c) View 1 features from RadGraph inference written for {res['mimic_graphs_studies']} MIMIC-CXR studies."]
    C.save_md(L, os.path.join(a.out, "radgraph_summary.md"))
    return res


if __name__ == "__main__":
    main()

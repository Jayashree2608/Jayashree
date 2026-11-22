"""
D6 - CorBenchX (Zou et al., 2025, arXiv:2505.12057): 26,326 MIMIC-CXR reports with
LLM-injected errors of five types (omission, insertion, spelling error, side confusion,
other), each paired with the clean report. The authors state the data were submitted to
PhysioNet; access follows MIMIC-CXR credentialing. ReXErr (PhysioNet "rexerr-v1") has the
same clean/erroneous-pair structure and can be run with the same code via --field-* options.

Role in Table VII: synthetic-error baseline comparison for triage. The clean report is used as
the second read r' of the same study, so each pair exercises the reader-reader axis with a
known injected error:

  (a) Which injected errors can the paper's text view see at all? Per error type, the share of
      pairs whose View 1 indicators (hedge, mention) differ between the clean and the
      erroneous report, and the direction of the change.
  (b) Cause assigned by the paper's rule over (r, r') vs the injected type mapped onto the
      taxonomy: omission -> omission; spelling / other -> style (wording only); insertion and
      side confusion -> outside the taxonomy (a confident factual conflict, Table I last row).
      Rule: r hedges where r' does not -> hedge; r' mentions a finding that r does not ->
      omission; r mentions a finding r' does not -> conflict; text differs otherwise -> style.
  (c) Triage (Algorithm 2) on the assigned cause: share of each injected type sent to review.
      Errors outside the taxonomy that the 4-class system must place on hedge or style are
      auto-corrected without review: this measures that failure mode, stated in Section II-E.

FILES: one CSV / JSON / JSONL file (or a directory of them) with one row per erroneous report.
Columns are discovered (printed) and can be set explicitly:
  --field-clean  (default: first column whose name contains original/clean/gt/reference)
  --field-error  (default: first column whose name contains error/corrupt/noisy + report/text)
  --field-type   (default: first column whose name contains type; several types may be
                  separated by ; , or | or given as a list)

Also runs on the open pairs written by inject_errors.py (E3), which add a "hedge" type.

USAGE  python3 corbenchx_run.py --corbenchx-path <file or dir> [--prefix corbenchx]
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

TYPE_MAP = {"omission": "omission", "insertion": "conflict", "side confusion": "conflict",
            "side": "conflict", "laterality": "conflict", "spelling error": "style", "spelling": "style",
            "other": "style", "hedge": "hedge"}
CAUSES = ["faithful", "omission", "hedge", "style", "conflict"]


def _read_any(p):
    if p.endswith(".jsonl"):
        return pd.read_json(p, lines=True)
    if p.endswith(".json"):
        d = json.load(open(p))
        if isinstance(d, dict):
            d = list(d.values()) if all(isinstance(v, dict) for v in d.values()) else d.get("data", d)
        return pd.json_normalize(d)
    return pd.read_csv(p)


def load(path, fc, fe, ft):
    files = [path] if os.path.isfile(path) else C.find_files(path, r"\.(csv|json|jsonl)$")
    if not files:
        sys.exit("no CSV/JSON/JSONL found at " + path)
    d = pd.concat([_read_any(p) for p in files], ignore_index=True)
    print(f"CorBenchX: {len(d)} rows from {len(files)} file(s); columns = {list(d.columns)}")
    low = {c: c.lower() for c in d.columns}
    pick = lambda pats, excl=(): next((c for c, l in low.items() if any(re.search(p, l) for p in pats)
                                        and not any(re.search(x, l) for x in excl)), None)
    fc = fc or pick([r"original", r"clean", r"\bgt\b", r"ground", r"reference"], excl=[r"span", r"type"])
    fe = fe or pick([r"(error|corrupt|noisy|erroneous).*(report|text)", r"(report|text).*(error|corrupt|noisy)"],
                    excl=[r"span", r"type", r"desc"])
    ft = ft or pick([r"type"])
    print(f"  clean report: {fc}; erroneous report: {fe}; error type: {ft}")
    if not (fc and fe and ft):
        sys.exit("set --field-clean / --field-error / --field-type to the column names printed above")
    d = d.rename(columns={fc: "clean", fe: "error", ft: "etype"})
    d["etype"] = d.etype.map(lambda v: [str(x).strip().lower() for x in (v if isinstance(v, list)
                                         else re.split(r"[;,|]", str(v))) if str(x).strip()])
    return d


def rule(clean, err):
    hc, mc = C.lexical_v1(clean); he, me = C.lexical_v1(err)
    if he and not hc:
        return "hedge"          # checked first: hedging an asserted finding also removes the assertion
    if mc and not me:
        return "omission"
    if me and not mc:
        return "conflict"
    if clean.strip() != err.strip():
        return "style"          # wording, spelling or punctuation differs; View 1 unchanged
    return "faithful"


def algorithm2(cause):
    return "review" if cause == "omission" else "auto"          # conf = 1 for a deterministic rule


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corbenchx-path", required=True)
    ap.add_argument("--field-clean"); ap.add_argument("--field-error"); ap.add_argument("--field-type")
    ap.add_argument("--out", default="results")
    ap.add_argument("--prefix", default="corbenchx")
    ap.add_argument("--title", default="D6 CorBenchX")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d = load(a.corbenchx_path, a.field_clean, a.field_error, a.field_type)
    v = np.array([C.lexical_v1(str(c)) + C.lexical_v1(str(e)) for c, e in zip(d.clean, d.error)])
    d["d_hedge"] = v[:, 2] - v[:, 0]; d["d_mention"] = v[:, 3] - v[:, 1]
    d["cause"] = [rule(str(c), str(e)) for c, e in zip(d.clean, d.error)]
    d["route"] = d.cause.map(algorithm2)
    single = d[d.etype.map(len) == 1].copy()
    single["etype1"] = single.etype.map(lambda x: x[0])
    single["mapped"] = single.etype1.map(lambda t: TYPE_MAP.get(t, "style"))
    vis = single.groupby("etype1").agg(n=("cause", "size"),
                                       v1_changed=("d_mention", lambda s: float(((s != 0) | (single.loc[s.index, "d_hedge"] != 0)).mean())),
                                       mention_lost=("d_mention", lambda s: float((s < 0).mean())),
                                       mention_gained=("d_mention", lambda s: float((s > 0).mean())),
                                       hedge_gained=("d_hedge", lambda s: float((s > 0).mean())),
                                       to_review=("route", lambda s: float((s == "review").mean()))).reset_index()
    conf = pd.crosstab(single.mapped, pd.Categorical(single.cause, CAUSES), dropna=False)
    recall = {m: float(conf.loc[m, m] / conf.loc[m].sum()) for m in conf.index if m in conf.columns and conf.loc[m].sum()}
    res = dict(n=len(d), n_single_error=len(single), recall_by_mapped_cause=recall,
               conflict_auto_corrected=float((single[single.mapped == "conflict"].route == "auto").mean())
               if (single.mapped == "conflict").any() else np.nan)
    vis.to_csv(os.path.join(a.out, f"{a.prefix}_visibility.csv"), index=False)
    conf.to_csv(os.path.join(a.out, f"{a.prefix}_confusion.csv"))
    C.save_json(res, os.path.join(a.out, f"{a.prefix}_results.json"))
    L = [f"# {a.title} ({len(d)} erroneous reports, {len(single)} with a single injected error)", "",
         "**(a, c) Visibility of each injected error to View 1, and Algorithm 2 routing**", ""]
    L += C.md_table(vis)
    cf = conf.copy(); cf.insert(0, "injected (mapped)", cf.index)
    L += ["", "**(b) Cause from the rule over (r, r') vs injected type mapped to the taxonomy**", ""]
    L += C.md_table(cf.reset_index(drop=True))
    L += ["", "* recall per mapped cause: " + ", ".join(f"{k} {v:.3f}" for k, v in recall.items()),
          f"* injected conflicts (insertion, side confusion) auto-corrected without review: {res['conflict_auto_corrected']:.3f}"]
    C.save_md(L, os.path.join(a.out, f"{a.prefix}_summary.md"))
    return res


if __name__ == "__main__":
    main()

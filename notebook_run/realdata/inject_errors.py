"""
E3 - Injected report errors without PhysioNet (an open stand-in for CorBenchX / ReXErr).

Takes real reports from an open dataset (IU/Open-I indiana_reports.csv, the ReXGradient table
written by E1, or any CSV with an id and a text column) and writes clean/erroneous pairs with
the five CorBenchX error types, plus one type CorBenchX lacks:

  omission        drop a sentence that asserts a finding
  insertion       append a sentence asserting a finding the report does not mention
  side confusion  swap left and right
  spelling error  drop one letter from a finding word
  other           change a measurement unit (cm <-> mm) or, if none, merge two sentences
  hedge           (not in CorBenchX) turn an asserted finding into "possible ..."

The injection is rule-based and deterministic given the seed, so it is a controlled stress test of
the paper's text view and triage, not an independent benchmark: say so when reporting it. The
output has the columns D6 (corbenchx_run) detects automatically.

USAGE  python3 inject_errors.py --reports <csv> [--id-col uid --text-col txt] [--iu] [--n 5000]
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rd_common as C

INSERT = ["There is a small right pneumothorax.", "There is a moderate left pleural effusion.",
          "Right lower lobe consolidation is present.", "The heart is enlarged.",
          "There is a 1.5 cm nodule in the left upper lobe.", "Mild pulmonary edema is present."]


def asserted_sentences(text):
    out = []
    for i, s in enumerate(C.sentences(text)):
        low = s.lower()
        if any(t in low for t in C.ANY_FINDING) and not C.HEDGE_RX.search(low) and not C.NEG_RX.search(low):
            out.append(i)
    return out


def inject(text, kind, rng):
    S = C.sentences(text)
    A = asserted_sentences(text)
    if kind == "omission" and A:
        i = A[rng.integers(len(A))]
        return " ".join(s for j, s in enumerate(S) if j != i), f"removed: {S[i]}"
    if kind == "insertion":
        low = text.lower()
        cand = [s for s in INSERT if not any(t in low for t in C.ANY_FINDING if t in s.lower())]
        if cand:
            s = cand[rng.integers(len(cand))]
            return text.rstrip() + " " + s, f"inserted: {s}"
    if kind == "side confusion" and re.search(r"\b(left|right)\b", text, re.I):
        sw = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}
        return re.sub(r"\b(left|right|Left|Right|LEFT|RIGHT)\b", lambda m: sw[m.group(0)], text), "left/right swapped"
    if kind == "spelling error":
        words = [m for m in re.finditer(r"[A-Za-z]{6,}", text) if any(t in m.group(0).lower() for t in C.ANY_FINDING)]
        if words:
            m = words[rng.integers(len(words))]
            w = m.group(0); k = int(rng.integers(1, len(w) - 1))
            return text[:m.start()] + w[:k] + w[k + 1:] + text[m.end():], f"misspelled: {w}"
    if kind == "other":
        if re.search(r"\b\d+(\.\d+)?\s*cm\b", text):
            return re.sub(r"\b(\d+(?:\.\d+)?)\s*cm\b", r"\1 mm", text, count=1), "unit cm -> mm"
        if len(S) >= 2:
            return S[0].rstrip(".") + ", " + S[1][0].lower() + S[1][1:] + " " + " ".join(S[2:]), "sentences merged"
    if kind == "hedge" and A:
        i = A[rng.integers(len(A))]
        s = S[i].rstrip(".")
        S2 = list(S); S2[i] = "Possible " + s[0].lower() + s[1:] + "."
        return " ".join(S2), f"hedged: {S[i]}"
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="CSV, or the IU folder with --iu")
    ap.add_argument("--iu", action="store_true", help="--reports is the IU/Open-I folder (indiana_reports.csv)")
    ap.add_argument("--id-col", default="study_id")
    ap.add_argument("--text-col", default="txt")
    ap.add_argument("--n", type=int, default=5000, help="clean reports used")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.iu:
        f = C.require(C.find_file(a.reports, "indiana_reports.csv"), "indiana_reports.csv")
        d = C.read_table(f)
        d.columns = [c.lower() for c in d.columns]
        clean = lambda s: re.sub(r"\s+", " ", re.sub(r"\bx{2,}\b", " ", "" if pd.isna(s) else str(s), flags=re.I)).strip()
        d = pd.DataFrame({"study_id": d.uid, "txt": (d.findings.map(clean) + " " + d.impression.map(clean)).str.strip()})
    else:
        d = pd.read_csv(a.reports)[[a.id_col, a.text_col]].rename(columns={a.id_col: "study_id", a.text_col: "txt"})
    d = d[d.txt.fillna("").str.split().str.len() >= 5].sample(frac=1, random_state=a.seed).head(a.n)
    rng = np.random.default_rng(a.seed)
    kinds = ["omission", "insertion", "side confusion", "spelling error", "other", "hedge"]
    rows = []
    for r in d.itertuples():
        for k in kinds:
            e, desc = inject(r.txt, k, rng)
            if e and e.strip() != r.txt.strip():
                rows.append(dict(study_id=r.study_id, original_report=r.txt, error_report=e, error_type=k,
                                 error_description=desc))
    out = pd.DataFrame(rows)
    p = os.path.join(a.out, "injected_errors.csv")
    out.to_csv(p, index=False)
    print(f"E3 wrote {p}: {len(out)} pairs from {len(d)} reports; per type {out.error_type.value_counts().to_dict()}")
    return p


if __name__ == "__main__":
    main()

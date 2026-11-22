"""
Added after the real-data run (real_data_run.py) showed the alignment loss
can be minimised by copying the image embedding into t' (Eq. 10 with w_v -> 1).
Checks whether the same shortcut is present in the SYNTHETIC as-specified run:
trains one seed, then reports mean w_v at refinement step 1, and the cosine
between v and the raw text embedding vs the corrected t'.
"""
import json
import torch
import torch.nn.functional as F
from train import run_experiment

out = {}
for supervised in (False, True):
    model, test_t = run_experiment(seed=0, supervised=supervised, return_model=True)
    model.eval()
    with torch.no_grad():
        o = model.encode_batch(test_t)
        w_v, _ = model.reliability(o["v_t"], o["t_t"], o["p_c"])
        tp = model.correct(o["v_t"], o["t_t"], o["p_c"])
        out["supervised" if supervised else "as_specified"] = dict(
            mean_w_v_step1=float(w_v.mean()),
            mean_cos_v_text_raw=float(F.cosine_similarity(o["v_t"], o["t_t"], dim=-1).mean()),
            mean_cos_v_tprime=float(F.cosine_similarity(o["v_t"], tp, dim=-1).mean()),
        )
    print(supervised, out["supervised" if supervised else "as_specified"])
json.dump(out, open("results/shortcut_check.json", "w"), indent=2)

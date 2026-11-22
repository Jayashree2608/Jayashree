"""
Algorithm 1 (training) + Algorithm 2 (inference with triage) run end-to-end
on the synthetic dataset from generator.py, using the full neural
architecture in model.py.

IMPORTANT FINDING (reported honestly, not patched around silently): Algorithm
1 as formally specified never backpropagates a gradient into the
disagreement classifier of Eq. (6). gamma (Eq. 12) feeds into Eq. (13) only
through a stop-gradient, and the anticipation KL (Eq. 8) trains the
anticipation head against a *detached* copy of p(c|g,delta) as the target --
so the classifier itself (W_c, b_c, phi in Eq. 6) receives no training
signal from either loss term in the algorithm as literally written. This is
only harmless in a setting where the classifier is never expected to learn
anything (e.g. if it were frozen/hand-set), which the paper does not state.
For a fair test we run BOTH:
  (a) "as-specified": exactly Algorithm 1's two loss terms, no classifier
      supervision -- expected to show the classifier near chance, confirming
      the gap empirically rather than asserting it from reading the pseudocode.
  (b) "with-supervision": (a) plus a direct cross-entropy term on
      p(c|g,delta) against the known synthetic label z -- analogous in spirit
      to how Sections VII-VIII use the known z to validate the EM harness,
      since this is semi-synthetic validation with ground truth available by
      construction, not a real-data claim.
This is a genuine property of the formal algorithm discovered while
implementing it for this test, not a bug in this implementation; see
README.md for the full writeup.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix)

from generator import generate_dataset, build_vocab, tokenize, CLASS_NAMES, K_CLASSES
from model import FullArchitecture, contrastive_loss_per_sample

MAX_LEN = 16
DEVICE = torch.device("cpu")


def build_tensors(df, vocab, max_len=MAX_LEN):
    n = len(df)
    img_t = np.stack(df["img_t"].values)[:, None, :, :].astype(np.float32)
    img_prev = np.zeros_like(img_t)
    for i, v in enumerate(df["img_prev"].values):
        if v is not None:
            img_prev[i, 0] = v

    def tok_col(col):
        return np.array([tokenize(t if t is not None else "", vocab, max_len) for t in df[col]],
                         dtype=np.int64)

    report_t_ids = tok_col("report_t")
    report_prev_ids = np.array(
        [tokenize(t, vocab, max_len) if isinstance(t, str) else [0] * max_len
         for t in df["report_prev"]], dtype=np.int64)
    report_second_ids = np.array(
        [tokenize(t, vocab, max_len) if isinstance(t, str) else [0] * max_len
         for t in df["report_second"]], dtype=np.int64)

    batch = dict(
        img_t=torch.from_numpy(img_t),
        img_prev=torch.from_numpy(img_prev),
        report_t_ids=torch.from_numpy(report_t_ids),
        report_prev_ids=torch.from_numpy(report_prev_ids),
        report_second_ids=torch.from_numpy(report_second_ids),
        has_prior=torch.tensor(df["has_prior"].values.astype(np.float32)),
        has_second=torch.tensor(df["has_second"].values.astype(np.float32)),
        z=torch.tensor(df["z"].values.astype(np.int64)),
    )
    return batch


def iterate_batches(tensors, batch_size, shuffle, rng):
    n = tensors["z"].shape[0]
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        sel_t = torch.from_numpy(sel)
        yield {k: v[sel_t] for k, v in tensors.items()}


def run_experiment(seed, supervised, n_patients=400, epochs=25, pretrain_epochs=8,
                    batch_size=32, lambda_max=1.0, ce_weight=1.0, tau_r=0.7, lr=1e-3,
                    return_model=False):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    df = generate_dataset(seed=seed, n_patients=n_patients)
    idx = np_rng.permutation(len(df))
    split = int(0.7 * len(df))
    train_df = df.iloc[idx[:split]].reset_index(drop=True)
    test_df = df.iloc[idx[split:]].reset_index(drop=True)

    vocab = build_vocab([t for t in train_df["report_t"]] +
                         [t for t in train_df["report_prev"] if isinstance(t, str)] +
                         [t for t in train_df["report_second"] if isinstance(t, str)])

    train_t = build_tensors(train_df, vocab)
    test_t = build_tensors(test_df, vocab)

    model = FullArchitecture(vocab_size=len(vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        pretraining = epoch < pretrain_epochs
        epoch_losses = []
        for batch in iterate_batches(train_t, batch_size, shuffle=True, rng=np_rng):
            out = model.encode_batch(batch)
            t_prime = model.correct(out["v_t"], out["t_t"], out["p_c"])
            ell_align = contrastive_loss_per_sample(out["v_t"], t_prime)
            gamma = 1.0 - out["p_c"][:, 0]
            L = torch.mean((1.0 + lambda_max * gamma.detach()) * ell_align)

            if pretraining:
                q_c = model.anticipation(out["v_t"])
                p_c_detached = out["p_c"].detach().clamp_min(1e-8)
                kl = (p_c_detached * (p_c_detached.log() - q_c.clamp_min(1e-8).log())).sum(-1).mean()
                L = L + kl

            if supervised:
                ce = F.cross_entropy(out["logits_c"], batch["z"])
                L = L + ce_weight * ce

            opt.zero_grad()
            L.backward()
            opt.step()
            epoch_losses.append(L.item())
        loss_history.append(float(np.mean(epoch_losses)))

    # ---- evaluation (Algorithm 2 style inference) ----
    model.eval()
    with torch.no_grad():
        out = model.encode_batch(test_t)
        t_prime = model.correct(out["v_t"], out["t_t"], out["p_c"])
        p_c = out["p_c"].numpy()
        pred = p_c.argmax(1)
        conf = p_c.max(1)
        true = test_t["z"].numpy()

        acc = accuracy_score(true, pred)
        prec, rec, f1, support = precision_recall_fscore_support(
            true, pred, labels=range(K_CLASSES), zero_division=0)
        bal_acc = rec.mean()
        macro_f1 = f1.mean()
        cm = confusion_matrix(true, pred, labels=range(K_CLASSES))

        majority_class = pd.Series(train_df["z"]).mode()[0]
        baseline_acc = accuracy_score(true, np.full_like(true, majority_class))

        # Algorithm 2 triage routing
        routes = []
        for p_hat, c in zip(conf, pred):
            if c == 0:  # faithful
                routes.append("auto_corrected")
            elif c == 1 and p_hat >= tau_r:  # omission, confident
                routes.append("flag_for_radiologist")
            elif p_hat >= tau_r:
                routes.append("auto_corrected")
            else:
                routes.append("flag_for_radiologist")
        route_counts = pd.Series(routes).value_counts(normalize=True).to_dict()

    if return_model:
        return model, test_t
    return dict(
        seed=seed, supervised=supervised, n_train=len(train_df), n_test=len(test_df),
        loss_history=loss_history, accuracy=acc, balanced_accuracy=bal_acc,
        macro_f1=macro_f1, baseline_accuracy=baseline_acc,
        per_class_precision=dict(zip(CLASS_NAMES, prec.tolist())),
        per_class_recall=dict(zip(CLASS_NAMES, rec.tolist())),
        per_class_f1=dict(zip(CLASS_NAMES, f1.tolist())),
        per_class_support=dict(zip(CLASS_NAMES, support.tolist())),
        confusion_matrix=cm.tolist(),
        route_distribution=route_counts,
        vocab_size=len(vocab),
    )


def main():
    t0 = time.time()
    results = []
    for supervised in [False, True]:
        n_seeds = 3 if supervised else 2
        for seed in range(n_seeds):
            print(f"=== supervised={supervised} seed={seed} ===")
            r = run_experiment(seed=seed, supervised=supervised)
            print(f"  acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
                  f"macro_f1={r['macro_f1']:.3f} baseline={r['baseline_accuracy']:.3f}")
            print(f"  final loss={r['loss_history'][-1]:.4f} first loss={r['loss_history'][0]:.4f}")
            print(f"  routes={r['route_distribution']}")
            results.append(r)

    with open("results/full_architecture_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote results/full_architecture_results.json ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()

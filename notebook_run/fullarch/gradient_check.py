"""
Empirical check of which loss terms of Algorithm 1 (as specified) send gradient
into the disagreement classifier of Eq. (6) (parameters: classifier.phi, classifier.head).

Added after a self-review found that the earlier claim "Algorithm 1 never
backpropagates a gradient into the classifier" was inferred from the pseudocode
and was inaccurate: p(c|g,delta) is an input to ReliabilityNet (Eq. 9) with no
stop-gradient, so the alignment loss reaches the classifier indirectly.

Reports, on one batch of the synthetic dataset, the L2 gradient norm on the
classifier parameters under each loss component separately:
  (a) alignment term of Eq. (13) alone (gamma stop-gradient, as specified)
  (b) anticipation KL of Eq. (8) alone (detached target, as specified)
  (c) supervised cross-entropy (NOT part of Algorithm 1; the test's own addition)
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

from generator import generate_dataset, build_vocab
from model import FullArchitecture, contrastive_loss_per_sample
from train import build_tensors

torch.manual_seed(0)
df = generate_dataset(seed=0, n_patients=64)
vocab = build_vocab(list(df["report_t"]) + [t for t in df["report_prev"] if isinstance(t, str)]
                    + [t for t in df["report_second"] if isinstance(t, str)])
batch = build_tensors(df, vocab)
model = FullArchitecture(vocab_size=len(vocab))
model.train()

cls_params = list(model.classifier.parameters())


def cls_grad_norm(loss_fn):
    model.zero_grad()
    out = model.encode_batch(batch)
    loss = loss_fn(out)
    loss.backward()
    gs = [p.grad for p in cls_params if p.grad is not None]
    if not gs:
        return 0.0
    return float(torch.sqrt(sum((g ** 2).sum() for g in gs)))


def align_loss(out):
    t_prime = model.correct(out["v_t"], out["t_t"], out["p_c"])
    ell = contrastive_loss_per_sample(out["v_t"], t_prime)
    gamma = 1.0 - out["p_c"][:, 0]
    return torch.mean((1.0 + 1.0 * gamma.detach()) * ell)


def kl_loss(out):
    q = model.anticipation(out["v_t"])
    p = out["p_c"].detach().clamp_min(1e-8)
    return (p * (p.log() - q.clamp_min(1e-8).log())).sum(-1).mean()


def ce_loss(out):
    return F.cross_entropy(out["logits_c"], batch["z"])


def align_loss_pc_detached(out):
    t_prime = model.correct(out["v_t"], out["t_t"], out["p_c"].detach())
    ell = contrastive_loss_per_sample(out["v_t"], t_prime)
    gamma = 1.0 - out["p_c"][:, 0]
    return torch.mean((1.0 + 1.0 * gamma.detach()) * ell)


res = {
    "align_term_as_specified": cls_grad_norm(align_loss),
    "anticipation_KL_as_specified": cls_grad_norm(kl_loss),
    "align_term_if_p_c_detached_in_Eq9 (counterfactual)": cls_grad_norm(align_loss_pc_detached),
    "supervised_CE (added, not in Algorithm 1)": cls_grad_norm(ce_loss),
}
for k, v in res.items():
    print(f"{k:55s} classifier grad L2 norm = {v:.3e}")
json.dump(res, open("results/gradient_check.json", "w"), indent=2)

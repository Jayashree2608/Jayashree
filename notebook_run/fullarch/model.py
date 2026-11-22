"""
Full neural architecture of manuscript Section IV, implemented module-for-
module against the formal spec in Section IV-C (Eqs. 1-13) and Algorithms 1-2:

  Image encoder (f_img), Text encoder (f_txt) -> shared d-dim embedding space
  RadGraph parser                              -> Eq. (3)
  Consistency graph (fusion + 3 disagreement axes) -> Eqs. (4)-(5)
  Disagreement classifier p(c | g, delta)      -> Eq. (6)
  Anticipation head q(c | v), pretraining-only -> Eqs. (7)-(8)
  Asymmetric self-correction (ReliabilityNet) + refinement loop -> Eqs. (9)-(10)
  Inconsistency-guided contrastive loss        -> Eqs. (11)-(13)
  Severity-weighted triage (inference only)    -> Algorithm 2

This is new engineering for this test and is NOT the Gaussian-mixture EM
harness used for the manuscript's own Section VII/VIII results -- it is a
from-scratch implementation of the full neural pipeline the manuscript
describes but never itself trains end-to-end (per Table XVI's evidential-
status table, every reported manuscript number uses either the EM harness or
hand-set constants, never this architecture). Kept deliberately small (d=32,
2-layer transformer, tiny CNN) since the goal is to confirm the architecture
is correctly wired and trainable, not to chase accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASS_NAMES = ["faithful", "omission", "hedge", "style"]
K_CLASSES = len(CLASS_NAMES)


class ImageEncoder(nn.Module):
    """f_img: x -> v in R^d. CNN/ViT backbone per Table III (Module Summary) --
    here a small CNN, since a full ResNet-50/ViT-B-16 is unnecessary and
    untrainable-from-scratch on a few hundred 32x32 synthetic images."""

    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, d)

    def forward(self, x):
        # x: (B, 1, H, W)
        z = self.net(x).flatten(1)
        return self.fc(z)


class TextEncoder(nn.Module):
    """f_txt: r -> t in R^d, and exposes tau(r), the token-level hidden
    states, for the RadGraph parser's attention (Eq. 3)."""

    def __init__(self, vocab_size, d=32, max_len=20, nhead=2, nlayers=2):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=nhead, dim_feedforward=4 * d,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, token_ids):
        # token_ids: (B, L) with 0 = pad
        pad_mask = (token_ids == 0)  # True where padding
        x = self.emb(token_ids) + self.pos[: token_ids.shape[1]].unsqueeze(0)
        # guard against all-pad rows (the "missing" placeholder), which would
        # otherwise make every key masked and produce a NaN from softmax
        all_pad = pad_mask.all(dim=1)
        safe_mask = pad_mask.clone()
        safe_mask[all_pad, 0] = False
        tau = self.encoder(x, src_key_padding_mask=safe_mask)  # (B, L, d)
        valid = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
        denom = valid.sum(1).clamp(min=1.0)
        t = (tau * valid).sum(1) / denom  # masked mean pool -> (B, d)
        t = torch.where(all_pad.unsqueeze(-1), torch.zeros_like(t), t)
        return t, tau, pad_mask


class RadGraphParser(nn.Module):
    """Eq. (3): T = MHA(Q_tr, tau(r), tau(r)) in R^{K x d}, K learned triple
    queries attending over the current report's token states."""

    def __init__(self, d=32, n_triples=4, nhead=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_triples, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, nhead, batch_first=True)

    def forward(self, tau, pad_mask):
        B = tau.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        all_pad = pad_mask.all(dim=1)
        safe_mask = pad_mask.clone()
        safe_mask[all_pad, 0] = False
        out, _ = self.mha(q, tau, tau, key_padding_mask=safe_mask)
        out = torch.where(all_pad.view(B, 1, 1), torch.zeros_like(out), out)
        return out  # (B, n_triples, d)


class ConsistencyGraph(nn.Module):
    """Eqs. (4)-(5): fuses all available node embeddings by self-attention
    into a pooled graph embedding g, and computes the three disagreement
    scores delta_ir, delta_rr, delta_tp (zeroed when the relevant node is
    unavailable, per the manuscript's explicit missing-axis convention)."""

    def __init__(self, d=32, nhead=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.w_g = nn.Linear(d, d)

    def forward(self, nodes, node_mask):
        # nodes: (B, n_nodes, d); node_mask: (B, n_nodes) True = PAD/missing
        fused, _ = self.mha(nodes, nodes, nodes, key_padding_mask=node_mask)
        valid = (~node_mask).float().unsqueeze(-1)
        denom = valid.sum(1).clamp(min=1.0)
        pooled = (fused * valid).sum(1) / denom
        g = self.w_g(pooled)
        return g

    @staticmethod
    def disagreement_scores(v_t, t_t, v_prev, t_second, has_prior, has_second):
        def one_minus_cos(a, b):
            return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-6)

        delta_ir = one_minus_cos(v_t, t_t)
        delta_tp = torch.where(has_prior, one_minus_cos(v_t, v_prev),
                                torch.zeros_like(delta_ir))
        delta_rr = torch.where(has_second, one_minus_cos(t_t, t_second),
                                torch.zeros_like(delta_ir))
        return torch.stack([delta_ir, delta_rr, delta_tp], dim=-1)  # (B, 3)


class DisagreementClassifier(nn.Module):
    """Eq. (6): p(c | g, delta) = softmax(W_c [g; phi(delta)] + b_c)."""

    def __init__(self, d=32, delta_dim=3, phi_dim=8, k=K_CLASSES):
        super().__init__()
        self.phi = nn.Linear(delta_dim, phi_dim)
        self.head = nn.Linear(d + phi_dim, k)

    def forward(self, g, delta):
        phi_delta = F.relu(self.phi(delta))
        logits = self.head(torch.cat([g, phi_delta], dim=-1))
        return F.softmax(logits, dim=-1), logits


class AnticipationHead(nn.Module):
    """Eq. (7): q(c | v) = softmax(W_a v + b_a), pretraining-only."""

    def __init__(self, d=32, k=K_CLASSES):
        super().__init__()
        self.head = nn.Linear(d, k)

    def forward(self, v):
        logits = self.head(v)
        return F.softmax(logits, dim=-1)


class ReliabilityNet(nn.Module):
    """Eq. (9): [w_v, w_t] = softmax(W_r [v; t'; p(c|g,delta)])."""

    def __init__(self, d=32, k=K_CLASSES, w_v_cap=None):
        super().__init__()
        self.head = nn.Linear(2 * d + k, 2)
        self.w_v_cap = w_v_cap  # None = as specified; else w_v is bounded to [0, w_v_cap]

    def forward(self, v, t_prime, p_c):
        logits = self.head(torch.cat([v, t_prime, p_c], dim=-1))
        w = F.softmax(logits, dim=-1)
        if self.w_v_cap is not None:
            w_v = w[:, 0:1] * self.w_v_cap
            return w_v, 1.0 - w_v
        return w[:, 0:1], w[:, 1:2]  # w_v, w_t


class FullArchitecture(nn.Module):
    def __init__(self, vocab_size, d=32, max_len=20, n_triples=4, refine_steps=2,
                 detach_v=False, w_v_cap=None):
        super().__init__()
        self.d = d
        self.refine_steps = refine_steps
        self.detach_v = detach_v  # Eq. (10) variant: stop-gradient on v inside the mixture
        self.img_enc = ImageEncoder(d)
        self.txt_enc = TextEncoder(vocab_size, d, max_len)
        self.radgraph = RadGraphParser(d, n_triples)
        self.cgraph = ConsistencyGraph(d)
        self.classifier = DisagreementClassifier(d)
        self.anticipation = AnticipationHead(d)
        self.reliability = ReliabilityNet(d, w_v_cap=w_v_cap)

    def encode_batch(self, batch):
        """batch is a dict of tensors; returns everything needed for the loss
        and for inference, mirroring Algorithm 1 lines 1-6."""
        v_t = self.img_enc(batch["img_t"])
        t_t, tau_t, pad_t = self.txt_enc(batch["report_t_ids"])
        T = self.radgraph(tau_t, pad_t)  # (B, n_triples, d)

        v_prev = self.img_enc(batch["img_prev"])  # zero image if missing
        t_prev, _, _ = self.txt_enc(batch["report_prev_ids"])
        t_second, _, _ = self.txt_enc(batch["report_second_ids"])

        has_prior = batch["has_prior"].bool()
        has_second = batch["has_second"].bool()
        # zero out embeddings for missing nodes (encoder on a zero/pad input
        # already tends to zero, but be explicit and exact)
        v_prev = torch.where(has_prior.unsqueeze(-1), v_prev, torch.zeros_like(v_prev))
        t_prev = torch.where(has_prior.unsqueeze(-1), t_prev, torch.zeros_like(t_prev))
        t_second = torch.where(has_second.unsqueeze(-1), t_second, torch.zeros_like(t_second))

        B = v_t.shape[0]
        nodes = torch.stack([v_t, v_prev, t_t, t_prev, t_second], dim=1)  # (B,5,d)
        nodes = torch.cat([nodes, T], dim=1)  # (B, 5+n_triples, d)
        node_missing = torch.stack([
            torch.zeros(B, dtype=torch.bool, device=v_t.device),  # v_t always present
            ~has_prior,
            torch.zeros(B, dtype=torch.bool, device=v_t.device),  # t_t always present
            ~has_prior,
            ~has_second,
        ], dim=1)
        triple_missing = torch.zeros(B, T.shape[1], dtype=torch.bool, device=v_t.device)
        node_mask = torch.cat([node_missing, triple_missing], dim=1)

        g = self.cgraph(nodes, node_mask)
        delta = ConsistencyGraph.disagreement_scores(
            v_t, t_t, v_prev, t_second, has_prior, has_second)
        p_c, logits_c = self.classifier(g, delta)
        return dict(v_t=v_t, t_t=t_t, g=g, delta=delta, p_c=p_c, logits_c=logits_c)

    def correct(self, v_t, t_t, p_c):
        """Eqs. (9)-(10): K refinement steps, t'^(0) = t_t."""
        t_prime = t_t
        v_mix = v_t.detach() if self.detach_v else v_t
        for _ in range(self.refine_steps):
            w_v, w_t = self.reliability(v_t, t_prime, p_c)
            t_prime = w_v * v_mix + w_t * t_prime
        return t_prime


def contrastive_loss_per_sample(v, t_prime, temperature=0.1):
    """Eq. (11), bidirectional InfoNCE, returned per-sample (not yet batch-
    averaged) so the Eq. (13) per-sample reweighting can be applied."""
    v = F.normalize(v, dim=-1)
    t_prime = F.normalize(t_prime, dim=-1)
    sim = v @ t_prime.t() / temperature  # (B, B)
    targets = torch.arange(v.shape[0], device=v.device)
    loss_v2t = F.cross_entropy(sim, targets, reduction="none")
    loss_t2v = F.cross_entropy(sim.t(), targets, reduction="none")
    return 0.5 * (loss_v2t + loss_t2v)

"""
Synthetic image+report dataset generator for a FULL end-to-end test of the
Section IV neural architecture (dual-encoder backbone, RadGraph parser,
consistency graph, latent-cause disagreement model, anticipation head,
asymmetric correction, triage) -- as distinct from the Section VII/VIII
Gaussian-mixture EM harness (harness.py), which deliberately collapses the
architecture into hand-engineered scalar "views" to get a tractable
identifiability argument.

This generator reuses the same conventions established in harness.py
(four-class taxonomy faithful/omission/hedge/style, the same hedge/style/
faithful/normal text templates, the same acquisition-quality-confound image
generator) so results here are comparable in spirit, but produces actual
image tensors and raw report text for every one of x_t, x_{t-1}, r_t,
r_{t-1}, r'_t -- i.e. populates all three disagreement axes (image--report,
reader--reader, current--prior) with real paired data, which harness.py does
not need to do because it only ever worked with pre-extracted scalar
features.

Design choices made explicit here (none of these are claims from the paper,
they are this test's own synthetic-data assumptions):
  - The current-visit label z in {faithful, omission, hedge, style} is drawn
    exactly as in harness.py's generate_data, and determines how r_t is
    worded relative to x_t's true_finding.
  - A prior visit (x_{t-1}, r_{t-1}) is present for PRIOR_PRESENT_PROB of
    cases; its finding is independently re-drawn (persistence or resolution)
    and its report is always written faithfully -- it exists only to give
    the current--prior axis a real, non-trivial signal (same finding vs.
    changed finding), not to test recovery of a second latent cause.
  - A second reader's independent report r'_t is present for
    SECOND_READER_PROB of cases. It describes the SAME true_finding_t but is
    generated from an independently-drawn reader "wording style" (its own
    faithful/hedge/style draw, with its own reader traits) -- this is what
    gives the reader--reader axis real signal (two readers can describe the
    same image differently) without conflating it with the primary label z.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

CLASS_NAMES = ["faithful", "omission", "hedge", "style"]
K_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 32

HEDGE_WORDS = ["may represent", "possibly", "cannot exclude", "likely represents",
               "questionable", "suggestive of", "not definitively", "probable"]
HEDGE_TEMPLATES = [
    "{f} may represent an early process correlate clinically .",
    "findings are suggestive of {f} though not definitively established .",
    "cannot exclude {f} on this study .",
]
FAITHFUL_TEMPLATES = ["{f} is present .", "findings consistent with {f} ."]
STYLE_TEMPLATES = ["impression notable for {f} as above .", "{f} noted no other acute findings ."]
NORMAL_TEMPLATES = ["no acute cardiopulmonary abnormality .", "unremarkable study ."]

PRIOR_PRESENT_PROB = 0.7
SECOND_READER_PROB = 0.5


def make_image(has_finding, q_true, rng):
    base = rng.normal(0.4, 0.05, size=(IMG_SIZE, IMG_SIZE))
    if has_finding:
        cx, cy = rng.integers(10, IMG_SIZE - 10, size=2)
        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0 ** 2)))
        base = base + 0.35 * blob
    blur_sigma = 0.3 + 3.0 * q_true
    img = gaussian_filter(base, sigma=blur_sigma)
    img = img + rng.normal(0, 0.03 + 0.05 * q_true, size=img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


def _draw_z(true_finding, rd, beta_q, q_true, rng):
    logit_faithful = 0.0
    logit_omission = -1.0 + (0.8 * rd["omission_trait"] if true_finding else -5.0)
    logit_hedge = -0.5 + beta_q * q_true + 0.8 * rd["hedge_trait"]
    logit_style = -1.0 + 0.8 * rd["style_trait"]
    logits = np.array([logit_faithful, logit_omission, logit_hedge, logit_style])
    p = np.exp(logits - logits.max()); p = p / p.sum()
    return int(rng.choice(K_CLASSES, p=p)), p


def _write_report(z, true_finding, rng):
    if not true_finding:
        return rng.choice(NORMAL_TEMPLATES)
    if z == 1:  # omission: true finding present, report says nothing
        return rng.choice(NORMAL_TEMPLATES)
    if z == 2:
        return rng.choice(HEDGE_TEMPLATES).format(f="the opacity")
    if z == 3:
        return rng.choice(STYLE_TEMPLATES).format(f="an opacity")
    return rng.choice(FAITHFUL_TEMPLATES).format(f="an opacity")


def generate_dataset(seed, n_patients=300, n_readers=20, beta_q=2.2):
    rng = np.random.default_rng(seed)
    readers = pd.DataFrame({
        "reader_id": range(n_readers),
        "hedge_trait": rng.normal(0, 1, n_readers),
        "omission_trait": rng.normal(0, 1, n_readers),
        "style_trait": rng.normal(0, 1, n_readers),
    }).set_index("reader_id")

    rows = []
    for pid in range(n_patients):
        true_finding_t = int(rng.integers(0, 2))
        q_t = float(rng.uniform(0, 1))
        img_t = make_image(true_finding_t, q_t, rng)

        reader_id = int(rng.integers(0, n_readers))
        rd = readers.loc[reader_id]
        z, _ = _draw_z(true_finding_t, rd, beta_q, q_t, rng)
        report_t = _write_report(z, true_finding_t, rng)

        has_prior = bool(rng.random() < PRIOR_PRESENT_PROB)
        img_prev = None
        report_prev = None
        true_finding_prev = None
        if has_prior:
            # independent re-draw: finding may persist or resolve/appear
            true_finding_prev = int(rng.integers(0, 2))
            q_prev = float(rng.uniform(0, 1))
            img_prev = make_image(true_finding_prev, q_prev, rng)
            # prior report always written faithfully (not the object under test)
            report_prev = _write_report(0, true_finding_prev, rng)

        has_second = bool(rng.random() < SECOND_READER_PROB)
        report_second = None
        if has_second:
            reader2_id = int(rng.integers(0, n_readers))
            while reader2_id == reader_id and n_readers > 1:
                reader2_id = int(rng.integers(0, n_readers))
            rd2 = readers.loc[reader2_id]
            z2, _ = _draw_z(true_finding_t, rd2, beta_q, q_t, rng)
            report_second = _write_report(z2, true_finding_t, rng)

        rows.append(dict(
            patient_id=pid, z=z, class_name=CLASS_NAMES[z],
            true_finding_t=true_finding_t, q_t=q_t, img_t=img_t, report_t=report_t,
            has_prior=has_prior, img_prev=img_prev, report_prev=report_prev,
            true_finding_prev=true_finding_prev,
            has_second=has_second, report_second=report_second,
        ))
    df = pd.DataFrame(rows)
    return df


def build_vocab(texts):
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in texts:
        for tok in t.split():
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def tokenize(text, vocab, max_len):
    ids = [vocab.get(tok, 1) for tok in text.split()][:max_len]
    ids = ids + [0] * (max_len - len(ids))
    return ids

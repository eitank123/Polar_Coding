"""Frame synchronization: blind decoder-metric search, energy pre-detection, ZC preamble.

Blind search (Parts 1, 2, 4, 5). Candidate frame starts are every `step`-th bit
of an LLR stream (step=N: the 5 burst slots of Parts 1/2; step=2: every QPSK
symbol in Parts 4/5); each candidate is a full length-N window. Two stages:
  1. SC scan:     SC frozen-bit metric of every candidate window (cheap, batched);
                  argmax = the SC sync decision.
  2. SCL rescore: the `num_candidates` best SC candidates are re-decoded with
                  CA-SCL; argmax of -PM = the CA-SCL sync decision.
                  (num_candidates=None: every candidate, as in the 5-slot case.)

Energy detector (Parts 4/5, blind schemes). Before any polar decoding, the
received energy of every candidate window is compared with a threshold; only
windows that pass are decoded, and a stream with no passing window is declared
empty without decoding at all.
"""
import functools

import torch
from scipy.stats import ncx2


def candidate_windows(llr_stream, N, step=1):
    """Length-N windows starting every `step` bits: [T, O, N] view."""
    return llr_stream.unfold(-1, N, step)


def _decode_windows(decoder, windows, max_windows):
    """Run `decoder` on windows [T, M, N] in chunks of whole trials; returns its outputs reshaped to [T, M, ...]."""
    T, M, N = windows.shape
    rows = max(1, max_windows // M)
    parts = [decoder(windows[r:r + rows].reshape(-1, N)) for r in range(0, T, rows)]
    return [torch.cat([p[i].reshape(-1, M, *p[i].shape[1:]) for p in parts]) for i in range(len(parts[0]))]


def blind_sync(link, llr_stream, num_candidates, step=1, max_windows=8192):
    """Estimate the packet start bit with SC and with CA-SCL; also return the decoded bits there."""
    windows = candidate_windows(llr_stream, link.N, step)
    T, O, _ = windows.shape
    rows = torch.arange(T)

    _, sc_metric = _decode_windows(link.sc, windows, max_windows)  # [T, O]
    est_sc = sc_metric.argmax(dim=1)  # first max wins
    u_sc, _ = link.sc(windows[rows, est_sc])

    M = O if num_candidates is None else min(num_candidates, O)
    candidates = sc_metric.topk(M, dim=1).indices  # [T, M]
    u_c, scl_metric, _ = _decode_windows(link.scl, windows[rows[:, None], candidates], max_windows)
    best = scl_metric.argmax(dim=1)
    return {
        "est_sc": est_sc * step, "u_sc": u_sc,
        "est_scl": candidates[rows, best] * step, "u_scl": u_c[rows, best],
        "candidates": candidates * step,
    }


# ---------------------------------------------------------------------------
# Energy detector (pre-stage that limits polar decoding, Parts 4/5)
# ---------------------------------------------------------------------------
def window_energy(y, n_sym, max_offset):
    """Received energy sum_{k<n_sym} |y[d+k]|^2 of every window start d = 0..max_offset: [T, D+1]."""
    c = torch.cumsum(y.abs().to(torch.float64) ** 2, dim=-1)
    c = torch.nn.functional.pad(c, (1, 0))
    return (c[:, n_sym:n_sym + max_offset + 1] - c[:, :max_offset + 1]).to(torch.float32)


def energy_threshold(no, n_sym, miss_prob, es=1.0):
    """Signal-based threshold: a frame-aligned window falls below it with probability `miss_prob`.

    With y = s + n, |s|^2 = es per symbol and complex noise variance no, the
    window energy satisfies 2E/no ~ noncentral chi^2(df=2*n_sym, nc=2*n_sym*es/no);
    the threshold is that distribution's `miss_prob` quantile.
    """
    no = float(no)
    return float(no / 2 * ncx2.ppf(miss_prob, 2 * n_sym, 2 * n_sym * es / no))


@functools.lru_cache(maxsize=None)
def _noise_max_energy_quantile(n_sym, max_offset, pass_prob, num_streams=50000):
    """(1 - pass_prob) quantile of max_d E(d) / N0 for a noise-only stream.

    Calibrated by Monte Carlo from the noise model alone (unit-variance complex
    Gaussian noise, fixed private seed so it does not touch the simulation RNG).
    Overlapping windows are strongly correlated, so a Bonferroni bound over all
    windows would be ~20x too conservative. The ratio max E / N0 does not depend
    on N0, so one calibration serves every SNR.
    """
    g = torch.Generator().manual_seed(12345)
    maxima = []
    for start in range(0, num_streams, 5000):
        T = min(5000, num_streams - start)
        noise = torch.complex(torch.randn(T, max_offset + n_sym, generator=g),
                              torch.randn(T, max_offset + n_sym, generator=g)) / 2 ** 0.5
        maxima.append(window_energy(noise, n_sym, max_offset).max(dim=1).values)
    return float(torch.quantile(torch.cat(maxima).double(), 1 - pass_prob))


def noise_floor_threshold(no, n_sym, max_offset, pass_prob):
    """Noise-floor (CFAR) threshold: a noise-only stream has some window above it
    with probability pass_prob. Uses only the noise model (not the frame energy)."""
    return float(no) * _noise_max_energy_quantile(n_sym, max_offset, pass_prob)


def gate_threshold(no, n_sym, max_offset, miss_prob, fade_margin_db, noise_pass_prob):
    """Absolute energy threshold = min(signal-based with a fade margin, noise-floor based).

    The signal-based term assumes the frame arrives up to `fade_margin_db` below
    nominal energy (keeps P_miss ~ 0 at low SNR); the noise-floor term does not
    depend on the frame energy at all, so at high SNR the gate tolerates fades
    far deeper than the margin. Taking the minimum keeps the more permissive one.
    """
    es = 10 ** (-fade_margin_db / 10)
    return min(energy_threshold(no, n_sym, miss_prob, es),
               noise_floor_threshold(no, n_sym, max_offset, noise_pass_prob))


def gate_windows(energy, no, n_sym, max_offset, miss_prob, fade_margin_db, noise_pass_prob, kappa):
    """Energy-detector window selection: which candidate windows get polar-decoded.

    A window passes if (1) its energy exceeds the absolute gate threshold and
    (2) it lies within kappa standard deviations of the stream's energy peak.
    (2) is relative, with the signal level estimated from the peak itself
    (es_hat = max(E_max/n - N0, 0)), so it adapts to the received energy instead
    of assuming the nominal one; sigma = sqrt(2 n (N0^2 + 2 N0 es_hat)) is the std
    of the energy difference between two windows. Returns a bool mask [T, D+1].
    """
    no = float(no)
    gamma_abs = gate_threshold(no, n_sym, max_offset, miss_prob, fade_margin_db, noise_pass_prob)
    e_max = energy.max(dim=1, keepdim=True).values
    es_hat = (e_max / n_sym - no).clamp(min=0)
    sigma = torch.sqrt(2 * n_sym * (no ** 2 + 2 * no * es_hat))
    return (energy >= gamma_abs) & (energy >= e_max - kappa * sigma)


def gated_sync(link, llr_stream, passed, num_candidates, step, max_windows=8192):
    """Blind sync restricted to the windows selected by the energy detector.

    `passed` [T, O] marks the candidate windows to decode (same indexing as
    candidate_windows(llr_stream, N, step)). Only passing windows are SC-scanned;
    CA-SCL re-scores the best `num_candidates` passing windows. Streams with no
    passing window get statistic -inf (declared empty, nothing decoded).
    Returns statistics, start estimates (bits), decoded bits, and the number of
    SC / SCL window decodes actually performed (complexity).
    """
    windows = candidate_windows(llr_stream, link.N, step)
    T, O, N = windows.shape
    sc_metric = torch.full((T, O), -float("inf"))
    idx = passed.nonzero()
    for start in range(0, len(idx), max_windows):
        i = idx[start:start + max_windows]
        sc_metric[i[:, 0], i[:, 1]] = link.sc(windows[i[:, 0], i[:, 1]])[1]

    rows = torch.arange(T)
    active = passed.any(dim=1)
    est_sc = sc_metric.argmax(dim=1)
    M = O if num_candidates is None else min(num_candidates, O)
    candidates = sc_metric.topk(M, dim=1).indices  # [T, M]; -inf entries are masked below
    cand_ok = passed[rows[:, None], candidates]
    scl_metric = torch.full((T, M), -float("inf"))
    u_cand = torch.zeros(T, M, link.K)
    ci = cand_ok.nonzero()
    for start in range(0, len(ci), max_windows):
        i = ci[start:start + max_windows]
        u, m, _ = link.scl(windows[i[:, 0], candidates[i[:, 0], i[:, 1]]])
        scl_metric[i[:, 0], i[:, 1]] = m
        u_cand[i[:, 0], i[:, 1]] = u
    best = scl_metric.argmax(dim=1)
    u_sc = torch.zeros(T, link.K)
    if active.any():
        u_sc[active] = link.sc(windows[rows[active], est_sc[active]])[0]
    return {
        "active": active,
        "sc_stat": sc_metric.max(dim=1).values, "est_sc": est_sc * step, "u_sc": u_sc,
        "scl_stat": scl_metric.max(dim=1).values, "est_scl": candidates[rows, best] * step,
        "u_scl": u_cand[rows, best],
        "sc_decodes": passed.sum(dim=1), "scl_decodes": cand_ok.sum(dim=1),
    }


# ---------------------------------------------------------------------------
# Classical preamble-based synchronization (Part 5)
# ---------------------------------------------------------------------------
def zc_sequence(length, root=1):
    """Zadoff-Chu sequence (odd length): exp(-j*pi*u*n*(n+1)/L), unit modulus."""
    n = torch.arange(length, dtype=torch.float64)
    return torch.exp(-1j * torch.pi * root * n * (n + 1) / length).to(torch.complex64)


def zc_detect(y, zc, max_offset):
    """Normalized ZC correlation at symbol offsets 0..max_offset.

    c[d] = |<zc, y[d:d+L]>|^2 / (||zc||^2 * ||y[d:d+L]||^2) in [0, 1] (scale-invariant).
    Returns (max_d c[d], argmax_d c[d]).
    """
    L = zc.shape[0]
    w = y.unfold(-1, L, 1)[:, :max_offset + 1]  # [T, D+1, L]
    corr = (w * zc.conj()).sum(dim=-1).abs() ** 2
    energy = (w.abs() ** 2).sum(dim=-1) * L
    c = corr / energy.clamp_min(1e-30)
    best = c.max(dim=1)
    return best.values, best.indices

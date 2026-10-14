"""Parts 4/5: sliding-window stream, energy gate, ZC preamble detection, detection operating points."""
import numpy as np
import pytest
import torch

import main
from sync import (blind_sync, energy_threshold, gate_threshold, gate_windows, gated_sync,
                  noise_floor_threshold, window_energy, zc_detect, zc_sequence)

N = 512


@pytest.mark.parametrize("L", [31, 61])
def test_zc_sequence_properties(L):
    z = zc_sequence(L).to(torch.complex128)
    assert torch.allclose(z.abs(), torch.ones(L, dtype=torch.float64), atol=1e-6)
    for shift in (1, 5, L // 2):  # ideal periodic autocorrelation: zero off-peak
        assert abs((z * torch.roll(z, shift).conj()).sum()) < 1e-3


def test_window_stream_energy_accounting(link128):
    zc = zc_sequence(31)
    _, y, no_pre, offset = link128.window_stream(4, 384, 2.0, preamble=zc)
    _, _, no_blind, _ = link128.window_stream(4, 384, 2.0)
    assert y.shape == (4, 384 + 31 + 256)
    # same Eb/N0 counting preamble energy -> N0 larger by (31 + 256) / 256 (= 0.49 dB)
    assert float(no_pre / no_blind) == pytest.approx(287 / 256, rel=1e-5)
    assert offset.max() <= 384


def test_zc_detects_exact_offset_and_rejects_noise(link128):
    zc = zc_sequence(61)
    _, y, _, offset = link128.window_stream(200, 384, 0.0, preamble=zc)
    stat, est = zc_detect(y, zc, 384)
    assert (est == offset).float().mean() > 0.99
    _, y0, _, _ = link128.window_stream(200, 384, 0.0, preamble=zc, signal=False)
    stat0, _ = zc_detect(y0, zc, 384)
    # at Es/N0 ~ -3.9 dB the peak is ~0.29 +- 20%: typical (not every) frame beats the noise maximum
    assert stat.median() > stat0.max()


# --- Energy detector -----------------------------------------------------------

def test_window_energy_matches_direct_sum():
    y = torch.randn(3, 100, dtype=torch.complex64)
    e = window_energy(y, 20, 80)
    direct = torch.stack([(y[:, d:d + 20].abs() ** 2).sum(dim=1) for d in range(81)], dim=1)
    assert e.shape == (3, 81)
    assert torch.allclose(e, direct, rtol=1e-4)


@pytest.mark.parametrize("snr", [0.0, 3.0])
def test_energy_threshold_miss_rate(link128, snr):
    """Aligned-window energy falls below the threshold with the designed probability."""
    _, y, no, offset = link128.window_stream(20000, 384, snr)
    e = window_energy(y, 256, 384)[torch.arange(20000), offset]
    for miss in (0.05, 0.2):
        assert (e < energy_threshold(no, 256, miss)).float().mean().item() == pytest.approx(miss, abs=0.01)


def test_gated_sync_equals_ungated_when_everything_passes(link128):
    _, y, no, _ = link128.window_stream(6, 384, 1.0)
    llr = link128.demapper(y, no)
    g = gated_sync(link128, llr, torch.ones(6, 385, dtype=torch.bool), 8, step=2)
    b = blind_sync(link128, llr, 8, step=2)
    assert torch.equal(g["est_sc"], b["est_sc"]) and torch.equal(g["est_scl"], b["est_scl"])
    assert torch.equal(g["u_scl"], b["u_scl"])
    assert (g["sc_decodes"] == 385).all() and (g["scl_decodes"] == 8).all()


def test_gate_rejects_noise_and_keeps_frames(link128):
    _, y1, no, offset = link128.window_stream(300, 384, 3.0)
    _, y0, _, _ = link128.window_stream(300, 384, 3.0, signal=False)
    g = energy_threshold(no, 256, 1e-6)
    assert (window_energy(y1, 256, 384)[torch.arange(300), offset] >= g).all()
    assert not (window_energy(y0, 256, 384) >= g).any()


@pytest.mark.parametrize("snr", [0.0, 6.0])
def test_noise_floor_threshold_pass_rate(link128, snr):
    """Noise-only streams exceed the CFAR threshold with probability <= the design value."""
    _, y0, no, _ = link128.window_stream(20000, 384, snr, signal=False)
    g = noise_floor_threshold(no, 256, 384, 0.05)
    rate = (window_energy(y0, 256, 384) >= g).any(dim=1).float().mean().item()
    assert rate == pytest.approx(0.05, abs=0.006)  # Monte-Carlo calibrated, not a loose bound


@pytest.mark.parametrize("snr", [0.0, 2.0, 5.0, 8.0])
def test_gate_threshold_is_min_and_survives_3db_fade(link128, snr):
    _, y0, no, _ = link128.window_stream(20000, 384, snr, signal=False)
    g = gate_threshold(no, 256, 384, 1e-6, 3.0, 0.01)
    assert g == min(energy_threshold(no, 256, 1e-6, 10 ** -0.3), noise_floor_threshold(no, 256, 384, 0.01))
    # a frame 3 dB below nominal energy keeps its true window through the whole gate
    _, clean, _, off = link128.window_stream(20000, 384, 300.0)  # ~noiseless frame placement
    y = 10 ** (-3 / 20) * clean + y0
    passed = gate_windows(window_energy(y, 256, 384), no, 256, 384, 1e-6, 3.0, 0.01, 4.0)
    assert passed[torch.arange(20000), off].all()


@pytest.mark.parametrize("scheme", ["blind", "zc31"])
def test_window_task_high_snr(scheme):
    h1 = main._run_window_task((128, 5.0, 20, 3, scheme, "h1", 8))
    h0 = main._run_window_task((128, 5.0, 20, 4, scheme, "h0", 8))
    pre = "zc" if scheme.startswith("zc") else "scl"
    assert h1[f"{pre}_sync_ok"].all() and h1[f"{pre}_dec_ok"].all()
    if pre == "zc":
        assert h1["zc_stat"].min() > h0["zc_stat"].max()
        assert set(h0) == {"zc_stat"} and np.isneginf(h0["zc_stat"]).all()  # same energy gate as blind
    else:
        assert np.isneginf(h0["scl_stat"]).all() and (h0["sc_decodes"] == 0).all()  # gate: nothing decoded
        assert h1["gate_true_pass"].all() and h1["genie_scl_ok"].all() and h1["sc_sync_ok"].all()
        assert h1["sc_decodes"].mean() < 300  # relative peak rule trims the windows decoded


# --- Detection operating points ---------------------------------------------------

def test_detection_summary_operating_points():
    rng = np.random.default_rng(0)
    h0 = rng.normal(0, 1, 100000)
    h1 = rng.normal(4, 1, 20000)
    sync_ok = np.ones(20000, bool)
    sync_ok[:100] = False
    dec_ok = np.ones(20000, bool)
    s = main.detection_summary(h0, h1, sync_ok, dec_ok)
    assert s["p_false_realized_1e-3"] == pytest.approx(1e-3, abs=2e-4)
    from scipy.stats import norm
    assert s["p_miss_at_pfa_1e-3"] == pytest.approx(norm.cdf(norm.ppf(0.999) - 4), rel=0.15)
    # zero-miss threshold: every H1 sample is detected, P_false = P(H0 >= min H1)
    assert s["gamma_pmiss_0"] == h1.min()
    assert s["p_false_at_pmiss_0"] == pytest.approx(np.mean(h0 >= h1.min()))
    missed = h1 < s["gamma_pfa_1e-3"]
    assert s["bler_at_pfa_1e-3"] == pytest.approx(np.mean(missed | ~sync_ok))
    assert s["bler_at_pmiss_0"] == pytest.approx(100 / 20000)


def test_detection_summary_with_gate_rejections():
    h0 = np.full(10000, -np.inf)
    h0[:5] = 1.0  # 5 noise streams passed the gate
    h1 = np.linspace(2, 3, 1000)
    ok = np.ones(1000, bool)
    s = main.detection_summary(h0, h1, ok, ok)
    assert s["p_false_realized_1e-3"] == pytest.approx(5e-4)  # gate alone meets the budget
    assert s["p_miss_at_pfa_1e-3"] == 0.0 and s["p_false_at_pmiss_0"] == 0.0
    h1[0] = -np.inf  # a frame rejected by the gate makes P_miss = 0 unreachable
    s = main.detection_summary(h0, h1, ok, ok)
    assert s["h1_rejected_by_gate"] == 1 and s["p_false_at_pmiss_0"] == 1.0


def test_pmiss_target_threshold_misses_fewest_frames():
    """P_miss <= p allows misses, but on separable data none should be spent."""
    h0 = np.linspace(-10, -5, 2000)
    h1 = np.linspace(0, 5, 1000)
    ok = np.ones(1000, bool)
    s = main.detection_summary(h0, h1, ok, ok)
    assert s["p_false_at_pmiss_1e-2"] == 0.0 and s["bler_at_pmiss_1e-3"] == 0.0
    # overlapping data: the allowed misses are used only where they lower P_false
    h1 = np.concatenate([np.full(5, -7.0), np.linspace(0, 5, 995)])  # 5 frames look like noise
    s = main.detection_summary(h0, h1, ok, ok)
    assert s["p_false_at_pmiss_0"] == pytest.approx(np.mean(h0 >= -7.0))
    assert s["p_false_at_pmiss_1e-2"] == 0.0  # 10 misses allowed, 5 needed
    # 1 allowed miss cannot lower P_false (5 noise-like frames), so none is spent
    assert s["p_false_at_pmiss_1e-3"] == s["p_false_at_pmiss_0"] and s["bler_at_pmiss_1e-3"] == 0.0

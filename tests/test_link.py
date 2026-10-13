"""Unit and cross-validation tests for the Sionna polar link (polar.py)."""
import numpy as np
import pytest
import torch
from sionna.phy.fec.crc import CRCDecoder, CRCEncoder
from sionna.phy.fec.polar import PolarEncoder, PolarSCDecoder, PolarSCLDecoder

from legacy import utils_numpy as legacy_utils
from legacy.polar_numba import PolarCode
from polar import PolarLink
from utils import get_frozen_bits, get_frozen_positions

N = 512


def _noisy_llr(link, num, ebno_db):
    return link.frame(num, ebno_db)


# --- Code construction -------------------------------------------------------

@pytest.mark.parametrize("K", [128, 171, 256])
def test_frozen_set_size(K):
    fb = get_frozen_bits(N, K)
    assert (~fb).sum() == K
    assert fb[0] and not fb[N - 1]  # worst / best synthetic channel in natural order


def test_frozen_set_close_to_5g_ranking():
    """Bhattacharyya (natural order) must largely agree with the 3GPP reliability order."""
    from sionna.phy.fec.polar.utils import generate_5g_ranking
    for K in (128, 256):
        _, info_5g = generate_5g_ranking(K, N)
        info_bh = np.where(~get_frozen_bits(N, K))[0]
        assert len(np.intersect1d(info_5g, info_bh)) >= 0.9 * K


def test_bitreversal_regression(link128):
    """The pre-fix construction gave BLER ~ 1.0 at 2 dB; a correct one is < 1e-2."""
    bits, llr = _noisy_llr(link128, 400, 2.0)
    u_hat, _, _ = link128.scl(llr)
    bler = (u_hat[:, :link128.info_len] != bits).any(dim=1).float().mean().item()
    assert bler < 0.02


# --- Encoder / CRC / modulation vs. legacy ---------------------------------

@pytest.mark.parametrize("K", [128, 256])
def test_encoder_matches_legacy(K):
    fb = get_frozen_bits(N, K)
    enc = PolarEncoder(get_frozen_positions(N, K), N)
    legacy = PolarCode(N, fb)
    u = torch.randint(0, 2, (20, K)).float()
    x = enc(u).numpy().astype(int)
    for i in range(20):
        np.testing.assert_array_equal(x[i], legacy.encode(u[i].numpy().astype(int)))


def test_crc6_detects_errors():
    enc = CRCEncoder("CRC6")
    dec = CRCDecoder(enc)
    b = torch.randint(0, 2, (100, 122)).float()
    c = enc(b)
    assert c.shape[-1] == 128
    _, ok = dec(c)
    assert ok.all()
    c[:, 5] = 1 - c[:, 5]  # any single bit error is detected by a CRC
    _, ok = dec(c)
    assert not ok.any()


def test_qpsk_and_noise_match_legacy(link128):
    bits = torch.randint(0, 2, (50, N)).float()
    s = link128.mapper(bits)
    for i in range(5):
        np.testing.assert_allclose(s[i].numpy(), legacy_utils.qpsk_modulate(bits[i].numpy().astype(int)), atol=1e-6)

    for snr in (0.0, 3.0):
        no = float(link128.noise_var(snr))
        # Eb is per *information* bit: rate = (K - CRC) / N = 122 / 512 (legacy used K / N)
        _, sigma2 = legacy_utils.get_noise_sigma(snr, 122, N)
        assert no / 2 == pytest.approx(sigma2, rel=1e-6)  # per-real-dimension variance

        y = link128.channel(s, no)
        logits = link128.demapper(y, no)
        legacy_llr = legacy_utils.qpsk_demodulate_llr(y[0].numpy().astype(np.complex128), sigma2)
        # Sionna: logits log p1/p0. Legacy: log p0/p1 with a spurious factor 2
        # (harmless for min-sum decoding, fixed by the migration).
        np.testing.assert_allclose(logits[0].numpy(), -legacy_llr / 2, rtol=1e-4, atol=1e-4)


# --- Decoders -----------------------------------------------------------------

def test_noiseless_decoding(link128):
    bits, llr = _noisy_llr(link128, 50, 30.0)
    u_sc, _ = link128.sc(llr)
    u_scl, _, crc_ok = link128.scl(llr)
    assert torch.equal(u_sc[:, :link128.info_len], bits)
    assert torch.equal(u_scl[:, :link128.info_len], bits)
    assert crc_ok.all()


@pytest.mark.parametrize("ebno_db", [0.0, 1.5])
def test_scl_wrapper_matches_sionna(link128, ebno_db):
    """Our metric-returning SCL must make exactly Sionna's decisions."""
    _, llr = _noisy_llr(link128, 300, ebno_db)
    ref = PolarSCLDecoder(link128.frozen_pos, N, list_size=8, crc_degree="CRC6", return_crc_status=True)
    u_ref, crc_ref = ref(llr)
    u_hat, metric, crc_ok = link128.scl(llr)
    assert torch.equal(u_hat, u_ref)
    assert torch.equal(crc_ok, crc_ref)
    assert (metric <= 0).all()  # -PM, PM >= 0


def _reference_sc_frozen_sum(llr, frozen):
    """Independent NumPy SC decoder (exact boxplus) returning the frozen-LLR sum."""
    def boxplus(a, b):
        a, b = np.clip(a, -30, 30), np.clip(b, -30, 30)
        return np.log1p(np.exp(a + b)) - np.log(np.exp(a) + np.exp(b))

    def rec(l, fr):
        if len(l) == 1:
            u = 0 if fr[0] else int(l[0] < 0)
            return np.array([u]), np.array([u]), (l[0] if fr[0] else 0.0)
        h = len(l) // 2
        u1, x1, s1 = rec(boxplus(l[:h], l[h:]), fr[:h])
        u2, x2, s2 = rec((1 - 2 * x1) * l[:h] + l[h:], fr[h:])
        return np.concatenate([u1, u2]), np.concatenate([x1 ^ x2, x2]), s1 + s2

    return rec(llr, frozen)[2]


def test_sc_wrapper_matches_sionna_and_reference(link128):
    _, llr = _noisy_llr(link128, 40, 1.0)
    u_hat, metric = link128.sc(llr)
    assert torch.equal(u_hat, PolarSCDecoder(link128.frozen_pos, N)(llr))
    frozen = get_frozen_bits(N, 128)
    for i in range(10):
        ref = _reference_sc_frozen_sum(-llr[i].double().numpy(), frozen)
        assert metric[i].item() == pytest.approx(ref, rel=1e-3, abs=1e-2)


def test_scl_agrees_with_legacy_decoder():
    """Same LLRs, no CRC: Sionna SCL vs the legacy numba SCL (min-sum, no fast nodes).

    The decoders differ in approximations (exact vs min-sum boxplus, fast-SCL
    rate-1 nodes), so require agreement on the vast majority of frames and
    near-identical frame error counts.
    """
    from sionna.phy.mapping import BinarySource
    K = 128
    fp = get_frozen_positions(N, K)
    enc = PolarEncoder(fp, N)
    dec = PolarSCLDecoder(fp, N, list_size=8)
    legacy = PolarCode(N, get_frozen_bits(N, K))
    src_link = PolarLink(N, K, 8, "CRC6")
    u = BinarySource()([150, K])
    no = src_link.noise_var(1.0)
    llr = src_link.demapper(src_link.channel(src_link.mapper(enc(u)), no), no)
    u_sionna = dec(llr).numpy().astype(int)
    agree, err_s, err_l = 0, 0, 0
    for i in range(len(u)):
        u_leg = legacy.decode_scl_numba(-llr[i].double().numpy(), L=8, crc_poly=None)
        agree += np.array_equal(u_leg, u_sionna[i])
        err_s += not np.array_equal(u_sionna[i], u[i].numpy().astype(int))
        err_l += not np.array_equal(u_leg, u[i].numpy().astype(int))
    assert agree >= 0.9 * len(u)
    assert abs(err_s - err_l) <= max(5, 0.5 * max(err_s, err_l))


# --- Burst / metrics ---------------------------------------------------------

def test_burst_random_start_structure(link128):
    max_start = 4 * N
    _, llr, start = link128.burst(300, max_start, 20.0)
    assert llr.shape == (300, max_start + N)
    assert start.min() >= 0 and start.max() <= max_start
    assert (start % 2 == 1).any() and (start % 2 == 0).any()  # bit (not symbol) granularity
    pos = torch.arange(max_start + N)
    inside = (pos >= start[:, None]) & (pos < start[:, None] + N)
    mag = llr.abs()
    # signal bits have |LLR| ~ sqrt(Es/N0) times larger than silent (noise-only) bits
    assert ((mag * inside).sum(1) / N > 5 * (mag * ~inside).sum(1) / (max_start)).all()


def test_true_window_decodes_for_odd_and_even_starts(link128):
    from sync import candidate_windows
    bits, llr, start = link128.burst(64, 4 * N, 30.0)
    window = candidate_windows(llr, N)[torch.arange(64), start]
    u_hat, _, crc_ok = link128.scl(window)
    assert torch.equal(u_hat[:, :link128.info_len], bits) and crc_ok.all()
    assert (start % 2 == 1).any()


def test_blind_sync_exact_at_high_snr(link128):
    from sync import blind_sync
    bits, llr, start = link128.burst(10, 4 * N, 4.0)
    s = blind_sync(link128, llr, num_candidates=8)
    assert torch.equal(s["est_sc"], start) and torch.equal(s["est_scl"], start)
    assert torch.equal(s["u_scl"][:, :link128.info_len], bits)
    assert (s["candidates"] == start[:, None]).any(dim=1).all()


def test_blind_sync_exhaustive_mode(link128):
    from sync import blind_sync
    _, llr, start = link128.burst(1, 64, 3.0)  # short window keeps exhaustive CA-SCL cheap
    s = blind_sync(link128, llr, num_candidates=None)
    assert s["candidates"].shape == (1, 65)
    assert torch.equal(s["est_scl"], start)


def test_metrics_separate_h0_h1(link128):
    _, llr_h1 = _noisy_llr(link128, 200, 3.0)
    llr_h0 = link128.noise_only(200, 3.0)
    for dec in ("sc", "scl"):
        m1 = getattr(link128, dec)(llr_h1)[1]
        m0 = getattr(link128, dec)(llr_h0)[1]
        assert m1.median() > m0.quantile(0.99), dec


def test_burst_task_reproducible_and_sane():
    import main
    for mode in ("slots", "random"):
        a = main._run_burst_task((128, 4.0, 20, 7, mode, 8))
        b = main._run_burst_task((128, 4.0, 20, 7, mode, 8))
        assert a == b
        assert a["sync_ok_scl"] == 20 and a["blk_err_scl"] == 0 and a["blk_err_genie_scl"] == 0


@pytest.mark.parametrize("mode", ["slots", "random"])
def test_blind_errors_split_into_sync_and_decoding(mode):
    import main
    t = main._run_burst_task((128, 0.0, 60, 3, mode, 8))
    for dec in ("sc", "scl"):
        assert t[f"blk_err_{dec}"] == t[f"blk_err_{dec}_sync_fail"] + t[f"blk_err_{dec}_decoding"]
        # a wrong window essentially never decodes to the right bits
        assert t[f"blk_err_{dec}_sync_fail"] == 60 - t[f"sync_ok_{dec}"]
        # with correct sync the blind decode *is* the genie decode
        assert t[f"blk_err_{dec}_decoding"] <= t[f"blk_err_genie_{dec}"]


def test_slots_mode_starts_on_slot_boundaries(link128):
    _, llr, start = link128.burst(200, 4 * N, 3.0, step=N)
    assert llr.shape == (200, 5 * N)
    assert set(start.tolist()) == {0, N, 2 * N, 3 * N, 4 * N}

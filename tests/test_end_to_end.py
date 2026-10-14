"""End-to-end smoke test of main.py (multi-process, all three parts, tiny trial counts)."""
import json
import os

import pytest

import main

# polar_bler_genie.png / polar_bler_nosync.png are only drawn when genie and blind BLER differ
PLOTS = ["polar_ber.png", "polar_bler.png", "polar_sync_prob.png", "polar_hypo_pmiss.png", "polar_hypo_pfalse.png",
         "part1_bler.csv", "part3_detection.csv",
         "polar_window_pmiss.png", "polar_window_pfalse.png", "polar_window_bler.png", "part4_window.csv",
         "polar_window_complexity.png", "progress.txt",
         "polar_zc_pmiss.png", "polar_zc_pfalse.png", "polar_zc_bler.png", "part5_zc.csv"]


@pytest.mark.slow
def test_main_all_parts(tmp_path):
    out = str(tmp_path / "run")
    main.main(["--snr", "1", "4", "--detection-snr", "1", "4", "--iterations", "40", "--h0-trials", "200",
               "--h1-trials", "100", "--trials-per-task", "20", "--workers", "2",
               "--seed", "5", "--out", out])
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    for name in PLOTS:
        assert os.path.getsize(os.path.join(out, name)) > 0

    p1 = res["part1"]["128"]
    assert p1["trials"] == [40, 40]
    assert p1["bler_scl"][1] <= p1["bler_scl"][0]  # monotone in SNR (4 dB is error-free)
    # Part 2 reuses Part 1's K=128 trials
    assert res["part2"]["p_sync_scl"][1] >= 0.95
    assert set(res["part3"]) == {"128", "171", "256"}
    assert os.path.exists(os.path.join(out, "part3_metrics", "K171_snr4.npz"))
    assert set(res["part4"]) == {"128", "256"}
    assert res["part4"]["128"]["gate_true_pass"] == [1.0, 1.0]
    assert set(res["part5"]["schemes"]) == {"blind_sc", "blind_scl", "zc31", "zc61"}
    with open(os.path.join(out, "progress.txt")) as f:
        assert "100.0%" in f.read()
    assert res["part5"]["bler_genie_scl"][1] == 0.0  # 4 dB
    for name in ("part1_bler.csv", "part3_detection.csv", "part4_window.csv", "part5_zc.csv"):
        with open(os.path.join(out, name)) as f:
            widths = {len(line.split(",")) for line in f}
        assert len(widths) == 1, name  # every row matches the header

    main.main(["--replot", out])  # plots regenerate from saved results


@pytest.mark.slow
def test_early_stopping(tmp_path):
    out = str(tmp_path / "es")
    main.main(["--parts", "1", "--snr", "0", "--iterations", "4000", "--target-errors", "20",
               "--trials-per-task", "20", "--workers", "2", "--seed", "1", "--out", out])
    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    for K, r in res["part1"].items():
        rarest = min(r[f"blk_err_{d}"][0] for d in ("sc", "scl", "genie_sc", "genie_scl",
                                                     "sc_sync_fail", "scl_sync_fail"))
        # stop as soon as the rarest event reached the target, otherwise run to the cap
        assert (r["trials"][0] < 4000 and rarest >= 20) or (r["trials"][0] == 4000 and rarest < 20 + 160)
    assert res["part1"]["128"]["trials"][0] < 4000  # at 0 dB every K=128 event is frequent

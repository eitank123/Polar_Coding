"""Polar code experiments (Sionna).

Part 1: BER/BLER with blind burst sync (+ genie / sync-vs-decoding split)
Part 2: burst sync probability
Part 3: H0/H1 frame detection on aligned windows
Part 4: sliding-window sync (length-N windows, energy gate): P_miss, P_false, BLER incl. sync
Part 5: Zadoff-Chu preamble sync vs blind decoder-metric sync vs genie baseline

Examples:
    python main.py                        # full run, all parts, config.py settings
    python main.py --quick                # ~1-2 min smoke run with few trials
    python main.py --parts 3 --snr 0 2 4  # selected parts / SNR points
"""
import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from utils import available_cpus

# ---------------------------------------------------------------------------
# Worker side: each process runs single-threaded torch and caches one
# PolarLink per K (Sionna blocks are built once per process, not per task).
# ---------------------------------------------------------------------------
_LINKS = {}


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _get_link(K):
    if K not in _LINKS:
        from polar import PolarLink
        _LINKS[K] = PolarLink(cfg.N, K, cfg.L, cfg.CRC_DEGREE,
                              bits_per_symbol=cfg.BITS_PER_SYMBOL,
                              design_snr_db=cfg.FROZEN_DESIGN_SNR_DB,
                              precision=cfg.PRECISION)
    return _LINKS[K]


def _run_burst_task(args):
    """Burst trials: blind sync + decode with SC and CA-SCL.

    burst_mode "slots": packet in one of 5 aligned N-bit slots, both decoders score
    every slot. "random": packet at any start bit, two-stage search (see sync.py).
    Also decodes the true window ("genie" sync), so every blind block error can be
    attributed to a sync failure or to a decoding failure.
    """
    import torch
    from sionna.phy import config as sionna_config
    from sync import blind_sync, candidate_windows
    K, snr, num_trials, seed, burst_mode, num_candidates = args
    sionna_config.seed = seed
    link = _get_link(K)
    step = cfg.N if burst_mode == "slots" else 1
    if burst_mode == "slots":
        num_candidates = None  # CA-SCL scores all 5 slots, as in the original design

    with torch.no_grad():
        info_bits, llr, start = link.burst(num_trials, cfg.max_start_bit(burst_mode), snr, step=step)
        sync = blind_sync(link, llr, num_candidates, step=step)
        true_window = candidate_windows(llr, cfg.N)[torch.arange(num_trials), start]
        u_genie_sc, _ = link.sc(true_window)
        u_genie_scl, _, _ = link.scl(true_window)

    out = {"trials": num_trials}
    for name, u_hat in (("sc", sync["u_sc"]), ("scl", sync["u_scl"]),
                        ("genie_sc", u_genie_sc), ("genie_scl", u_genie_scl)):
        bit_err = (u_hat[:, :link.info_len] != info_bits).sum(dim=1)
        out[f"bit_err_{name}"] = int(bit_err.sum())
        out[f"blk_err_{name}"] = int((bit_err > 0).sum())
    for name in ("sc", "scl"):
        sync_ok = sync[f"est_{name}"] == start
        blk = (sync[f"u_{name}"][:, :link.info_len] != info_bits).any(dim=1)
        out[f"sync_ok_{name}"] = int(sync_ok.sum())
        out[f"blk_err_{name}_sync_fail"] = int((blk & ~sync_ok).sum())  # wrong window decoded
        out[f"blk_err_{name}_decoding"] = int((blk & sync_ok).sum())    # right window, decoder failed
    out["sc_candidates_hit"] = int((sync["candidates"] == start[:, None]).any(dim=1).sum())
    return out


def _run_detection_task(args):
    """CA-SCL sync metrics under H0 (noise only) or H1 (valid frame + noise)."""
    import torch
    from sionna.phy import config as sionna_config
    K, snr, num_trials, seed, hypothesis = args
    sionna_config.seed = seed
    link = _get_link(K)
    with torch.no_grad():
        if hypothesis == "h0":
            llr = link.noise_only(num_trials, snr)
        else:
            _, llr = link.frame(num_trials, snr)
        _, metric, _ = link.scl(llr)
    return metric.numpy().astype(np.float64)


def _run_window_task(args):
    """Sliding-window sync trials (Parts 4/5) under H0 (noise only) or H1 (frame present).

    scheme "blind": energy detector + decoder-metric search. The energy of every
        length-N candidate window (every QPSK symbol offset) is compared with the
        energy detector (sync.gate_windows); only passing windows are decoded:
        SC metric on all of them, CA-SCL -PM on the best `num_candidates`. The
        detection statistic is the max metric (-inf if no window passes, i.e. the
        stream is declared empty without decoding). H1 also decodes the true
        window (genie baseline).
    scheme "zc<L>": length-L Zadoff-Chu preamble before the frame, the same energy
        gate (over the preamble + data block), normalized correlation search,
        CA-SCL decoding at the detected start.
    Returns per-trial arrays: the detection statistic, the number of window
    decodes, and for H1 whether the estimated start / decoded frame are right.
    """
    import torch
    from sionna.phy import config as sionna_config
    from sync import candidate_windows, gate_windows, gated_sync, window_energy, zc_detect, zc_sequence

    def energy_gate(y, n_sym, no):
        energy = window_energy(y, n_sym, D)
        return energy, gate_windows(energy, no, n_sym, D, cfg.ENERGY_GATE_MISS_PROB, cfg.ENERGY_GATE_FADE_MARGIN_DB,
                                    cfg.ENERGY_GATE_NOISE_PASS_PROB, cfg.ENERGY_PEAK_KAPPA)
    K, snr, num_trials, seed, scheme, hypothesis, num_candidates = args
    sionna_config.seed = seed
    link = _get_link(K)
    D = cfg.WINDOW_MAX_OFFSET_SYM
    bps = cfg.BITS_PER_SYMBOL
    n_data = cfg.N // bps
    h1 = hypothesis == "h1"
    out = {}
    rows = torch.arange(num_trials)

    def ok(u_hat):
        return (u_hat[:, :link.info_len] == info_bits).all(dim=1).numpy()

    with torch.no_grad():
        if scheme == "blind":
            info_bits, y, no, offset = link.window_stream(num_trials, D, snr, signal=h1)
            llr = link.demapper(y, no)
            energy, passed = energy_gate(y, n_data, no)
            sync = gated_sync(link, llr, passed, num_candidates, step=bps)
            out["sc_decodes"] = sync["sc_decodes"].numpy()
            out["scl_decodes"] = sync["scl_decodes"].numpy()
            for dec in ("sc", "scl"):
                out[f"{dec}_stat"] = sync[f"{dec}_stat"].numpy().astype(np.float64)
                if h1:
                    out[f"{dec}_sync_ok"] = (sync["active"] & (sync[f"est_{dec}"] == offset * bps)).numpy()
                    out[f"{dec}_dec_ok"] = ok(sync[f"u_{dec}"])
            if h1:
                out["gate_true_pass"] = passed[rows, offset].numpy()
                true_window = candidate_windows(llr, cfg.N, bps)[rows, offset]
                out["genie_sc_ok"] = ok(link.sc(true_window)[0])
                out["genie_scl_ok"] = ok(link.scl(true_window)[0])
        else:
            zc = zc_sequence(int(scheme[2:]), cfg.ZC_ROOT)
            info_bits, y, no, offset = link.window_stream(num_trials, D, snr, preamble=zc, signal=h1)
            # same energy gate as the blind schemes, over the whole [preamble | data] block
            _, passed = energy_gate(y, zc.shape[0] + n_data, no)
            active = passed.any(dim=1)
            stat, est = zc_detect(y, zc, D)
            out["zc_stat"] = torch.where(active, stat.double(), torch.full_like(stat.double(), -np.inf)).numpy()
            if h1:
                out["zc_sync_ok"] = (active & (est == offset)).numpy()
                data = y.unfold(-1, n_data, 1)[rows, est + zc.shape[0]]
                out["zc_dec_ok"] = ok(link.scl(link.demapper(data, no))[0])
    return out


def detection_summary(h0_stat, h1_stat, sync_ok, dec_ok):
    """Operating points of a frame detector 'declare a frame iff statistic >= gamma'.

    A statistic of -inf means the stream was rejected before decoding (energy
    gate) and is never declared a frame.
    A: gamma = 99.9th pct of H0 (an actual sample, so gate-rejected -inf values are
       handled exactly) -> P_false <= 1e-3; report P_miss, P(detected at the wrong
       start) and BLER (block error = missed OR wrong start OR decoding failure).
    B: gamma = min of H1 -> empirical P_miss = 0; report P_false, P(wrong start)
       and BLER (no misses, so BLER = wrong start OR decoding failure). If the
       gate rejected any H1 stream, P_miss = 0 is unreachable: P_false = 1.
    """
    def declared(stat, gamma):
        return np.isfinite(stat) & (stat >= gamma)

    gamma_a = np.quantile(h0_stat, 0.999, method="higher")
    det_a = declared(h1_stat, gamma_a)
    good = sync_ok & dec_ok
    gate_missed = int(np.sum(~np.isfinite(h1_stat)))
    gamma_b = np.min(h1_stat)
    return {
        "gamma_pfa_1e-3": float(gamma_a),
        "p_false_realized_1e-3": float(np.mean(declared(h0_stat, gamma_a))),
        "p_miss_at_pfa_1e-3": float(np.mean(~det_a)),
        "p_wrong_start_at_pfa_1e-3": float(np.mean(det_a & ~sync_ok)),
        "bler_at_pfa_1e-3": float(np.mean(~(det_a & good))),
        "gamma_pmiss_0": float(gamma_b),
        "p_false_at_pmiss_0": 1.0 if gate_missed else float(np.mean(declared(h0_stat, gamma_b))),
        "p_wrong_start_at_pmiss_0": float(np.mean(~sync_ok)),
        "bler_at_pmiss_0": float(np.mean(~good)),
        "h1_rejected_by_gate": gate_missed,
    }


# ---------------------------------------------------------------------------
# Driver side
# ---------------------------------------------------------------------------
# Relative worker cost per trial of each task type (measured on this machine). Only
# ratios matter: the ETA is calibrated on the measured rate as the run progresses.
TASK_COST = {"burst": 22.0, "detect": 4.0, "window_h0": 3.0, "window_h1": 30.0,
             "zc_h0": 0.3, "zc_h1": 5.0}


def _fmt_seconds(sec):
    if not np.isfinite(sec):
        return "--:--:--"
    sec = int(sec)
    return f"{sec // 3600:d}:{sec // 60 % 60:02d}:{sec % 60:02d}"


class Progress:
    """Overall progress bar with percentage and ETA (stdout + <out>/progress.txt)."""

    def __init__(self, total, out_dir, every=15.0):
        self.total = max(total, 1e-9)
        self.done = 0.0
        self.t0 = time.time()
        self.every = every
        self.last = 0.0
        self.path = os.path.join(out_dir, "progress.txt")
        self.tty = sys.stdout.isatty()

    def add(self, weight, force=False):
        self.done = min(self.total, self.done + weight)
        now = time.time()
        if force or now - self.last >= self.every:
            self.last = now
            self.show()

    def line(self):
        frac = self.done / self.total
        elapsed = time.time() - self.t0
        eta = elapsed * (1 - frac) / frac if frac > 0 else float("inf")
        bar = "#" * int(40 * frac) + "-" * (40 - int(40 * frac))
        return f"[{bar}] {100 * frac:5.1f}% | elapsed {_fmt_seconds(elapsed)} | ETA {_fmt_seconds(eta)}"

    def show(self):
        text = self.line()
        if self.tty:
            print("\r" + text, end="", flush=True)
        else:
            print(text, flush=True)
        with open(self.path, "w") as f:
            f.write(text + "\n")


class Runner:
    def __init__(self, args):
        self.args = args
        self.snrs = [float(s) for s in args.snr]
        self.det_snrs = [float(s) for s in args.detection_snr]
        self.seq = np.random.SeedSequence(args.seed)
        self.out_dir = args.out
        os.makedirs(self.out_dir, exist_ok=True)
        self.results = {"meta": self._meta()}
        self._burst_cache = {}
        self._window_cache = {}
        self.progress = Progress(self._planned_cost(), self.out_dir)
        ctx = multiprocessing.get_context("spawn")  # torch + fork is unsafe
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=ctx, initializer=_worker_init)

    def _meta(self):
        import sionna
        import torch
        a = self.args
        return {
            "started": datetime.now().isoformat(timespec="seconds"),
            "host": platform.node(), "workers": a.workers,
            "seed": a.seed, "sionna": sionna.__version__, "torch": torch.__version__,
            "N": cfg.N, "L": cfg.L, "crc": cfg.CRC_DEGREE, "modulation": cfg.MODULATION,
            "frozen_design_snr_db": cfg.FROZEN_DESIGN_SNR_DB,
            "burst_mode": a.burst, "max_start_bit": cfg.max_start_bit(a.burst),
            "sync_scl_candidates": None if a.burst == "slots" else a.sync_candidates,
            "target_errors": a.target_errors,
            "detection_snr_db": self.det_snrs, "window_max_offset_sym": cfg.WINDOW_MAX_OFFSET_SYM,
            "energy_gate": {"miss_prob": cfg.ENERGY_GATE_MISS_PROB, "fade_margin_db": cfg.ENERGY_GATE_FADE_MARGIN_DB,
                            "noise_pass_prob": cfg.ENERGY_GATE_NOISE_PASS_PROB, "peak_kappa": cfg.ENERGY_PEAK_KAPPA},
            "eb_definition": "energy per information bit (CRC and preamble counted as overhead)",
            "zc_lengths": cfg.ZC_LENGTHS,
            "zc_root": cfg.ZC_ROOT,
            "snr_db": self.snrs,
            "iterations": a.iterations, "h0_trials": a.h0_trials, "h1_trials": a.h1_trials,
            "trials_per_task": a.trials_per_task, "precision": cfg.PRECISION,
        }

    def _planned_cost(self):
        """Total TASK_COST-weighted trials of the requested parts (upper bound: early stopping)."""
        a, n_snr, n_det = self.args, len(self.snrs), len(self.det_snrs)
        parts = set(a.parts)
        cost = 0.0
        burst_ks = set(cfg.K_LIST) if 1 in parts else set()
        if 2 in parts:
            burst_ks.add(cfg.PART2_K)
        cost += len(burst_ks) * n_snr * a.iterations * TASK_COST["burst"]
        if 3 in parts:
            cost += len(cfg.K_RATES) * n_det * (a.h0_trials * TASK_COST["detect"] + a.h1_trials * TASK_COST["detect"])
        window_ks = set(cfg.PART4_K_LIST) if 4 in parts else set()
        if 5 in parts:
            window_ks.add(cfg.PART5_K)
            cost += len(cfg.ZC_LENGTHS) * n_det * (a.h0_trials * TASK_COST["zc_h0"] + a.h1_trials * TASK_COST["zc_h1"])
        cost += len(window_ks) * n_det * (a.h0_trials * TASK_COST["window_h0"] + a.h1_trials * TASK_COST["window_h1"])
        return cost

    def _seeds(self, count):
        return [int(s.generate_state(1)[0]) for s in self.seq.spawn(count)]

    def _chunks(self, total):
        size = self.args.trials_per_task
        return [min(size, total - start) for start in range(0, total, size)]

    def save(self):
        path = os.path.join(self.out_dir, "results.json")
        with open(path + ".tmp", "w") as f:
            json.dump(self.results, f, indent=1)
        os.replace(path + ".tmp", path)

    @staticmethod
    def _rarest_event_count(total):
        """Smallest count among the error events Parts 1/2 report (drives early stopping)."""
        return min(total["blk_err_sc"], total["blk_err_scl"],
                   total["blk_err_genie_sc"], total["blk_err_genie_scl"],
                   total["trials"] - total["sync_ok_sc"], total["trials"] - total["sync_ok_scl"])

    def burst_point(self, K, snr):
        """Aggregated burst statistics for one (K, SNR); cached for Part 1/2 reuse.

        Tasks run in waves; the point stops early once every error event has been
        seen `target_errors` times (relative std ~ 1/sqrt(target)), else at
        `iterations` trials.
        """
        if (K, snr) not in self._burst_cache:
            t0 = time.time()
            a = self.args
            chunks = self._chunks(a.iterations)
            wave = 4 * a.workers
            total = {}
            for w in range(0, len(chunks), wave):
                sizes = chunks[w:w + wave]
                tasks = [(K, snr, n, s, a.burst, a.sync_candidates)
                         for n, s in zip(sizes, self._seeds(len(sizes)))]
                for res in self.executor.map(_run_burst_task, tasks):
                    for key, val in res.items():
                        total[key] = total.get(key, 0) + val
                    self.progress.add(res["trials"] * TASK_COST["burst"])
                if a.target_errors and self._rarest_event_count(total) >= a.target_errors:
                    # skipped trials count as done for the progress bar
                    self.progress.add((a.iterations - total["trials"]) * TASK_COST["burst"])
                    break
            total["seconds"] = round(time.time() - t0, 1)
            self._burst_cache[(K, snr)] = total
        return self._burst_cache[(K, snr)]

    # -- Part 1 -------------------------------------------------------------
    def part1(self):
        res = {}
        for K in cfg.K_LIST:
            info_len = K - cfg.CRC_LEN
            print(f"\n--- PART 1: BER/BLER (N={cfg.N}, K={K}, {cfg.MODULATION}, {self._burst_desc()}) ---",
                  flush=True)
            decs = ("sc", "scl", "genie_sc", "genie_scl")
            r = {"trials": []}
            for dec in decs:
                r.update({f"ber_{dec}": [], f"bler_{dec}": [], f"blk_err_{dec}": []})
            for dec in ("sc", "scl"):
                for cause in ("sync_fail", "decoding"):
                    r.update({f"blk_err_{dec}_{cause}": [], f"bler_{dec}_{cause}": []})
            for snr in self.snrs:
                t = self.burst_point(K, snr)
                n = t["trials"]
                r["trials"].append(n)
                for dec in decs:
                    r[f"blk_err_{dec}"].append(t[f"blk_err_{dec}"])
                    r[f"ber_{dec}"].append(t[f"bit_err_{dec}"] / (n * info_len))
                    r[f"bler_{dec}"].append(t[f"blk_err_{dec}"] / n)
                for dec in ("sc", "scl"):
                    for cause in ("sync_fail", "decoding"):
                        r[f"blk_err_{dec}_{cause}"].append(t[f"blk_err_{dec}_{cause}"])
                        r[f"bler_{dec}_{cause}"].append(t[f"blk_err_{dec}_{cause}"] / n)
                print(f"SNR {snr:4.1f} dB | BLER blind/genie: SC {r['bler_sc'][-1]:.3e}/{r['bler_genie_sc'][-1]:.3e} "
                      f"| CA-SCL {r['bler_scl'][-1]:.3e}/{r['bler_genie_scl'][-1]:.3e} "
                      f"({t['blk_err_scl']} err) | {n} trials in {t['seconds']} s", flush=True)
            res[str(K)] = r
            self.results["part1"] = res
            self.save()
        write_part1_csv(self.results, self.out_dir)
        plot_part1(self.results, self.out_dir)

    def _burst_desc(self):
        if self.args.burst == "slots":
            return f"packet in 1 of {cfg.SLOTS_MAX_START_BIT // cfg.N + 1} slots"
        return (f"random start bit 0..{cfg.RANDOM_MAX_START_BIT}, CA-SCL rescoring "
                f"{self.args.sync_candidates or 'all'} SC candidates")

    # -- Part 2 -------------------------------------------------------------
    def part2(self):
        K = cfg.PART2_K
        print(f"\n--- PART 2: Blind Frame Sync (N={cfg.N}, K={K}, {cfg.MODULATION}, {self._burst_desc()}) ---",
              flush=True)
        r = {"p_sync_sc": [], "p_sync_scl": [], "p_sc_candidates_hit": [], "trials": []}
        for snr in self.snrs:
            t = self.burst_point(K, snr)  # reuses Part 1 trials when K is in K_LIST
            r["trials"].append(t["trials"])
            r["p_sync_sc"].append(t["sync_ok_sc"] / t["trials"])
            r["p_sync_scl"].append(t["sync_ok_scl"] / t["trials"])
            r["p_sc_candidates_hit"].append(t["sc_candidates_hit"] / t["trials"])
            print(f"SNR {snr:4.1f} dB | SC sync prob {r['p_sync_sc'][-1]:.4f} | "
                  f"CA-SCL sync prob {r['p_sync_scl'][-1]:.4f} | {t['trials']} trials", flush=True)
        self.results["part2"] = {"K": K, **r}
        self.save()
        plot_part2(self.results, self.out_dir)

    # -- Parts 4/5 shared -----------------------------------------------------
    def _window_arrays(self, K, snr, scheme, hypothesis, total):
        chunks = self._chunks(total)
        tasks = [(K, snr, n, s, scheme, hypothesis, self.args.sync_candidates)
                 for n, s in zip(chunks, self._seeds(len(chunks)))]
        kind = ("window" if scheme == "blind" else "zc") + f"_{hypothesis}"
        parts = []
        for n, res in zip(chunks, self.executor.map(_run_window_task, tasks)):
            parts.append(res)
            self.progress.add(n * TASK_COST[kind])
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def window_point(self, K, snr, scheme):
        """H0/H1 per-trial arrays for one (K, SNR, scheme); cached so Part 5 reuses Part 4."""
        if (K, snr, scheme) not in self._window_cache:
            t0 = time.time()
            h0 = self._window_arrays(K, snr, scheme, "h0", self.args.h0_trials)
            h1 = self._window_arrays(K, snr, scheme, "h1", self.args.h1_trials)
            d = os.path.join(self.out_dir, "window_metrics")
            os.makedirs(d, exist_ok=True)
            np.savez_compressed(os.path.join(d, f"{scheme}_K{K}_snr{snr:g}.npz"),
                                **{f"h0_{k}": v for k, v in h0.items()}, **{f"h1_{k}": v for k, v in h1.items()})
            self._window_cache[(K, snr, scheme)] = (h0, h1, round(time.time() - t0, 1))
        return self._window_cache[(K, snr, scheme)]

    @staticmethod
    def _append(store, summary):
        for key, val in summary.items():
            store.setdefault(key, []).append(val)

    # -- Part 4 -------------------------------------------------------------
    def part4(self):
        D = cfg.WINDOW_MAX_OFFSET_SYM
        print(f"\n--- PART 4: Sliding-window sync, length-N windows (N={cfg.N}, frame start 0..{D} symbols, "
              f"energy gate + SC scan + CA-SCL on {self.args.sync_candidates or 'all'} SC candidates) ---", flush=True)
        res = {}
        for K in cfg.PART4_K_LIST:
            print(f"\nK={K}", flush=True)
            r = {"sc": {}, "scl": {}, "bler_genie_sc": [], "bler_genie_scl": [],
                 "sc_decodes_per_stream_h0": [], "sc_decodes_per_stream_h1": [],
                 "streams_decoded_h0": [], "gate_true_pass": []}
            for snr in self.det_snrs:
                h0, h1, secs = self.window_point(K, snr, "blind")
                for dec in ("sc", "scl"):
                    self._append(r[dec], detection_summary(h0[f"{dec}_stat"], h1[f"{dec}_stat"],
                                                           h1[f"{dec}_sync_ok"], h1[f"{dec}_dec_ok"]))
                r["bler_genie_sc"].append(float(np.mean(~h1["genie_sc_ok"])))
                r["bler_genie_scl"].append(float(np.mean(~h1["genie_scl_ok"])))
                r["sc_decodes_per_stream_h0"].append(float(np.mean(h0["sc_decodes"])))
                r["sc_decodes_per_stream_h1"].append(float(np.mean(h1["sc_decodes"])))
                r["streams_decoded_h0"].append(float(np.mean(h0["sc_decodes"] > 0)))
                r["gate_true_pass"].append(float(np.mean(h1["gate_true_pass"])))
                a, b = r["sc"], r["scl"]
                print(f"SNR {snr:4.1f} dB | P_miss@Pfa=1e-3 SC {a['p_miss_at_pfa_1e-3'][-1]:.2e} CA-SCL "
                      f"{b['p_miss_at_pfa_1e-3'][-1]:.2e} | Pfa@Pmiss=0 SC {a['p_false_at_pmiss_0'][-1]:.2e} CA-SCL "
                      f"{b['p_false_at_pmiss_0'][-1]:.2e} | BLER@Pmiss=0 SC {a['bler_at_pmiss_0'][-1]:.2e} CA-SCL "
                      f"{b['bler_at_pmiss_0'][-1]:.2e} genie {r['bler_genie_scl'][-1]:.2e} | windows decoded/stream "
                      f"H0 {r['sc_decodes_per_stream_h0'][-1]:.1f} H1 {r['sc_decodes_per_stream_h1'][-1]:.1f} of {D + 1} "
                      f"| {secs} s", flush=True)
            res[str(K)] = r
            self.results["part4"] = res
            self.save()
        write_window_csv(self.results, self.out_dir)
        plot_part4(self.results, self.out_dir)

    # -- Part 5 -------------------------------------------------------------
    def part5(self):
        K = cfg.PART5_K
        n_data = cfg.N // cfg.BITS_PER_SYMBOL
        print(f"\n--- PART 5: ZC preamble vs blind sync (N={cfg.N}, K={K}, "
              f"equal total energy per information bit) ---", flush=True)
        schemes = {"blind_sc": {}, "blind_scl": {}}
        overhead = {"blind_sc": {"preamble_symbols": 0, "time_fraction": 0.0, "energy_penalty_db": 0.0},
                    "blind_scl": {"preamble_symbols": 0, "time_fraction": 0.0, "energy_penalty_db": 0.0}}
        for L in cfg.ZC_LENGTHS:
            schemes[f"zc{L}"] = {}
            overhead[f"zc{L}"] = {"preamble_symbols": L, "time_fraction": L / (L + n_data),
                                  "energy_penalty_db": float(10 * np.log10((L + n_data) / n_data))}
            print(f"ZC-{L}: {L} preamble symbols = {100 * L / (L + n_data):.1f}% of air time, "
                  f"{overhead[f'zc{L}']['energy_penalty_db']:.2f} dB energy overhead", flush=True)
        genie = []
        for snr in self.det_snrs:
            h0, h1, _ = self.window_point(K, snr, "blind")  # shared with Part 4
            for dec in ("sc", "scl"):
                self._append(schemes[f"blind_{dec}"], detection_summary(
                    h0[f"{dec}_stat"], h1[f"{dec}_stat"], h1[f"{dec}_sync_ok"], h1[f"{dec}_dec_ok"]))
            genie.append(float(np.mean(~h1["genie_scl_ok"])))
            for L in cfg.ZC_LENGTHS:
                z0, z1, _ = self.window_point(K, snr, f"zc{L}")
                self._append(schemes[f"zc{L}"], detection_summary(
                    z0["zc_stat"], z1["zc_stat"], z1["zc_sync_ok"], z1["zc_dec_ok"]))
            line = " | ".join(f"{n} Pmiss {v['p_miss_at_pfa_1e-3'][-1]:.1e} Pfa {v['p_false_at_pmiss_0'][-1]:.1e} "
                              f"BLER {v['bler_at_pmiss_0'][-1]:.1e}" for n, v in schemes.items())
            print(f"SNR {snr:4.1f} dB | {line} | genie BLER {genie[-1]:.1e}", flush=True)
            self.results["part5"] = {"K": K, "schemes": schemes, "bler_genie_scl": genie, "overhead": overhead}
            self.save()
        write_zc_csv(self.results, self.out_dir)
        plot_part5(self.results, self.out_dir)

    # -- Part 3 -------------------------------------------------------------
    def _metrics(self, K, snr, total, hypothesis):
        chunks = self._chunks(total)
        tasks = [(K, snr, n, s, hypothesis) for n, s in zip(chunks, self._seeds(len(chunks)))]
        out = []
        for n, res in zip(chunks, self.executor.map(_run_detection_task, tasks)):
            out.append(res)
            self.progress.add(n * TASK_COST["detect"])
        return np.concatenate(out)

    def part3(self):
        print(f"\n--- PART 3: Hypothesis Testing (N={cfg.N}, CA-SCL L={cfg.L}) ---", flush=True)
        res = {}
        hist_dir = os.path.join(self.out_dir, "part3_metrics")
        os.makedirs(hist_dir, exist_ok=True)
        for K in cfg.K_RATES:
            print(f"\nRate {K}/{cfg.N} (K={K})", flush=True)
            r = {"p_miss_at_pfa_1e-3": [], "p_false_at_zero_pmiss": [],
                 "gamma_fixed_pfalse": [], "gamma_zero_pmiss": []}
            for snr in self.det_snrs:
                t0 = time.time()
                m0 = self._metrics(K, snr, self.args.h0_trials, "h0")
                m1 = self._metrics(K, snr, self.args.h1_trials, "h1")
                np.savez_compressed(os.path.join(hist_dir, f"K{K}_snr{snr:g}.npz"), h0=m0, h1=m1)

                # Same detector rule as Parts 4/5: declare a frame iff metric >= gamma
                # Scenario A: gamma = 99.9th percentile of H0 -> P_false = 1e-3
                gamma_a = np.quantile(m0, 0.999, method="higher")
                p_miss = np.mean(m1 < gamma_a)
                # Scenario B: gamma = smallest H1 metric -> empirical P_miss = 0
                gamma_b = np.min(m1)
                p_false = np.mean(m0 >= gamma_b)

                r["p_miss_at_pfa_1e-3"].append(float(p_miss))
                r["p_false_at_zero_pmiss"].append(float(p_false))
                r["gamma_fixed_pfalse"].append(float(gamma_a))
                r["gamma_zero_pmiss"].append(float(gamma_b))
                print(f"SNR {snr:4.1f} dB | P_miss (P_false=1e-3) {p_miss:.4e} | "
                      f"P_false (P_miss=0) {p_false:.4e} | {time.time() - t0:.1f} s", flush=True)
            res[str(K)] = r
            self.results["part3"] = res
            self.save()
        write_part3_csv(self.results, self.out_dir)
        plot_part3(self.results, self.out_dir)


# ---------------------------------------------------------------------------
# Plotting (from the saved results dict, so plots can be regenerated)
# ---------------------------------------------------------------------------
def _rate_str(K):
    return {128: "1/4", 171: "1/3", 256: "1/2"}.get(K, f"{K}/{cfg.N}")


def _full_x(snr, ax=None):
    """Show the whole SNR sweep even where values are 0 (not drawable on a log axis)."""
    ax = ax or plt.gca()
    ax.set_xlim(min(snr) - 0.25, max(snr) + 0.25)
    ax.set_xticks(snr)


def _masked(values):
    """Zero error rates can't be drawn on a log axis; plot them as gaps."""
    v = np.asarray(values, dtype=float)
    return np.where(v > 0, v, np.nan)


def plot_part1(results, out_dir):
    snr = results["meta"]["snr_db"]
    L_ = results["meta"]["L"]
    for metric, ylabel, ylim in (("ber", "Bit Error Rate (BER)", 1e-7),
                                 ("bler", "Block Error Rate (BLER)", 1e-5)):
        plt.figure(figsize=(10, 6))
        for K, r in results["part1"].items():
            plt.semilogy(snr, _masked(r[f"{metric}_sc"]), "--o", label=f"SC {metric.upper()} (K={K})")
            plt.semilogy(snr, _masked(r[f"{metric}_scl"]), "-s", label=f"CA-SCL {metric.upper()} (K={K})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlabel("E_b/N_0 (dB)")
        plt.ylabel(ylabel)
        plt.ylim([ylim, 1])
        plt.legend()
        plt.title(f"Polar Code {metric.upper()} vs SNR (N={cfg.N})")
        _full_x(snr)
        plt.savefig(os.path.join(out_dir, f"polar_{metric}.png"), dpi=300)
        plt.close()

    if not all("bler_genie_sc" in r for r in results["part1"].values()):
        return  # results from before genie decoding was added

    # Blind vs genie (true start known) BLER: the gap is the cost of synchronization
    plt.figure(figsize=(10, 6))
    for K, r in results["part1"].items():
        for dec, label, style in (("sc", "SC", "o"), ("scl", "CA-SCL", "s")):
            line, = plt.semilogy(snr, _masked(r[f"bler_{dec}"]), f"-{style}", label=f"{label} (K={K})")
            plt.semilogy(snr, _masked(r[f"bler_genie_{dec}"]), f":{style}", color=line.get_color(),
                         mfc="none", label=f"{label} genie (K={K})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("E_b/N_0 (dB)")
    plt.ylabel("Block Error Rate (BLER)")
    plt.ylim([1e-5, 1])
    plt.legend(fontsize=8)
    plt.title(f"BLER: blind vs genie synchronization (N={cfg.N})")
    _full_x(snr)
    plt.savefig(os.path.join(out_dir, "polar_bler_genie.png"), dpi=300)
    plt.close()

    # Decoding-only BLER: true packet start known, i.e. block errors without sync errors
    plt.figure(figsize=(10, 6))
    for K, r in results["part1"].items():
        plt.semilogy(snr, _masked(r["bler_genie_sc"]), "--o", label=f"SC (K={K})")
        plt.semilogy(snr, _masked(r["bler_genie_scl"]), "-s", label=f"CA-SCL (K={K})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("E_b/N_0 (dB)")
    plt.ylabel("Block Error Rate (BLER), true start known")
    plt.ylim([1e-5, 1])
    plt.legend()
    plt.title(f"BLER without synchronization errors (N={cfg.N})")
    _full_x(snr)
    plt.savefig(os.path.join(out_dir, "polar_bler_nosync.png"), dpi=300)
    plt.close()


def write_part1_csv(results, out_dir):
    """Per (K, SNR) table: blind BLER split into sync-failure and decoding parts, plus genie BLER."""
    cols = ["trials"] + [f"{q}_{d}" for d in ("sc", "scl") for q in
                         ("bler", "bler_genie", "bler_sync_fail", "bler_decoding", "ber", "ber_genie")]
    with open(os.path.join(out_dir, "part1_bler.csv"), "w") as f:
        f.write("K,snr_db," + ",".join(cols) + "\n")
        for K, r in results["part1"].items():
            for i, snr in enumerate(results["meta"]["snr_db"]):
                vals = []
                for c in cols:
                    if c == "trials":
                        vals.append(str(r["trials"][i]))
                        continue
                    q, d = c.rsplit("_", 1)
                    key = {"bler_genie": f"bler_genie_{d}", "ber_genie": f"ber_genie_{d}",
                           "bler_sync_fail": f"bler_{d}_sync_fail", "bler_decoding": f"bler_{d}_decoding"
                           }.get(q, f"{q}_{d}")
                    vals.append(f"{r[key][i]:.6e}")
                f.write(f"{K},{snr:g}," + ",".join(vals) + "\n")


def plot_part2(results, out_dir):
    snr = results["meta"]["snr_db"]
    r = results["part2"]
    plt.figure(figsize=(10, 6))
    plt.plot(snr, r["p_sync_sc"], "--o", label=f"SC (K={r['K']})")
    plt.plot(snr, r["p_sync_scl"], "-s", label=f"CA-SCL (K={r['K']})")
    m = results["meta"]
    if m.get("burst_mode") == "random":
        plt.plot(snr, r["p_sc_candidates_hit"], ":", color="gray",
                 label="CA-SCL upper bound (SC pre-selection)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("E_b/N_0 (dB)")
    plt.ylabel("Probability of Correct Synchronization")
    plt.ylim([0, 1.05])
    plt.legend()
    plt.title(f"Synchronization Probability vs SNR (N={cfg.N})")
    _full_x(snr)
    plt.savefig(os.path.join(out_dir, "polar_sync_prob.png"), dpi=300)
    plt.close()


def write_part3_csv(results, out_dir):
    """Per (K, SNR) detection operating points; raw H0/H1 scores stay in part3_metrics/*.npz."""
    cols = ["p_miss_at_pfa_1e-3", "p_false_at_zero_pmiss", "gamma_fixed_pfalse", "gamma_zero_pmiss"]
    with open(os.path.join(out_dir, "part3_detection.csv"), "w") as f:
        f.write("K,snr_db," + ",".join(cols) + "\n")
        for K, r in results["part3"].items():
            for i, snr in enumerate(results["meta"].get("detection_snr_db", results["meta"]["snr_db"])):
                f.write(f"{K},{snr:g}," + ",".join(f"{r[c][i]:.6e}" for c in cols) + "\n")


def plot_part3(results, out_dir):
    snr = results["meta"].get("detection_snr_db", results["meta"]["snr_db"])
    for key, fname, marker, ylabel, title in (
            ("p_miss_at_pfa_1e-3", "polar_hypo_pmiss.png", "-o",
             "$P_{miss}$ (for fixed $P_{false} = 10^{-3}$)", "Probability of Missed Detection vs SNR"),
            ("p_false_at_zero_pmiss", "polar_hypo_pfalse.png", "-s",
             "$P_{false}$ (for empirical $P_{miss} = 0$)", "Probability of False Alarm vs SNR (Zero Miss Detection)")):
        plt.figure(figsize=(10, 6))
        for K, r in results["part3"].items():
            plt.semilogy(snr, _masked(r[key]), marker, label=f"Rate {_rate_str(int(K))} (K={K})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlabel("E_b/N_0 (dB)")
        plt.ylabel(ylabel)
        plt.ylim([1e-7, 1.1])
        plt.legend()
        plt.title(f"{title} (N={cfg.N})")
        _full_x(snr)
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.close()


_OP_POINTS = (
    ("p_miss_at_pfa_1e-3", "$P_{miss}$ ($P_{false}=10^{-3}$)", "pmiss"),
    ("p_false_at_pmiss_0", "$P_{false}$ ($P_{miss}=0$)", "pfalse"),
    ("bler_at_pmiss_0", "BLER incl. sync errors ($P_{miss}=0$)", "bler"),
)


def plot_part4(results, out_dir):
    snr = results["meta"].get("detection_snr_db", results["meta"]["snr_db"])
    D = results["meta"]["window_max_offset_sym"]
    for key, ylabel, tag in _OP_POINTS:
        plt.figure(figsize=(10, 6))
        for K, r in results["part4"].items():
            line, = plt.semilogy(snr, _masked(r["sc"][key]), "--o", label=f"SC (K={K})")
            plt.semilogy(snr, _masked(r["scl"][key]), "-s", label=f"CA-SCL (K={K})")
            if tag == "bler":
                plt.semilogy(snr, _masked(r["bler_genie_scl"]), ":", marker="^", mfc="none",
                             label=f"Genie CA-SCL (K={K})")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlabel("E_b/N_0 (dB)")
        plt.ylabel(ylabel)
        plt.ylim([1e-5, 1.1])
        plt.legend(fontsize=8)
        plt.title(f"Sliding-Window Synchronization (N={cfg.N})")
        _full_x(snr)
        plt.savefig(os.path.join(out_dir, f"polar_window_{tag}.png"), dpi=300)
        plt.close()

    # Complexity: SC window decodes per stream with the energy gate vs the ungated search
    plt.figure(figsize=(10, 6))
    for K, r in results["part4"].items():
        plt.plot(snr, r["sc_decodes_per_stream_h1"], "-o", label=f"Frame present (K={K})")
        plt.plot(snr, r["sc_decodes_per_stream_h0"], "--s", label=f"Noise only (K={K})")
    plt.axhline(D + 1, color="gray", ls=":", label="Without energy detector")
    plt.grid(True, ls="--", alpha=0.5)
    plt.xlabel("E_b/N_0 (dB)")
    plt.ylabel("Decoded windows per stream")
    plt.legend(fontsize=8)
    plt.title("Decoding Complexity with Energy Detector")
    _full_x(snr)
    plt.savefig(os.path.join(out_dir, "polar_window_complexity.png"), dpi=300)
    plt.close()


def plot_part5(results, out_dir):
    snr = results["meta"].get("detection_snr_db", results["meta"]["snr_db"])
    r = results["part5"]
    for key, ylabel, tag in _OP_POINTS:
        plt.figure(figsize=(11, 6.5))
        for name, v in r["schemes"].items():
            o = r["overhead"][name]
            if name.startswith("zc"):
                label = f"ZC-{o['preamble_symbols']} preamble"
                style = "-^" if o["preamble_symbols"] < 50 else "-v"
            else:
                label = f"Blind {'SC' if name == 'blind_sc' else 'CA-SCL'}"
                style = "--o" if name == "blind_sc" else "-s"
            plt.semilogy(snr, _masked(v[key]), style, label=label)
        if tag == "bler":
            plt.semilogy(snr, _masked(r["bler_genie_scl"]), ":", color="black", label="Genie CA-SCL")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.xlabel("E_b/N_0 (dB), incl. preamble energy")
        plt.ylabel(ylabel)
        plt.ylim([1e-5, 1.1])
        plt.legend(fontsize=8)
        plt.title(f"Preamble vs Blind Synchronization (N={cfg.N}, K={r['K']})")
        _full_x(snr)
        plt.savefig(os.path.join(out_dir, f"polar_zc_{tag}.png"), dpi=300)
        plt.close()


_SUMMARY_COLS = ["p_miss_at_pfa_1e-3", "p_wrong_start_at_pfa_1e-3", "bler_at_pfa_1e-3", "p_false_realized_1e-3",
                 "p_false_at_pmiss_0", "p_wrong_start_at_pmiss_0", "bler_at_pmiss_0", "h1_rejected_by_gate",
                 "gamma_pfa_1e-3", "gamma_pmiss_0"]


def write_window_csv(results, out_dir):
    extra = ["bler_genie", "sc_decodes_per_stream_h0", "sc_decodes_per_stream_h1", "streams_decoded_h0",
             "gate_true_pass"]
    with open(os.path.join(out_dir, "part4_window.csv"), "w") as f:
        f.write("K,snr_db,scheme," + ",".join(_SUMMARY_COLS + extra) + "\n")
        for K, r in results["part4"].items():
            for i, snr in enumerate(results["meta"].get("detection_snr_db", results["meta"]["snr_db"])):
                for dec in ("sc", "scl"):
                    vals = [f"{r[dec][c][i]:.6e}" for c in _SUMMARY_COLS]
                    vals += [f"{r[f'bler_genie_{dec}'][i]:.6e}"] + [f"{r[c][i]:.6e}" for c in extra[1:]]
                    f.write(f"{K},{snr:g},blind_{dec}," + ",".join(vals) + "\n")


def write_zc_csv(results, out_dir):
    r = results["part5"]
    with open(os.path.join(out_dir, "part5_zc.csv"), "w") as f:
        f.write("snr_db,scheme,preamble_symbols,overhead_time_fraction,overhead_energy_db,"
                + ",".join(_SUMMARY_COLS) + "\n")
        for i, snr in enumerate(results["meta"].get("detection_snr_db", results["meta"]["snr_db"])):
            for name, v in r["schemes"].items():
                o = r["overhead"][name]
                vals = ",".join(f"{v[c][i]:.6e}" for c in _SUMMARY_COLS)
                f.write(f"{snr:g},{name},{o['preamble_symbols']},{o['time_fraction']:.4f},"
                        f"{o['energy_penalty_db']:.3f},{vals}\n")
            row = {c: "" for c in _SUMMARY_COLS}
            row["bler_at_pmiss_0"] = row["bler_at_pfa_1e-3"] = f"{r['bler_genie_scl'][i]:.6e}"
            f.write(f"{snr:g},genie_scl,0,0,0," + ",".join(row[c] for c in _SUMMARY_COLS) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", type=int, nargs="+", default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5])
    p.add_argument("--snr", type=float, nargs="+", default=list(cfg.SNR_DB), help="Eb/N0 points [dB]")
    p.add_argument("--detection-snr", type=float, nargs="+", default=list(cfg.DETECTION_SNR_DB),
                   help="Eb/N0 points [dB] for the detection parts 3-5")
    p.add_argument("--iterations", type=int, default=cfg.ITERATIONS, help="burst trials per SNR point (Parts 1/2)")
    p.add_argument("--h0-trials", type=int, default=cfg.NUM_H0_TRIALS)
    p.add_argument("--h1-trials", type=int, default=cfg.NUM_H1_TRIALS)
    p.add_argument("--trials-per-task", type=int, default=cfg.TRIALS_PER_TASK)
    p.add_argument("--burst", choices=["slots", "random"], default=cfg.BURST_MODE,
                   help="packet start: one of 5 aligned slots, or any bit (see config.py)")
    p.add_argument("--target-errors", type=lambda v: None if v == "none" else int(v), default=cfg.TARGET_ERRORS,
                   help="stop an SNR point once every error event occurred this often ('none' = always full)")
    p.add_argument("--sync-candidates", type=lambda v: None if v == "all" else int(v),
                   default=cfg.SYNC_SCL_CANDIDATES,
                   help="random mode: SC-scan offsets re-scored by CA-SCL, or 'all' (exhaustive, ~1 s/trial)")
    p.add_argument("--workers", type=int, default=available_cpus())
    p.add_argument("--seed", type=int, default=None, help="master seed (random if omitted; recorded in results.json)")
    p.add_argument("--out", default=None, help="output directory (default results/<timestamp>)")
    p.add_argument("--quick", action="store_true", help="smoke run: 1000 burst trials, 4000/2000 H0/H1 trials")
    p.add_argument("--replot", metavar="RESULTS_DIR", help="only regenerate plots from RESULTS_DIR/results.json")
    a = p.parse_args(argv)
    if a.quick:
        a.iterations, a.h0_trials, a.h1_trials = 1000, 4000, 2000
    if a.seed is None:
        a.seed = int.from_bytes(os.urandom(4), "little")
    if a.out is None:
        a.out = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S") + ("_quick" if a.quick else ""))
    return a


def main(argv=None):
    args = parse_args(argv)
    if args.replot:
        with open(os.path.join(args.replot, "results.json")) as f:
            results = json.load(f)
        if "part1" in results and all("bler_genie_sc" in r for r in results["part1"].values()):
            write_part1_csv(results, args.replot)
        if "part3" in results:
            write_part3_csv(results, args.replot)
        if "part4" in results:
            write_window_csv(results, args.replot)
        if "part5" in results:
            write_zc_csv(results, args.replot)
        for part, fn in (("part1", plot_part1), ("part2", plot_part2), ("part3", plot_part3),
                         ("part4", plot_part4), ("part5", plot_part5)):
            if part in results:
                fn(results, args.replot)
        return

    runner = Runner(args)
    print(f"Output: {args.out} | workers={args.workers} | seed={args.seed}", flush=True)
    t0 = time.time()
    try:
        for part in sorted(args.parts):
            getattr(runner, f"part{part}")()
    finally:
        runner.executor.shutdown()
    runner.progress.done = runner.progress.total
    runner.progress.show()
    if runner.progress.tty:
        print()
    runner.results["meta"]["total_seconds"] = round(time.time() - t0, 1)
    runner.save()
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Results in {args.out}", flush=True)


if __name__ == "__main__":
    main()

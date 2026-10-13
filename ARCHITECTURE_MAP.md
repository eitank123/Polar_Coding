# Architecture Map — Polar_Coding

Reference this file in future sessions instead of re-scanning the repo. Regenerate it if files are added/removed or responsibilities shift significantly.

## Purpose

A simulation/research codebase for **Polar Codes** over an AWGN/QPSK channel, evaluating:
1. BER/BLER performance of SC vs CRC-Aided SCL (CA-SCL) decoding.
2. Blind frame synchronization (unknown timing offset) using decoder soft-output metrics.
3. Binary hypothesis testing (signal-present vs noise-only) for frame detection, i.e. P_miss / P_false tradeoffs.

There is no package structure — it's a flat set of top-level scripts/modules meant to be run directly (e.g. via `python main.py`), likely executed on the BGU HPC/Run:AI cluster (see `README_BGU_HPC_Claude.md`).

## File Map & Responsibilities (post-Sionna migration, 2026-10-13)

- **`config.py`** — all constants: `N`, `K_LIST` (Part 1), `K_RATES` (Part 3), `PART2_K`, `L`, `CRC_DEGREE="CRC6"` (Sionna/3GPP x^6+x^5+1), `SNR_DB`, `ITERATIONS`, `BURST_MODE` ("slots" default | "random"), `MAX_START_BIT` (=4N), `SYNC_SCL_CANDIDATES`, `TARGET_ERRORS` (early stop), `NUM_H0/H1_TRIALS`, `TRIALS_PER_TASK`, `PRECISION`.
- **`utils.py`** — `get_frozen_bits` (Bhattacharyya, **natural order** — bit-reversal fix), `get_frozen_positions`, `available_cpus` (cgroup quota).
- **`polar.py`** — Sionna (PyTorch) link. `PolarLink(N,K,...)`: BinarySource → CRCEncoder → PolarEncoder → QAM4 Mapper → AWGN → APP Demapper; `burst(step=1|N)` (packet at a uniform random start in {0, step, ..., MAX_START_BIT}; step=N = 1-in-5 slots inside a MAX_START_BIT+N-bit window, silence elsewhere; odd offsets put bit 0 on the Q rail), `frame()` (aligned, Part 3 H1), `noise_only()` (Part 3 H0). Decoders `PolarSCDecoderWithMetric` (returns frozen-leaf LLR sum) and `PolarSCLDecoderWithMetric` (returns -PM of CRC-selected path, uses Sionna private `_decode`; pinned to sionna 2.2.0).
- **`sync.py`** — blind sync over windows every `step` bits (slots: SC and CA-SCL score all 5); random mode two-stage: SC frozen-sum metric on every offset (`unfold` windows), then CA-SCL -PM on the top `num_candidates` (None = exhaustive, ~1 s/trial).
- **Parts 4/5 (sliding window W < N, ZC)** — receiver observes only W∈`WINDOW_SIZES` coded bits per start hypothesis: the frame's last W bits = tail polar subcode (`PolarLink.tail_decoders(W)`), scored by `sync.tail_window_sync` (SC metric all offsets → SCL -PM on top-8). W-windows alias at W/2 steps (every half-block is a valid tail codeword), so `sync.resolve_block_ambiguity` decodes the ≤(N-W)/(W/2)+1 aliases as full frames (CRC first). ZC: `window_stream(preamble=)`, `zc_detect`. Operating points: P_miss@P_false=1e-3, P_false@P_miss=0 (gamma=min H1), BLER@P_miss=0. Own SNR sweep `WINDOW_SNR_DB` (0–10 dB). Eb = energy per *information* bit (CRC + preamble = overhead).
- **`main.py`** — CLI runner (`--quick`, `--parts`, `--snr`, `--seed`, `--out`, `--replot`). Spawn-based ProcessPoolExecutor, 1 torch thread/worker, batched tasks of `TRIALS_PER_TASK` trials. Burst task also decodes the true window (genie BLER) and splits each blind block error into sync-failure vs decoding; early-stops a point when the rarest error event reaches `--target-errors`. Writes `part1_bler.csv`, `polar_bler_nosync.png`, Part 3 extra op point P_false@P_miss=1e-2. `--burst slots|random`. Part 2 reuses Part 1 burst trials for K=128. `--sync-candidates N|all`. Writes `results/<run>/results.json` (checkpointed per point), plots, and `part3_metrics/*.npz`.
- **`tests/`** — pytest suite (`python -m pytest tests`, ~35 s): construction regression, encoder/QPSK/noise equivalence with legacy, wrapper==Sionna decisions, independent SC metric reference, legacy-vs-Sionna SCL agreement, reproducibility, end-to-end smoke.
- **`legacy/`** — pre-Sionna NumPy/numba code (`polar_numba.py`, `utils_numpy.py`, `sync.py`, `benchmark_decoder.py`), used only as a test oracle.

## Things to recheck when revisiting

- `config.py` K/SNR/iteration values if asked about runtime/results reproducibility — these are the knobs that drive experiment cost and precision.
- Whether `polar_*.png` outputs exist locally (generated artifacts, not checked in).
- Whether `decode_scl` and `decode_scl_numba` have diverged if either was edited since this map was written — re-run `benchmark_decoder.py`'s correctness check to confirm.

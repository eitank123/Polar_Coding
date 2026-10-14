# Polar_Coding: Implementation Reference (for correctness review)

This document describes everything the code does, the conventions it relies on, the
exact definitions of every reported quantity, and the claims a reviewer should verify.
File and function names are given as `file.py:function` so each statement can be
checked against the source. Section 9 is a checklist of concrete claims to audit.
Section 10 lists the results of the reference run and Section 11 interprets them.

---

## 1. Purpose

Monte-Carlo study of polar codes (N = 512) over QPSK/AWGN with **blind frame
synchronization** that uses the polar decoder's own soft metrics, compared with
classical Zadoff-Chu (ZC) preamble synchronization and with a genie (perfect-sync)
baseline. Five experiments ("Parts"):

| Part | Question | Main outputs |
|---|---|---|
| 1 | BER/BLER when the receiver must first find which of 5 slots holds the packet | `polar_ber.png`, `polar_bler.png`, `polar_bler_genie.png`*, `polar_bler_nosync.png`*, `part1_bler.csv` (*only if genie ≠ blind) |
| 2 | Probability of choosing the correct slot (SC metric vs CA-SCL metric) | `polar_sync_prob.png` |
| 3 | Frame-vs-noise hypothesis test on aligned windows (CA-SCL metric), rates 1/4, 1/3, 1/2 | `polar_hypo_pmiss.png`, `polar_hypo_pfalse.png`, `polar_hypo_pfalse_pmiss1e-3.png`, `polar_hypo_{pmiss,pfalse}_presets.png`, `part3_detection.csv`, `part3_metrics/*.npz` |
| 4 | Sliding-window sync: frame at an unknown symbol offset, every candidate window decoded (after an energy gate); P_miss, P_false, BLER incl. sync | `polar_window_{pmiss,pfalse,bler,complexity}.png`, `polar_window_{pfalse,bler}_pmiss1e-3.png`, `polar_window_{pmiss,pfalse}_presets.png`, `part4_window.csv`, `window_metrics/*.npz` |
| 5 | Blind sync (Part 4 receiver) vs ZC preambles of 31 / 61 symbols vs genie | `polar_zc_{pmiss,pfalse,bler}.png`, `polar_zc_{pfalse,bler}_pmiss1e-3.png`, `polar_zc_{pmiss,pfalse}_presets.png`, `part5_zc.csv` |

All outputs of a run go to one folder (`results/<run>/`), together with `results.json`
(every number, plus run metadata: seed, versions, config) and `progress.txt`.

---

## 2. Environment and how to run

* Conda env `~/env/polar-sionna` (Python 3.11): `sionna-no-rt==2.2.0` (PyTorch backend),
  `torch==2.14.1+cpu`, numpy 2.x, scipy, matplotlib, numba (legacy oracle only), pytest.
  See `requirements.txt`. The base `/opt/conda` Python has no torch/Sionna; run everything
  with `~/env/polar-sionna/bin/python` (or `conda activate polar-sionna`).
* Hardware assumed: CPU only; the worker count defaults to the cgroup CPU quota
  (`utils.py:available_cpus`), each worker runs single-threaded torch.
* Commands:
  * `python -m pytest tests` runs 45 tests, about 3 min.
  * `python main.py --quick` is a smoke run with 1000 burst trials and 4000/2000 H0/H1 trials.
  * `python main.py --seed 2026 --out results/final` is the full run, all parts.
  * `python main.py --parts 4 5 --snr 0 2 4` selects parts and SNR points.
  * `python main.py --replot results/final` regenerates plots and CSVs from `results.json`
    (and fills in operating points missing from older runs from the saved `.npz` statistics,
    see 5.6). It overwrites the PNG/CSV files in that folder; copy the folder first to keep
    the old plots.
  * Other flags: `--burst slots|random`, `--target-errors N|none`, `--sync-candidates N|all`,
    `--full-sweep`, `--detection-snr`, `--iterations`, `--h0-trials`, `--h1-trials`,
    `--workers`, `--trials-per-task`.

---

## 3. System model and conventions (single source of truth: `config.py`)

### 3.1 Code
* N = 512, n = 9. K counts **information + CRC** bits; `info_len = K - 6`.
  K ∈ {128, 256} (Parts 1, 4), {128, 171, 256} (Part 3), 128 (Parts 2, 5).
* CRC: Sionna `CRCEncoder("CRC6")` = 3GPP polynomial x^6 + x^5 + 1, appended after the
  info bits. (The pre-Sionna code used x^6 + x + 1; Sionna only supports 3GPP polynomials.)
* Encoder: Sionna `PolarEncoder(frozen_pos, N)`: x = u·G_N with G_N = F^{⊗n},
  F = [[1,0],[1,1]], **natural order (no bit-reversal permutation)**; info bits fill the
  non-frozen positions of u in increasing index order; frozen bits = 0.
* Frozen set: `utils.py:get_frozen_bits(N, K, design_snr_dB=0)`, Bhattacharyya bound
  recursion z → (2z − z², z²) starting from z = exp(−SNR_design), then **bit-reversed**
  so that the reliabilities are in the natural order of the encoder (see 8.1); the K
  smallest Z are information positions.
* Modulation: QPSK, Gray. Sionna `Mapper("qam", 2)`: bit 2i → I rail, bit 2i+1 → Q rail,
  bit 0 → +1/√2, bit 1 → −1/√2; Es = 1 per symbol. Verified equal to the legacy mapper.
* Channel: complex AWGN, Sionna `AWGN`: n ~ CN(0, N0), i.e. N0/2 per real dimension.
* Demapper: Sionna `Demapper("app", "qam", 2)`, exact APP (for QPSK this is per-rail):
  outputs **logits** ℓ = log p(b=1)/p(b=0) = −2√2·y/N0 per rail.

### 3.2 Energy definition (`polar.py:PolarLink.noise_var`, `window_stream`)
* Eb/N0 is **energy per information bit** (excludes CRC): N0 = 1 / (Eb/N0 · 2 · info_len/N)
  via Sionna `ebnodb2no(ebno, 2, info_len/N)`.
* With a preamble of L_p symbols (Part 5), the total transmitted energy is
  (L_p + N/2)·Es, so at equal Eb/N0 the noise is scaled by (L_p + N/2)/(N/2): the
  preamble is charged as overhead (ZC-31: 0.50 dB, ZC-61: 0.93 dB).
* Earlier versions used rate K/N (CRC counted as information); fixed (see 8.2).

### 3.3 LLR conventions
* Sionna blocks exchange logits log p1/p0. The decoders internally negate to classic
  LLRs λ = log p0/p1. **All sync/detection metrics in this repo are in the λ convention**
  (larger = more consistent with a valid codeword).

### 3.4 Decoders and metrics (`polar.py`)
* `PolarSCDecoderWithMetric` subclasses Sionna `PolarSCDecoder` (exact boxplus check-node
  update, LLRs clamped to ±30 at CN inputs). Override of `_polar_decode_sc` intercepts each
  frozen leaf and accumulates its decision LLR λ_i.
  **SC metric** = Σ_{i ∈ frozen} λ_i along the SC path. Larger means more consistent.
* `PolarSCLDecoderWithMetric` subclasses Sionna `PolarSCLDecoder` (fast-SCL with R0/REP/R1
  nodes, R1 uses the single-flip M=1 approximation; exact LLR-domain path metric
  PM += softplus(−(1−2û)λ)). Its `call` replicates the parent's path selection: add
  penalty `llr_max·k` to every path failing the CRC, take argmin; output the info bits of
  that path, the CRC status, and **CA-SCL metric = −PM(selected path)** *without* the
  penalty. Uses the private `_decode` method (Sionna 2.2.0 pinned); a test asserts
  decisions equal the stock decoder.
* L = 8.

### 3.5 Randomness and reproducibility
* Master seed (`--seed`, recorded in `results.json`) → `numpy.random.SeedSequence`; every
  worker task gets a spawned child seed and sets `sionna.phy.config.seed`, which seeds all
  Sionna/torch generators used in the task. Same seed ⇒ identical results (tested).

---

## 4. Module map

| File | Contents |
|---|---|
| `config.py` | All constants (Section 3, plus trial counts, SNR grid, burst/window settings). |
| `utils.py` | `get_frozen_bits`, `get_frozen_positions`, `bit_reversal_permutation`, `available_cpus`. |
| `polar.py` | Metric-returning decoders (3.4) and `PolarLink` (one per K): `frame`, `burst`, `window_stream`, `noise_only`, `noise_var`. |
| `sync.py` | `candidate_windows`, `blind_sync`, energy detector (`window_energy`, `energy_threshold`, `gated_sync`), ZC (`zc_sequence`, `zc_detect`). |
| `main.py` | Worker tasks (`_run_burst_task`, `_run_detection_task`, `_run_window_task`), `detection_summary`, `Runner` (Parts 1–5, early stopping, caching, progress bar), plotting/CSV, CLI. |
| `tests/` | `test_link.py` (link/decoders/legacy cross-checks), `test_window.py` (Parts 4/5 machinery, energy gate, ZC), `test_end_to_end.py` (whole pipeline, early stopping). |
| `legacy/` | Original NumPy/numba implementation, used only as a test oracle. |

### 4.1 Signal generators (`polar.py:PolarLink`)
* `frame(T, ebno)` gives T aligned frames and logits [T, N].
* `noise_only(T, ebno)` gives logits [T, N] of pure noise (Part 3 H0).
* `burst(T, max_start, ebno, step)` gives the packet starting at a uniform random bit in
  {0, step, …, max_start}, silence elsewhere. The stream has max_start + N coded bits.
  * Built per rail: the packet's per-bit amplitudes are scattered into a zero rail array,
    paired into complex symbols, sent through AWGN and demapped.
  * step = N with max_start = 4N gives the default "1 packet in 1 of 5 slots".
  * step = 1 gives bit-level offsets (odd starts put bit 0 on the Q rail).
* `window_stream(T, D, ebno, preamble=None, signal=True)` returns received **symbols**
  [T, D + L_p + N/2], N0 and the true symbol offset.
  * The block [preamble | QPSK data] starts at a uniform symbol offset 0..D, with
    silence elsewhere. D = 384 symbols (1.5 frames).
  * `signal=False` gives noise only (H0).

### 4.2 Sync primitives (`sync.py`)
* `candidate_windows(llr, N, step)` returns all length-N windows starting every `step` bits
  ([T, O, N] view).
* `blind_sync(link, llr, M, step)`:
  1. SC metric on every window. The SC estimate is the argmax (first max wins).
  2. CA-SCL on the M windows with the best SC metric. The CA-SCL estimate is the argmax of −PM.
  * M = None means every window. Returns estimates (in bits) and the decoded bits at each estimate.
* `window_energy(y, n_sym, D)` returns E(d) = Σ_{k<n_sym} |y[d+k]|², d = 0..D, computed by cumsum.
* **Energy detector (`gate_windows`).** A candidate window is decoded only if both conditions hold.
  1. **Absolute threshold**, γ_abs = min(γ_sig, γ_cfar), computed by `gate_threshold`.
     * γ_sig (`energy_threshold`) is the `ENERGY_GATE_MISS_PROB` (1e-6) quantile of the
       aligned-window energy of a frame that arrives `ENERGY_GATE_FADE_MARGIN_DB` (3 dB) below
       nominal energy. It uses 2E/N0 ~ noncentral χ²(df = 2n, nc = 2n·Es/N0).
     * γ_cfar (`noise_floor_threshold`) is the noise-floor threshold that a *noise-only stream*
       exceeds (in any window) with probability `ENERGY_GATE_NOISE_PASS_PROB` (1%). It uses only
       the noise model.
       * It is calibrated by Monte Carlo of max_d E(d)/N0. That ratio does not depend on N0, so it
         is computed once with a private seed and cached.
       * A Bonferroni bound over the 385 heavily overlapping windows was about 20× too conservative.
     * Taking the minimum keeps whichever is more permissive. At high SNR γ_cfar is far below the
       frame energy, so frames survive deep fades.
  2. **Relative peak rule**: E(d) ≥ E_max − κ·σ̂, with κ = `ENERGY_PEAK_KAPPA` = 4.
     * σ̂ = sqrt(2n(N0² + 2·N0·ês)) is the std of an energy difference between two windows.
     * ês = max(E_max/n − N0, 0) is estimated from the stream's own peak, so the rule adapts to
       the received energy instead of assuming the nominal one.
  * Measured behaviour: no true window was dropped for fades up to 3 dB at any SNR (tested).
    Empty streams pass at the design rate of 1% once the noise floor dominates (about 3 dB
    and up). Below that the signal term dominates and nearly every empty stream is decoded.
* `gated_sync(link, llr, passed, M, step)` is `blind_sync` restricted to the windows in `passed`:
  * SC runs only on passing windows; CA-SCL runs on the best M passing windows.
  * A stream with no passing window gets statistic −inf and nothing is decoded.
  * Also returns the decode counts per stream (complexity).
* `zc_sequence(L, u)` is z[n] = exp(−jπ·u·n(n+1)/L), with odd prime L and unit modulus.
* `zc_detect(y, z, D)` computes, for d = 0..D, the normalized correlation
  c(d) = |⟨z, y[d:d+L]⟩|² / (‖z‖²‖y[d:d+L]‖²) ∈ [0,1], and returns (max, argmax).

---

## 5. The experiments

### Part 1: BER/BLER with blind slot sync (`main.py:_run_burst_task`, `Runner.part1`)
* **Scenario (default `--burst slots`):**
  * Each trial is a burst of 5 slots of N bits, with the packet in a uniformly random slot.
  * The other 4 slots are silence plus noise (noise-only LLRs).
  * K ∈ {128, 256}, Eb/N0 ∈ {0..5} dB.
* **Receiver:**
  * SC metric on all 5 slots; the SC estimate is the argmax, and SC decodes that slot.
  * CA-SCL metric on all 5 slots (M = None); the CA-SCL estimate is the argmax, and its decision
    at that slot is the decoded frame.
  * The true slot is also decoded with SC and CA-SCL (**genie**).
* **Counted per trial and decoder:**
  * bit errors over the info_len information bits (CRC bits excluded);
  * block error = any information bit wrong;
  * sync_ok.
  * Every blind block error is attributed to exactly one cause:
    `blk_err_<dec>_sync_fail` (wrong slot decoded) or `blk_err_<dec>_decoding` (right slot, decoder
    failed). The identity blind = sync_fail + decoding is tested.
  * With correct sync, the blind decision *is* the genie decision on that trial.
* **Early stopping (`Runner.burst_point`):**
  * Tasks run in waves of 4 × workers × 200 trials.
  * A point stops when every reported error event has occurred at least
    `TARGET_ERRORS` = 1000 times. The events are blind SC/SCL block errors, genie SC/SCL
    block errors, and SC/SCL sync failures. Otherwise the point runs to `ITERATIONS` = 64000.
  * Stopping on counts makes the ratio estimator slightly biased, about 1/1000 relative.
* **Outputs:**
  * blind BER/BLER;
  * genie BER/BLER ("without sync errors");
  * the BLER split into sync-failure and decoding parts;
  * the trial counts.
* **`polar_bler_genie.png`** / **`polar_bler_nosync.png`** are only drawn when the genie BLER
  differs from the blind BLER (`main.py:_coincide`: within 5% or 3 error events at every
  point). In slot mode the two coincide (Section 11.1), so both plots are skipped (and stale
  copies removed on `--replot`); the genie numbers stay in `part1_bler.csv`. When drawn: one
  panel per K, blind BLER solid, genie BLER as large hollow markers, and a dotted line for the
  part of the blind BLER whose decoded slot was wrong. The same rule hides the
  "known frame start" curve on the Parts 4/5 BLER plots when it matches blind CA-SCL.

### Part 2: slot sync probability (`Runner.part2`)
* Reuses Part 1's trials for K = `PART2_K` = 128 (cached; same random draws).
* Reports P(correct slot) for SC and CA-SCL.
* `--burst random` switches Parts 1/2 to bit-level start offsets in 0..768, using the two-stage
  search with M = `SYNC_SCL_CANDIDATES` = 8. This is an option, not the default.

### Part 3: hypothesis test on aligned windows (`_run_detection_task`, `Runner.part3`)
* H0: `noise_only` (N0 of the same Eb/N0); H1: aligned frame (`frame`). Statistic: CA-SCL
  metric (−PM, CRC-selected path). K ∈ {128, 171, 256}; 96000 H0 and 16000 H1 trials per point (`NUM_H0_TRIALS`, `NUM_H1_TRIALS`).
* Detector: declare frame iff statistic ≥ γ.
  * Scenario A: γ = 0.999 quantile of H0 (`np.quantile(..., method="higher")`, an actual H0
    sample, same rule as Parts 4/5) → P_false ≈ 1e-3; report P_miss = P(H1 < γ).
  * Scenario B: γ = min(H1) → empirical P_miss = 0; report P_false = P(H0 ≥ γ).
  * Plus every preset of Section 5.6 (`main.py:operating_points`), including the second
    P_miss operating point P_miss ≤ 1e-3.
* Raw statistics saved in `part3_metrics/K{K}_snr{snr}.npz` (arrays `h0`, `h1`).
* Parts 1–2 sweep `SNR_DB` = 0..5 dB in 0.5 dB steps (`--snr`); Parts 3–5 sweep `DETECTION_SNR_DB` = 0..3 dB in 0.25 dB steps (`--detection-snr`).

### Part 4: sliding-window sync with energy gate (`_run_window_task` scheme "blind", `Runner.part4`)
* **Scenario:**
  * The frame starts at a uniform random symbol offset d ∈ {0..384} in a stream of 384 + 256
    symbols, with silence elsewhere.
  * Candidate starts are every symbol (385 candidates). Each candidate window has the full
    frame length N (256 symbols), so most windows contain only part of the frame.
  * H0 streams are noise only, with the same length and N0.
  * K ∈ {128, 256}; 96000 H0 and 16000 H1 streams per point.
* **Receiver (`gated_sync`):**
  1. Compute the energy of every candidate window.
  2. Decode only the windows selected by the energy detector (Section 4.2). If no window
     passes, the stream is declared empty with statistic −inf and nothing is decoded.
  3. Run SC on the passing windows. SC statistic = max SC metric; SC estimate = argmax.
  4. Run CA-SCL on the 8 passing windows with the best SC metric. CA-SCL statistic =
     max −PM; CA-SCL estimate = argmax.
  5. The decoded frame is the decoder's output at its own estimate: SC for the SC scheme,
     CA-SCL for the CA-SCL scheme.
  6. H1 streams are also decoded at the true offset (genie).
* **`main.py:detection_summary`.** A frame is declared iff the statistic is finite and ≥ γ.
  * Scenario A (P_false = 1e-3):
    * γ = the 0.999 quantile of H0 using `method="higher"`, an actual sample, so −inf values
      are handled exactly.
    * Reports P_miss, P(declared at the wrong start), and BLER = P(missed or wrong start or
      decoding failure).
    * Also reports the realized P_false.
  * Scenario B (P_miss = 0):
    * γ = min(H1).
    * Reports P_false = P(H0 ≥ γ), P(wrong start), and BLER = P(wrong start or decoding
      failure), since there are no misses.
    * If the gate rejected any H1 stream, P_miss = 0 is unreachable and P_false is reported as 1.
  * Scenario C (P_miss ≤ `cfg.PMISS_OPERATING_POINT` = 1e-3): threshold of Section 5.6;
    reports P_false, P(wrong start) and BLER = P(missed or wrong start or decoding failure).
  * Plus every preset of Section 5.6.
* **Complexity outputs:**
  * mean SC window decodes per stream under H0 and H1 (out of 385 without the gate);
  * the fraction of H0 streams that needed any decoding;
  * the empirical gate pass rate of the true window.
* Per-trial arrays are saved in `window_metrics/blind_K{K}_snr{snr}.npz`.

### Part 5: ZC preamble vs blind vs genie (`_run_window_task` scheme "zc31"/"zc61", `Runner.part5`)
* K = 128. The blind schemes are exactly Part 4's (results reused from the cache).
* ZC schemes:
  * The transmitter sends [ZC_L | data], with L ∈ {31, 61} and root 1.
  * The block starts at a uniform symbol offset 0..384.
  * N0 is scaled by (L + 256)/256 (energy overhead).
  * Detection statistic = max normalized correlation over offsets 0..384; estimate = argmax.
  * Data is demapped at est + L and decoded with CA-SCL.
  * The ZC schemes use **the same energy gate** (over the [preamble | data] block), so every
    scheme has the same pre-detector. A stream rejected by the gate has statistic −inf.
* **Comparison criteria (stated in the plots/CSV):**
  * Equal Eb/N0 with Eb = total transmitted energy (preamble + data) per information bit.
  * The same detector rule and operating points as Part 4.
  * The same offset uncertainty (385 candidates).
  * Overhead reported as air-time fraction L/(L + 256): 10.8% and 19.2%, and as energy:
    0.50 dB and 0.93 dB.
  * Baseline: genie CA-SCL at full energy.

### 5.6 Operating points, SNR sweep stop and plots (Parts 1–5)
* **Detector rule** (all of Parts 3–5): declare a frame iff the statistic is finite and ≥ γ.
* **P_false targets** (`cfg.PFA_TARGETS` = 1e-2, 1e-3, 1e-4): γ = the (1 − p) quantile of H0
  with `method="higher"` (`main.py:p_miss_at_pfa`); report P_miss. With 32000 H0 trials,
  p = 1e-4 leaves only about 3 H0 samples above γ, so that preset is noisy.
* **P_miss targets** (`cfg.PMISS_TARGETS` = 0, 1e-4, 1e-3, 1e-2): at most m = floor(p·n_H1)
  frames may be missed (0, 1, 16, 160 of 16000). `main.py:_gamma_at_pmiss`:
  1. γ_hi = the (m+1)-th smallest H1 statistic. It is the highest threshold that misses at
     most m frames, so it gives the lowest P_false.
  2. P_false only changes when γ crosses an H0 sample. γ is therefore lowered to the smallest
     H1 statistic above every H0 sample below γ_hi. P_false is unchanged and as few frames as
     possible are missed.
  * Without step 2, separable data (high SNR) would still miss exactly m frames, so the BLER
    at P_miss ≤ 1e-3 would be stuck at exactly 1e-3 and the sweep stop would never trigger.
    This was found in the reference run and fixed; the test
    `test_pmiss_target_threshold_misses_fewest_frames` guards it.
  * p = 0 gives γ = min(H1), i.e. Scenario B.
  * If more than m H1 streams were rejected by the energy gate (−inf), the target is
    unreachable and P_false = 1 is reported.
* **Second P_miss operating point** `cfg.PMISS_OPERATING_POINT` = 1e-3 also gets P(wrong start)
  and BLER in Parts 4/5 (`detection_summary`), and its own plots
  `polar_{hypo,window,zc}_pfalse_pmiss1e-3.png`, `polar_{window,zc}_bler_pmiss1e-3.png`. The
  P_miss = 0 plots are kept unchanged next to them.
* **Preset plots** `polar_{hypo,window,zc}_{pmiss,pfalse}_presets.png`: one panel per rate /
  decoder / scheme, one line per preset.
* **SNR sweep stop** (`Runner._all_zero`): each part stops its sweep at the first SNR where every
  reported error rate is 0. This is per K in Parts 1, 3, 4; over all schemes plus genie in
  Part 5; on sync failures in Part 2. Higher SNRs can only give more zeros, and in Parts 1/2 an
  all-zero point costs a full 64000 trials because early stopping never triggers.
  `cfg.STOP_SWEEP_AT_ZERO`; `--full-sweep` disables it. Each result block stores the SNRs
  it actually ran in its own `snr_db` list.
* **Plots** span the SNR points that were run, and the x axis of an error-rate plot ends at
  the last SNR with a nonzero (drawn) value; panels of one preset figure share that range.
  Zero points are left empty (a log axis cannot draw 0). No Monte-Carlo resolution line is
  drawn; the limits are 1/n_H0 for P_false and 1/n_H1 for P_miss and BLER.
* **`--replot` backfill** (`main.py:backfill_presets`): results.json files written before the
  presets existed are completed from `part3_metrics/*.npz` and `window_metrics/*.npz`
  (statistics plus `sync_ok`/`dec_ok`), filling only missing keys. The result is not written
  back to results.json.

### Progress bar (`main.py:Progress`)
* The planned cost is Σ trials × `TASK_COST[type]`, the relative worker cost per trial.
* Each completed task adds its cost; early-stopped points add their skipped cost.
* A line `[####----] xx.x% | elapsed h:mm:ss | ETA h:mm:ss` is printed every 15 s and written
  to `progress.txt`.
* ETA = elapsed · (1 − f)/f, so it self-calibrates.

---

## 6. Parallel execution
* `Runner` owns a spawn-context `ProcessPoolExecutor` (fork is unsafe with torch). Workers
  set `torch.set_num_threads(1)` and cache one `PolarLink` per K.
* Each task processes `TRIALS_PER_TASK` = 200 trials as one batch (Sionna blocks are batched);
  large window batches are chunked to ≤ 8192 windows per decoder call.
* Results are aggregated in submission order (`executor.map`), so results do not depend on
  scheduling.

---

## 7. Verification (tests)
* **Construction and legacy equivalence** (`test_link.py`):
  * frozen-set size and orientation, plus ≥ 90% overlap with the 3GPP 5G reliability ranking;
  * a BLER regression guarding the bit-reversal fix;
  * Sionna encoder output identical to the legacy encoder;
  * CRC6 detects single errors;
  * mapping and N0 identical to legacy (with the information-bit Eb), and Sionna logits =
    −legacy LLR/2.
* **Decoders** (`test_link.py`):
  * noiseless decoding is perfect;
  * the metric-SCL decisions and CRC status equal the stock Sionna SCL;
  * the metric-SC decisions equal stock Sionna SC, and the SC metric equals an independent
    NumPy SC implementation;
  * Sionna SCL agrees with the legacy numba SCL on ≥ 90% of frames, with similar error counts.
* **Burst** (`test_link.py`):
  * structure for random and slot starts;
  * the true window decodes for odd and even starts;
  * blind sync is exact at high SNR;
  * the exhaustive mode works;
  * the sync_fail + decoding = blind identity holds;
  * reproducibility.
* **Windows** (`test_window.py`):
  * ZC sequence properties;
  * preamble energy accounting;
  * ZC detection;
  * window energy equals a direct sum;
  * the energy-threshold miss rate matches its design (0.05 and 0.2, at 0 and 3 dB);
  * the noise-floor threshold pass rate matches its design (5%, at 0 and 6 dB);
  * the gate threshold is the minimum of its two terms, and a frame faded by 3 dB keeps its
    true window at 0/2/5/8 dB;
  * the gated search is identical to the ungated one when γ_E = −inf;
  * the gate rejects noise and keeps frames;
  * the window tasks are exact at 5 dB;
  * the detection-summary math, including −inf / gate rejections;
  * the P_miss-target threshold spends no allowed misses on separable data and only spends
    them where they lower P_false.
* **End to end** (`test_end_to_end.py`):
  * all parts run multi-process;
  * every plot and CSV exists, and CSV rows match their headers;
  * Part 2 reuses Part 1;
  * `--replot`;
  * early stopping obeys its rule.

---

## 8. Known issues fixed, deviations, approximations, limitations

### 8.1 Fixed bug: frozen set orientation (pre-existing)
* The original Bhattacharyya recursion produces reliabilities in **bit-reversed** order,
  while the encoder/decoder are natural-order.
* Effect: the code was built on the wrong frozen set, giving BLER ≈ 1 at 1–2 dB in both the
  original and the Sionna pipeline.
* Fixed by permuting Z with the bit-reversal permutation (`utils.py:get_frozen_bits`).
* Results produced before the fix are invalid.

### 8.2 Fixed: Eb counted CRC bits as information
* The rate used for N0 was K/N; it is now info_len/N. All curves shift right by
  10·log10(K/(K−6)): 0.21 dB for K=128, 0.10 dB for K=256, 0.15 dB for K=171.
* This also makes the energy accounting consistent with Part 5, where the preamble is
  charged as overhead.

### 8.3 Fixed: legacy QPSK LLR scale
* The legacy LLRs were 2× too large. That was harmless for min-sum decoding but changed the
  metric scale.
* Sionna's APP demapper is exact.

### 8.4 Deviations from the original code (intentional)
* CRC polynomial x^6+x^5+1 instead of x^6+x+1 (same length).
* Decoders use exact boxplus and an exact path metric (not min-sum / hard PM). SCL uses
  fast-SCL nodes with the R1 single-flip approximation.
* Part 2 reuses Part 1's trials instead of independent ones (same distribution, identical
  draws).
* Early stopping in Parts 1/2, and the SNR sweep stop in every part (5.6).
* In Parts 4/5 offsets are whole QPSK symbols, because ZC is a symbol sequence. Bit-level
  offsets exist for Parts 1/2 via `--burst random`.
* The Part 4/5 blind CA-SCL search re-scores the 8 best SC candidates instead of all
  windows.
  * Measured on the bit-level 0..2048 search, the exhaustive search improves CA-SCL sync at
    0–1 dB (for example 0.91 → 0.99 at 1 dB). So the CA-SCL blind results are pessimistic at
    low SNR. Use `--sync-candidates all` for the exhaustive search (about 30× slower).

### 8.5 Properties a reviewer should not mistake for bugs
* **−PM is not normalized.**
  * Signal energy enlarges |λ|, so a frame that fails to decode (or a partial window) can score
    *below* pure noise.
  * Consequences: non-monotone P_false at P_miss = 0, and the CA-SCL statistic being worse than
    the SC statistic for detection at low SNR.
  * This is a property of the metric as specified; possible remedies are listed in 8.6.
* **P_false at P_miss = 0 is an extreme-value statistic.**
  * It is set by the single worst H1 sample, so it is noisy and depends on the H1 sample size.
  * Empirical P_miss = 0 means P_miss < ~1/16000.
  * This is why P_miss ≤ 1e-3 (16 of 16000 frames) is reported as a second operating point
    (5.6, 11.3).
* **BLER at P_miss ≤ 1e-3 can exceed BLER at P_miss = 0** (e.g. ZC-31 at 4–5 dB). The allowed
  misses are spent when they lower P_false, and a missed frame is a block error.
* **Normalizing the CA-SCL metric was evaluated and rejected.**
  * Proposal: subtract the hard-decision path metric.
  * Exact form: ΔPM = PM(x̂) − Σ softplus(−|λ|) = Σ_{x̂_j ≠ sign-decision_j} |λ_j|.
  * Result on Part 3 samples: much worse. P_miss at P_false = 1e-3 went from 0.29 to 0.96
    (K=128, 0 dB) and from 0.012 to 0.99 (K=256, 1 dB).
  * Why: raw −PM = −(ΔPM + PM_hd), and PM_hd is large for noise and small with signal, so it
    carries most of the detection power.
  * A scale-free variant, ΔPM/Σ|λ|, was mixed: better in one case (K=128 at 2 dB, P_false at
    P_miss = 0: 0.018 vs 0.85) and worse elsewhere (K=256 at 0–1 dB).
  * So raw −PM is kept. Most of the non-monotone P_false at P_miss = 0 comes from the
    extreme-value statistic below, which every metric has.
* **Requiring a CRC pass in the detection metric was tested and rejected.**
  * It made detection much worse at low SNR (K=256 at 0 dB: P_miss 0.048 → 0.73), because the
    CRC passes for about 12% of noise windows and fails for many real frames.
* **Energy gate limits.**
  * Energy can say whether a frame is present but localizes it poorly. A window's energy
    changes by only |s|² = 1 per symbol of shift, against noise fluctuations of tens.
  * With a frame present, the relative peak rule removes about 25–60% of windows at mid/high SNR.
  * Its main saving is on empty streams: about 1% are decoded from about 3 dB up. At 0–2 dB the
    fade-margin signal term makes nearly every empty stream pass. That is the price of not
    missing frames that arrive weaker than nominal.
  * The gate's miss rate on the true window is measured and reported as `gate_true_pass`.
  * An earlier gate was tuned to exactly nominal AWGN frame energy. It dropped 50% of frames
    for a 2 dB fade at 5 dB, which is why it was replaced.
  * The simulator itself is AWGN only. Fade robustness is verified with attenuated frames
    (tests), not with a fading channel model.
* **Zero BLER points** mean fewer than one error in the trials run (BLER < 1/trials).
  They are drawn as gaps on log plots.

### 8.6 Not implemented (possible extensions)
* A normalized blind metric, e.g. PM minus the PM of the hard-decision path.
* A hybrid scheme: detect with energy/ZC/SC, locate and decode with CA-SCL.
* An Es/N0 axis for detection plots.
* A 5G reliability-sequence construction instead of Bhattacharyya.

---

## 9. Checklist for the reviewing agent
1. `utils.get_frozen_bits`: is the bit reversal correct for the natural-order G_N used by
   Sionna's `PolarEncoder`? (Expected: info set ≈ 5G ranking's; BLER ≈ 1e-2 at ~1 dB for
   K=128 CA-SCL.)
2. `PolarLink.noise_var`: N0 = 1/(Eb/N0 · 2 · info_len/N) with Es = 1. `window_stream`: N0
   scaled by (L_p + N/2)/(N/2).
3. `PolarSCDecoderWithMetric`: the frozen-leaf intercept sums λ in the log p0/p1 convention
   (Sionna's SC negates logits before recursion). Larger = better.
4. `PolarSCLDecoderWithMetric.call` matches Sionna 2.2 `PolarSCLDecoder.call` selection;
   metric = −PM without the CRC penalty.
5. `PolarLink.burst`: rails/scatter put bit j of the packet at stream bit start + j; complex
   pairing (2i, 2i+1) = (I, Q) matches the mapper.
6. `sync.blind_sync` / `gated_sync`: window indexing (step), argmax/topk semantics, estimates
   returned in bits; in `gated_sync`, unpassed windows can never be selected.
7. `sync.energy_threshold`: the noncentral χ² parameters (df = 2n, nc = 2n·Es/N0, scale N0/2).
   `sync.noise_floor_threshold`: the Monte-Carlo calibration of max E/N0. `sync.gate_windows`:
   the min of the two thresholds and the relative peak rule (σ̂, ês). The ZC path uses the same gate.
8. `sync.zc_detect`: normalization and offset range 0..D; Part 5 data slice at est + L.
9. `main._run_burst_task`: error attribution (sync_fail vs decoding), genie decoding at the
   true slot, BER over info bits only.
10. `main.detection_summary`: the quantile and zero-miss thresholds, treatment of −inf, and
    the BLER definitions at all three operating points. `main._gamma_at_pmiss`: γ keeps the
    minimal P_false and misses the fewest frames.
11. `Runner.burst_point`: the early-stopping rule (rarest event ≥ target) and that Part 2
    reuses Part 1 data.
12. Statistics: Part 3 percentile with 32000 H0 samples (32 above the threshold), zero-miss
    threshold from 16000 H1 samples.
13. Seeds: every task gets a distinct child seed (`SeedSequence.spawn`), so no two tasks
    share random draws.
14. `Runner._all_zero` / `_rates`: the sweep stops only when *every* reported error rate of the
    point is 0 (thresholds and counts excluded). Plots and CSVs index each block by its own
    `snr_db` list.

---

## 10. Results of the reference run

* **Run:** `python main.py --seed 2026 --out results/final`, started 2026-10-13, 8 workers,
  82.2 min, Sionna 2.2.0, torch 2.14.1+cpu. Parts 1/2: up to 64000 burst trials per point
  (early stopping at 1000 errors). Parts 3–5: 96000 H0 and 16000 H1 trials per point.
* This run used commit 37216d2, before the sweep stop existed, so every SNR point was
  simulated (0–5 dB for Parts 1/2, 0–10 dB for Parts 3–5).
* The presets and the P_miss ≤ 1e-3 operating point (5.6) were added afterwards. They come from
  the saved per-trial statistics via `--replot` and need no new simulation. `results/final` holds
  the raw run with its original plots; `results/final_replot` holds the same data with the current
  plots, CSVs and operating points.
* Notation: "0" means no event in the trials run (rate below 1/trials). Rows end at the first
  SNR where every column is 0; all later points of the 0–10 dB sweep are 0 as well.
  "Genie" = decoding the true window (perfect sync).

### 10.1 Part 1: BLER with blind slot sync (1 packet in 5 slots)

| K | Eb/N0 (dB) | Trials | SC blind | SC genie | SC wrong-slot errors | CA-SCL blind | CA-SCL genie | CA-SCL wrong-slot errors |
|---|---|---|---|---|---|---|---|---|
| 128 | 0 | 12800 | 8.1e-01 | 8.1e-01 | 1.9e-01 | 3.7e-01 | 3.7e-01 | 1.3e-01 |
| 128 | 1 | 64000 | 3.9e-01 | 3.9e-01 | 4.1e-02 | 4.6e-02 | 4.6e-02 | 1.3e-02 |
| 128 | 2 | 64000 | 7.1e-02 | 7.1e-02 | 4.6e-03 | 9.7e-04 | 9.7e-04 | 2.5e-04 |
| 128 | 3 | 64000 | 3.6e-03 | 3.6e-03 | 2.2e-04 | 0 | 0 | 0 |
| 128 | 4 | 64000 | 4.7e-05 | 4.7e-05 | 0 | 0 | 0 | 0 |
| 128 | 5 | 64000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 256 | 0 | 64000 | 9.8e-01 | 9.8e-01 | 1.3e-01 | 8.6e-01 | 8.6e-01 | 1.4e-03 |
| 256 | 1 | 64000 | 6.7e-01 | 6.7e-01 | 1.3e-02 | 2.6e-01 | 2.6e-01 | 1.7e-03 |
| 256 | 2 | 64000 | 1.5e-01 | 1.5e-01 | 1.6e-04 | 9.8e-03 | 9.8e-03 | 7.8e-05 |
| 256 | 3 | 64000 | 1.2e-02 | 1.2e-02 | 0 | 9.4e-05 | 9.4e-05 | 0 |
| 256 | 4 | 64000 | 3.9e-04 | 3.9e-04 | 0 | 0 | 0 | 0 |
| 256 | 5 | 64000 | 4.7e-05 | 4.7e-05 | 0 | 0 | 0 | 0 |

* "Wrong-slot errors" is the part of the blind BLER whose decoded slot was wrong
  (`bler_<dec>_sync_fail`). Blind and genie agree to 3–4 digits at every point; the largest gap
  is K=128 CA-SCL at 0 dB, 0.3720 vs 0.3698. Full precision is in `part1_bler.csv`.
* BER (blind), 0..5 dB:
  * K=128: SC 3.1e-1, 1.3e-1, 2.1e-2, 9.7e-4, 4.6e-6, 0; CA-SCL 1.5e-1, 1.6e-2, 3.2e-4, 0, 0, 0.
  * K=256: SC 3.9e-1, 1.9e-1, 2.8e-2, 1.6e-3, 2.9e-5, 1.5e-6; CA-SCL 3.1e-1, 6.6e-2, 1.7e-3,
    8.2e-6, 0, 0.

### 10.2 Part 2: probability of choosing the correct slot (K=128)

| Eb/N0 (dB) | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| SC metric | 0.8087 | 0.9592 | 0.9954 | 0.9998 | 1 | 1 |
| CA-SCL metric | 0.8686 | 0.9866 | 0.9998 | 1 | 1 | 1 |

### 10.3 Part 3: aligned-frame detection with the CA-SCL metric

| K (rate) | Eb/N0 (dB) | P_miss @ P_false=1e-2 | @1e-3 | @1e-4 | P_false @ P_miss=0 | @≤1e-4 | @≤1e-3 | @≤1e-2 | H0 metric mean ± std | H1 metric mean ± std | Separation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 (1/4) | 0 | 2.3e-01 | 3.0e-01 | 3.6e-01 | 1.0e+00 | 1.0e+00 | 1.0e+00 | 9.5e-01 | -274 ± 2.6 | -257 ± 12.0 | 1.4σ |
| 128 (1/4) | 1 | 2.5e-02 | 3.4e-02 | 4.5e-02 | 1.0e+00 | 1.0e+00 | 9.9e-01 | 4.1e-01 | -274 ± 2.9 | -236 ± 13.1 | 2.8σ |
| 128 (1/4) | 2 | 1.9e-04 | 4.4e-04 | 5.0e-04 | 7.0e-01 | 2.2e-01 | 0 | 0 | -276 ± 3.4 | -213 ± 12.5 | 4.8σ |
| 128 (1/4) | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | -280 ± 4.0 | -188 ± 12.7 | 6.9σ |
| 171 (1/3) | 0 | 1.9e-01 | 2.7e-01 | 3.5e-01 | 1.0e+00 | 1.0e+00 | 9.9e-01 | 8.3e-01 | -248 ± 2.9 | -229 ± 11.6 | 1.5σ |
| 171 (1/3) | 1 | 1.7e-02 | 2.5e-02 | 3.6e-02 | 1.0e+00 | 1.0e+00 | 9.4e-01 | 8.8e-02 | -246 ± 3.2 | -206 ± 13.4 | 2.9σ |
| 171 (1/3) | 2 | 6.3e-05 | 6.3e-05 | 6.3e-05 | 5.2e-01 | 9.4e-05 | 0 | 0 | -247 ± 3.6 | -180 ± 12.7 | 5.1σ |
| 171 (1/3) | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | -249 ± 4.2 | -153 ± 12.6 | 7.3σ |
| 256 (1/2) | 0 | 1.2e-02 | 4.9e-02 | 1.3e-01 | 2.9e-01 | 2.4e-01 | 9.3e-02 | 1.2e-02 | -196 ± 2.7 | -177 ± 7.5 | 2.4σ |
| 256 (1/2) | 1 | 5.9e-03 | 1.2e-02 | 2.7e-02 | 9.3e-01 | 7.6e-01 | 2.3e-01 | 1.8e-03 | -192 ± 2.6 | -157 ± 11.7 | 2.9σ |
| 256 (1/2) | 2 | 6.3e-05 | 6.3e-05 | 2.5e-04 | 3.5e-01 | 3.1e-04 | 0 | 0 | -189 ± 2.7 | -130 ± 12.1 | 4.7σ |
| 256 (1/2) | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | -187 ± 2.8 | -102 ± 11.6 | 7.1σ |

* Separation = (mean H1 − mean H0) / sqrt(std H0² + std H1²) of the CA-SCL metric (−PM).
  At 4 and 5 dB it keeps growing: 9.2σ / 11.9σ (K=128), 9.8σ / 13.0σ (K=171),
  10.1σ / 13.9σ (K=256).

### 10.4 Part 4: sliding-window sync with energy gate

| K | Decoder | Eb/N0 (dB) | P_miss @ P_false=1e-3 | P_false @ P_miss=0 | P_false @ P_miss≤1e-3 | BLER @ P_miss=0 | BLER @ P_miss≤1e-3 | Genie BLER | Noise-only streams decoded | SC decodes/stream (H0 / H1) |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | SC | 0 | 5.2e-01 | 9.5e-01 | 8.2e-01 | 8.1e-01 | 8.1e-01 | 8.1e-01 | 100.0% | 357.5 / 352.8 |
| 128 | SC | 1 | 1.0e-01 | 9.4e-01 | 5.2e-01 | 3.8e-01 | 3.8e-01 | 3.8e-01 | 99.5% | 287.6 / 318.7 |
| 128 | SC | 2 | 4.1e-03 | 2.2e-01 | 3.5e-02 | 7.0e-02 | 7.0e-02 | 7.0e-02 | 88.7% | 144.0 / 284.5 |
| 128 | SC | 3 | 6.3e-05 | 1.7e-03 | 0 | 4.7e-03 | 4.7e-03 | 4.7e-03 | 37.7% | 25.5 / 255.1 |
| 128 | SC | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 2.1% | 0.6 / 227.4 |
| 128 | CA-SCL | 0 | 5.2e-01 | 1.0e+00 | 1.0e+00 | 5.3e-01 | 5.3e-01 | 3.7e-01 | 100.0% | 357.5 / 352.8 |
| 128 | CA-SCL | 1 | 1.1e-01 | 9.9e-01 | 9.9e-01 | 1.1e-01 | 1.1e-01 | 4.6e-02 | 99.5% | 287.6 / 318.7 |
| 128 | CA-SCL | 2 | 6.0e-03 | 8.9e-01 | 8.6e-01 | 7.7e-03 | 7.7e-03 | 5.6e-04 | 88.7% | 144.0 / 284.5 |
| 128 | CA-SCL | 3 | 2.5e-04 | 3.8e-01 | 0 | 3.1e-04 | 3.1e-04 | 0 | 37.7% | 25.5 / 255.1 |
| 128 | CA-SCL | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 2.1% | 0.6 / 227.4 |
| 256 | SC | 0 | 1.9e-02 | 5.9e-02 | 1.4e-02 | 9.8e-01 | 9.8e-01 | 9.8e-01 | 30.4% | 18.6 / 251.5 |
| 256 | SC | 1 | 0 | 5.6e-04 | 6.3e-05 | 6.6e-01 | 6.6e-01 | 6.6e-01 | 1.2% | 0.3 / 225.3 |
| 256 | SC | 2 | 0 | 6.3e-05 | 0 | 1.6e-01 | 1.6e-01 | 1.6e-01 | 1.1% | 0.3 / 200.0 |
| 256 | SC | 3 | 0 | 0 | 0 | 1.2e-02 | 1.2e-02 | 1.2e-02 | 1.1% | 0.2 / 178.2 |
| 256 | SC | 4 | 0 | 0 | 0 | 5.6e-04 | 5.6e-04 | 5.6e-04 | 1.1% | 0.3 / 158.7 |
| 256 | SC | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 1.1% | 0.3 / 141.5 |
| 256 | CA-SCL | 0 | 1.5e-02 | 2.6e-02 | 5.1e-03 | 8.9e-01 | 8.9e-01 | 8.5e-01 | 30.4% | 18.6 / 251.5 |
| 256 | CA-SCL | 1 | 2.1e-03 | 6.5e-03 | 1.3e-03 | 2.9e-01 | 2.9e-01 | 2.6e-01 | 1.2% | 0.3 / 225.3 |
| 256 | CA-SCL | 2 | 3.8e-04 | 8.8e-03 | 0 | 1.1e-02 | 1.2e-02 | 1.1e-02 | 1.1% | 0.3 / 200.0 |
| 256 | CA-SCL | 3 | 0 | 0 | 0 | 6.3e-05 | 6.3e-05 | 6.3e-05 | 1.1% | 0.2 / 178.2 |
| 256 | CA-SCL | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1.1% | 0.3 / 158.7 |

* "Noise-only streams decoded" = fraction of H0 streams in which the energy gate passed at least
  one window. "SC decodes/stream" = mean number of windows decoded, out of 385 without the gate.
* The last two columns depend only on the energy gate, so they are the same for SC and CA-SCL.

### 10.5 Part 5: ZC preamble vs blind sync (K=128, equal energy per information bit)

| Scheme | Eb/N0 (dB) | P_miss @ P_false=1e-3 | P_false @ P_miss=0 | P_false @ P_miss≤1e-3 | BLER @ P_miss=0 | BLER @ P_miss≤1e-3 |
|---|---|---|---|---|---|---|
| Blind SC | 0 | 5.2e-01 | 9.5e-01 | 8.2e-01 | 8.1e-01 | 8.1e-01 |
| Blind SC | 1 | 1.0e-01 | 9.4e-01 | 5.2e-01 | 3.8e-01 | 3.8e-01 |
| Blind SC | 2 | 4.1e-03 | 2.2e-01 | 3.5e-02 | 7.0e-02 | 7.0e-02 |
| Blind SC | 3 | 6.3e-05 | 1.7e-03 | 0 | 4.7e-03 | 4.7e-03 |
| Blind SC | 4 | 0 | 0 | 0 | 0 | 0 |
| Blind CA-SCL | 0 | 5.2e-01 | 1.0e+00 | 1.0e+00 | 5.3e-01 | 5.3e-01 |
| Blind CA-SCL | 1 | 1.1e-01 | 9.9e-01 | 9.9e-01 | 1.1e-01 | 1.1e-01 |
| Blind CA-SCL | 2 | 6.0e-03 | 8.9e-01 | 8.6e-01 | 7.7e-03 | 7.7e-03 |
| Blind CA-SCL | 3 | 2.5e-04 | 3.8e-01 | 0 | 3.1e-04 | 3.1e-04 |
| Blind CA-SCL | 4 | 0 | 0 | 0 | 0 | 0 |
| ZC-31 | 0 | 6.6e-01 | 1.0e+00 | 9.6e-01 | 6.6e-01 | 6.6e-01 |
| ZC-31 | 1 | 4.3e-01 | 9.9e-01 | 8.5e-01 | 1.8e-01 | 1.8e-01 |
| ZC-31 | 2 | 2.1e-01 | 8.1e-01 | 5.6e-01 | 1.6e-02 | 1.6e-02 |
| ZC-31 | 3 | 4.6e-02 | 4.0e-01 | 9.2e-02 | 1.4e-03 | 2.0e-03 |
| ZC-31 | 4 | 6.3e-04 | 2.3e-02 | 7.5e-04 | 1.9e-04 | 1.1e-03 |
| ZC-31 | 5 | 0 | 1.3e-04 | 3.1e-05 | 0 | 6.9e-04 |
| ZC-31 | 6 | 0 | 0 | 0 | 0 | 0 |
| ZC-61 | 0 | 6.7e-02 | 9.9e-01 | 3.0e-01 | 8.1e-01 | 8.1e-01 |
| ZC-61 | 1 | 7.5e-03 | 4.7e-01 | 1.6e-02 | 3.3e-01 | 3.3e-01 |
| ZC-61 | 2 | 6.9e-04 | 1.1e-01 | 5.6e-04 | 3.5e-02 | 3.6e-02 |
| ZC-61 | 3 | 0 | 7.2e-04 | 3.1e-05 | 4.4e-04 | 6.3e-04 |
| ZC-61 | 4 | 0 | 0 | 0 | 0 | 0 |

* ZC overhead: ZC-31 = 10.8% of air time and 0.50 dB of energy; ZC-61 = 19.2% and 0.93 dB.
  The x-axis Eb counts the preamble energy.
* Genie CA-SCL BLER (full energy, no preamble): 3.7e-1, 4.6e-2, 5.6e-4, then 0 from 3 dB.
* The blind rows are Part 4's K=128 rows (same trials).

---

## 11. Analysis

### 11.1 Blind slot sync costs (almost) nothing in the 1-of-5 slot mode
* Blind BLER equals genie BLER at every point (10.1). The wrong slot only wins when the true
  slot's metric is poor, and that happens only when the true slot would not decode anyway.
  The "wrong-slot errors" are therefore (almost entirely) frames that genie decoding also
  fails, not extra errors.
* CA-SCL's −PM is the better slot selector (Part 2: 0.9998 vs 0.9954 at 2 dB). Since BLER is
  unchanged, this only reassigns which failed frames are labeled "wrong slot".
* This is a property of the easy slot scenario: 5 well-separated candidates, 4 of them pure
  noise. The sliding-window scenario (Part 4) has 385 overlapping candidates. There, blind
  CA-SCL BLER at P_miss = 0 is 7.7e-3 against a genie 5.6e-4 at 2 dB (K=128), so sync does cost.

### 11.2 Why the detection curves fall so fast
1. **The two hypotheses separate quickly.** In Part 3 the separation between the H0 and H1
   metric distributions grows by about 2.5σ per dB (1.4σ at 0 dB, 6.9σ at 3 dB, 11.9σ at
   5 dB for K=128). Tail probabilities of a separation that grows linearly in dB fall
   faster than exponentially, so on a log axis they look like a cliff.
2. **Missed detections are decoding failures.** In Part 4, almost every H1 stream below the
   P_false = 1e-3 threshold is a frame that also fails to decode or syncs to the wrong start:

   | K, decoder, Eb/N0 | H1 streams below threshold | of which wrong start or decoding failure |
   |---|---|---|
   | 128, SC, 1 dB | 1647 | 1646 |
   | 128, CA-SCL, 1 dB | 1708 | 1583 |
   | 128, SC, 2 dB | 66 | 66 |
   | 128, CA-SCL, 2 dB | 96 | 95 |
   | 128, CA-SCL, 3 dB | 4 | 4 |
   | 256, CA-SCL, 2 dB | 6 | 5 |

   So P_miss follows the BLER curve and falls exactly as steeply.
3. **P_miss = 0 behaves like a step function.** The threshold is the single worst of 16000
   H1 statistics. While even one frame fails to decode, that frame looks like noise and
   P_false ≈ 1. Once none fails, P_false drops to 0. Example: K=128 in Part 3, 0.70 at 2 dB
   and 0 at 3 dB.
4. **In Part 4 the energy gate does much of the work.** From about 1 dB (K=256) or 4–5 dB (K=128),
   only about 1% of noise-only streams reach the decoder (the CFAR design rate). P_false at those
   SNRs is mostly a property of the gate, not of the decoder metric. K=256 reaches this earlier
   because Eb is per information bit: at equal Eb/N0 its symbols carry 2× (3 dB) more energy.
5. **The simulation floor.** P_false cannot be measured below 1/32000 = 3.1e-5 and P_miss/BLER
   not below 1/16000 = 6.3e-5. Most curves cross these floors 1–2 dB after leaving 1, which is
   why everything is 0 from about 3–4 dB on.
6. **No extrapolation below the floor.** The H0 tails are heavier than Gaussian. The empirical
   z-score of the 1e-4 quantile is 3.9–4.6 against 3.72 for a Gaussian (K=128 at 0 dB: 2.57 / 3.72
   / 4.57 at 1e-2 / 1e-3 / 1e-4, vs 2.33 / 3.09 / 3.72 Gaussian). A Gaussian fit would understate
   P_false, so the curves are reported empirically only.

### 11.3 Which operating point to report
* **P_miss at fixed P_false (1e-2 / 1e-3 / 1e-4)** gives smooth, well-resolved curves that barely
  depend on the P_false preset (preset plots). P_miss at P_false = 1e-3 is the recommended
  headline detection curve.
* **P_false at P_miss = 0** is kept because it is the assignment's definition. It is decided by
  one frame out of 16000 (11.2, item 3) and is statistically weak.
* **P_false at P_miss ≤ 1e-3** (16 of 16000 frames may be missed) is the recommended companion
  curve. It is set by 16 frames, so it is stable, and it moves the drop about 1 dB earlier than
  P_miss = 0 (e.g. Part 3 K=128 at 2 dB: 0 instead of 0.70; Part 4 SC K=128 at 2 dB: 3.5e-2
  instead of 0.22).
* Every P_miss preset shows the same pattern. P_false stays near 1 while BLER is above the
  target, then collapses within about 1 dB once BLER drops below it. The preset only moves the
  step, it cannot make it smooth.
* Resolving P_false below 3e-5, or the high-SNR tail, needs more H0 trials (10× more → floor
  about 3e-6), not a different preset.

### 11.4 Decoder and code-rate observations
* **CA-SCL decodes better than SC but is not a better detector.** Part 4 K=128 at 2 dB: BLER at
  P_miss = 0 is 7.7e-3 (CA-SCL) vs 7.0e-2 (SC). But P_false at P_miss = 0 is 0.89 (CA-SCL) vs 0.22
  (SC), and at 3 dB 0.38 vs 1.7e-3. The unnormalized −PM lets a frame that fails to decode, or a
  partial window, score below pure noise (8.5). That costs little when a decision is forced
  (Part 1) but a lot when the threshold must admit every frame.
* **Rate 1/2 detects better than rate 1/4 at equal Eb/N0.** Part 3 at 0 dB: P_miss at
  P_false = 1e-3 is 4.9e-2 (K=256) vs 0.30 (K=128). At equal energy per information bit, K=256
  puts 2× the energy into each symbol, so the frame stands out more from noise, even though it
  decodes worse (BLER 0.86 vs 0.37 for CA-SCL in Part 1). Detection and decoding reward
  different things.
* **SC with K=256 in Part 4** has P_miss = 0 at P_false = 1e-3 from 1 dB while its BLER is still
  0.66. The SC metric plus the gate detect that a frame is present well before SC can decode it.

### 11.5 ZC preamble vs blind sync (Part 5)
* **ZC-61** detects best at low SNR. P_miss at P_false = 1e-3 is 6.7e-2 at 0 dB vs 0.52 for blind
  sync; P_false at P_miss ≤ 1e-3 is 1.6e-2 at 1 dB vs 0.52 / 0.99. Its 0.93 dB energy overhead
  costs BLER: 3.5e-2 at 2 dB vs 7.7e-3 for blind CA-SCL.
* **ZC-31** is worse than blind sync on every detection metric (P_miss 0.21 vs 6.0e-3 at 2 dB).
  31 symbols carry too little energy to detect reliably among 385 offsets.
* **Blind CA-SCL** has the best BLER at every SNR (7.7e-3 at 2 dB, 3.1e-4 at 3 dB), and no
  preamble overhead, but the worst P_false at P_miss = 0 (11.4).
* Trade-off: if missed/false detections dominate the requirements, a long preamble (ZC-61)
  wins at 0–2 dB. If BLER at equal energy matters most, blind CA-SCL wins, and SC is a good
  detector for the blind scheme. A hybrid (detect with SC or energy/ZC, decode with CA-SCL) is
  the natural next step (8.6).

### 11.6 Possible next steps
* More H0 trials (e.g. 320000) at 2–4 dB to resolve P_false below 3e-5 where curves now hit 0.
* A finer SNR grid (0.25–0.5 dB) around 1–3 dB, where all curves drop.
* The hybrid detector of 8.6, and the exhaustive CA-SCL search (`--sync-candidates all`),
  which 8.4 shows is pessimistic at 0–1 dB.

# Polar_Coding: Implementation Reference (for correctness review)

This document describes everything the code does, the conventions it relies on, the
exact definitions of every reported quantity, and the claims a reviewer should verify.
File and function names are given as `file.py:function` so each statement can be
checked against the source. Section 9 is a checklist of concrete claims to audit.

---

## 1. Purpose

Monte-Carlo study of polar codes (N = 512) over QPSK/AWGN with **blind frame
synchronization** that uses the polar decoder's own soft metrics, compared with
classical Zadoff-Chu (ZC) preamble synchronization and with a genie (perfect-sync)
baseline. Five experiments ("Parts"):

| Part | Question | Main outputs |
|---|---|---|
| 1 | BER/BLER when the receiver must first find which of 5 slots holds the packet | `polar_ber.png`, `polar_bler.png`, `polar_bler_genie.png`, `polar_bler_nosync.png`, `part1_bler.csv` |
| 2 | Probability of choosing the correct slot (SC metric vs CA-SCL metric) | `polar_sync_prob.png` |
| 3 | Frame-vs-noise hypothesis test on aligned windows (CA-SCL metric), rates 1/4, 1/3, 1/2 | `polar_hypo_pmiss.png`, `polar_hypo_pfalse.png`, `part3_detection.csv`, `part3_metrics/*.npz` |
| 4 | Sliding-window sync: frame at an unknown symbol offset, every candidate window decoded (after an energy gate); P_miss, P_false, BLER incl. sync | `polar_window_{pmiss,pfalse,bler,complexity}.png`, `part4_window.csv`, `window_metrics/*.npz` |
| 5 | Blind sync (Part 4 receiver) vs ZC preambles of 31 / 61 symbols vs genie | `polar_zc_{pmiss,pfalse,bler}.png`, `part5_zc.csv` |

All outputs of a run go to one folder (`results/<run>/`), together with `results.json`
(every number, plus run metadata: seed, versions, config) and `progress.txt`.

---

## 2. Environment and how to run

* Conda env `~/env/polar-sionna` (Python 3.11): `sionna-no-rt==2.2.0` (PyTorch backend),
  `torch==2.14.1+cpu`, numpy 2.x, scipy, matplotlib, numba (legacy oracle only), pytest.
  See `requirements.txt`.
* Hardware assumed: CPU only; the worker count defaults to the cgroup CPU quota
  (`utils.py:available_cpus`), each worker runs single-threaded torch.
* Commands:
  * `python -m pytest tests` runs 44 tests, about 2 min.
  * `python main.py --quick` is a smoke run with 1000 burst trials and 4000/2000 H0/H1 trials.
  * `python main.py --seed 2026 --out results/final` is the full run, all parts.
  * `python main.py --parts 4 5 --snr 0 2 4` selects parts and SNR points.
  * `python main.py --replot results/final` regenerates plots and CSVs from `results.json`.
  * Other flags: `--burst slots|random`, `--target-errors N|none`, `--sync-candidates N|all`,
    `--iterations`, `--h0-trials`, `--h1-trials`, `--workers`, `--trials-per-task`.

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

### Part 2: slot sync probability (`Runner.part2`)
* Reuses Part 1's trials for K = `PART2_K` = 128 (cached; same random draws).
* Reports P(correct slot) for SC and CA-SCL.
* `--burst random` switches Parts 1/2 to bit-level start offsets in 0..768, using the two-stage
  search with M = `SYNC_SCL_CANDIDATES` = 8. This is an option, not the default.

### Part 3: hypothesis test on aligned windows (`_run_detection_task`, `Runner.part3`)
* H0: `noise_only` (N0 of the same Eb/N0); H1: aligned frame (`frame`). Statistic: CA-SCL
  metric (−PM, CRC-selected path). K ∈ {128, 171, 256}; 32000 H0 and 16000 H1 trials per point.
* Detector: declare frame iff statistic ≥ γ.
  * Scenario A: γ = 99.9th percentile of H0 (linear interpolation) → P_false ≈ 1e-3; report
    P_miss = P(H1 < γ).
  * Scenario B: γ = min(H1) → empirical P_miss = 0; report P_false = P(H0 ≥ γ).
* Raw statistics saved in `part3_metrics/K{K}_snr{snr}.npz` (arrays `h0`, `h1`).
* Scenario A uses the same quantile rule as Parts 4/5 (`np.quantile(..., method="higher")`).
* Parts 3–5 sweep `DETECTION_SNR_DB` = 0..10 dB (`--detection-snr`); Parts 1–2 sweep `SNR_DB` = 0..5 dB.

### Part 4: sliding-window sync with energy gate (`_run_window_task` scheme "blind", `Runner.part4`)
* **Scenario:**
  * The frame starts at a uniform random symbol offset d ∈ {0..384} in a stream of 384 + 256
    symbols, with silence elsewhere.
  * Candidate starts are every symbol (385 candidates). Each candidate window has the full
    frame length N (256 symbols), so most windows contain only part of the frame.
  * H0 streams are noise only, with the same length and N0.
  * K ∈ {128, 256}; 32000 H0 and 16000 H1 streams per point.
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
  * the detection-summary math, including −inf / gate rejections.
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
* Early stopping in Parts 1/2.
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
    the BLER definitions at both operating points.
11. `Runner.burst_point`: the early-stopping rule (rarest event ≥ target) and that Part 2
    reuses Part 1 data.
12. Statistics: Part 3 percentile with 32000 H0 samples (32 above the threshold), zero-miss
    threshold from 16000 H1 samples.
13. Seeds: every task gets a distinct child seed (`SeedSequence.spawn`), so no two tasks
    share random draws.

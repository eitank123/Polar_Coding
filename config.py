import numpy as np

# Polar Code Parameters
N = 512
K_LIST = [128, 256]              # Part 1 code dimensions (K includes the CRC bits)
K_RATES = [128, 171, 256]        # Part 3 rates: 1/4, ~1/3, 1/2 for N=512
PART2_K = 128                    # Part 2 (burst slot sync) code dimension
FROZEN_DESIGN_SNR_DB = 0.0       # Bhattacharyya construction design SNR

# SCL Decoder Parameters
L = 8
# 6-bit CRC. Sionna only ships the 3GPP polynomials; "CRC6" is x^6 + x^5 + 1
# (the pre-Sionna code used x^6 + x + 1 -- same length, same rate loss).
CRC_DEGREE = "CRC6"
CRC_LEN = 6

# Modulation Parameters
MODULATION = 'QPSK'
BITS_PER_SYMBOL = 2

# Simulation Parameters
SNR_DB = np.arange(0, 6, 1)      # Eb/N0 sweep in dB (Parts 1/2)
DETECTION_SNR_DB = np.arange(0, 11, 1)  # Eb/N0 sweep for the detection parts (3, 4, 5)
ITERATIONS = 64000               # Monte-Carlo trials per SNR point

# Burst Transmission Parameters
# Receive window = max start bit + N coded bits; the packet start is uniform over
#   "slots":  {0, N, ..., SLOTS_MAX_START_BIT} -> one packet in a random slot of 5 (default, fast)
#   "random": any bit 0..RANDOM_MAX_START_BIT  -> start within 1.5 packets, 769 candidate offsets
BURST_MODE = "slots"
SLOTS_MAX_START_BIT = 4 * N
RANDOM_MAX_START_BIT = 3 * N // 2


def max_start_bit(mode):
    return SLOTS_MAX_START_BIT if mode == "slots" else RANDOM_MAX_START_BIT
SYNC_SCL_CANDIDATES = 8          # random mode: best SC-scan offsets re-scored by CA-SCL (None = all, ~1 s/trial)

# Stop an SNR point (Parts 1/2) once every counted error event (SC/CA-SCL blind and
# genie block errors, SC/CA-SCL sync failures) has occurred this often; ITERATIONS
# stays the cap. None = always run ITERATIONS trials.
TARGET_ERRORS = 1000

NUM_H0_TRIALS = 32000            # Need >=1000 to resolve P_false = 1e-3 accurately
NUM_H1_TRIALS = 16000            # (Parts 3-5 use these H0/H1 counts per SNR point)

# Parts 4/5: sliding-window frame sync. The frame (preamble + data for ZC schemes)
# starts at a uniform random QPSK-symbol offset 0..WINDOW_MAX_OFFSET_SYM inside a
# noisy observation stream (silence elsewhere); every symbol offset is a candidate,
# so most length-N candidate windows contain only part of the frame.
WINDOW_MAX_OFFSET_SYM = RANDOM_MAX_START_BIT // BITS_PER_SYMBOL   # 384 symbols = 1.5 frames
# Energy detector (Parts 4/5): only windows above the threshold are decoded. The
# threshold is min(signal-based, noise-floor-based) so it does not rely on the
# received frame energy being exactly nominal (fading / mismatch):
ENERGY_GATE_MISS_PROB = 1e-6     # P(aligned frame window below the signal-based threshold) ...
ENERGY_GATE_FADE_MARGIN_DB = 3.0 # ... for a frame up to this many dB weaker than nominal
ENERGY_GATE_NOISE_PASS_PROB = 0.01  # noise-floor (CFAR) threshold: P(noise-only stream passes)
ENERGY_PEAK_KAPPA = 4.0          # also require E >= E_max - kappa*sigma (relative to the stream's peak)
PART4_K_LIST = [128, 256]
PART5_K = 128
ZC_LENGTHS = [31, 61]            # Zadoff-Chu preamble lengths (symbols, odd primes)
ZC_ROOT = 1

# Execution Parameters
TRIALS_PER_TASK = 200            # Trials per worker task; bounds per-worker memory (~300 MB)
PRECISION = "single"             # Sionna precision: "single" (float32) or "double"

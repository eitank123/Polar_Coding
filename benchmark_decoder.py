import time
import numpy as np

from config import *
from utils import (
    get_frozen_bits,
    append_crc,
    get_noise_sigma,
    qpsk_modulate,
    qpsk_demodulate_llr,
)
from polar import PolarCode


# ============================================================
# Configuration
# ============================================================

K = K_LIST[0]

frozen_bits = get_frozen_bits(
    N,
    K,
    design_snr_dB=0.0
)

polar = PolarCode(
    N,
    frozen_bits
)

crc_len = len(CRC_POLY) - 1
info_len = K - crc_len


# ============================================================
# Generate one realistic received frame
# ============================================================

np.random.seed(42)

info_bits = np.random.randint(
    0,
    2,
    info_len
)

u_encoded = append_crc(
    info_bits,
    CRC_POLY
)

x = polar.encode(
    u_encoded
)

s = qpsk_modulate(
    x
)

snr = SNR_DB[0]

sigma, sigma2 = get_noise_sigma(
    snr,
    K,
    N,
    bits_per_symbol=BITS_PER_SYMBOL
)

noise = (
    np.random.randn(len(s))
    + 1j * np.random.randn(len(s))
) * sigma

y = s + noise

llr = qpsk_demodulate_llr(
    y,
    sigma2
)


# ============================================================
# Numba warm-up
# ============================================================

print("Compiling Numba decoder...")

polar.decode_scl_numba(
    llr,
    L
)

print("Compilation complete.")


# ============================================================
# Benchmark
# ============================================================

NUM_RUNS = 100

start = time.perf_counter()

for _ in range(NUM_RUNS):

    polar.decode_scl_numba(
        llr,
        L
    )

elapsed = time.perf_counter() - start


# ============================================================
# Results
# ============================================================

print()
print("=" * 60)
print("NUMBA SCL DECODER BENCHMARK")
print("=" * 60)

print(f"N          : {N}")
print(f"K          : {K}")
print(f"L          : {L}")
print(f"SNR        : {snr} dB")
print(f"Runs       : {NUM_RUNS}")

print(f"Total time : {elapsed:.3f} s")

print(
    f"Per decode : "
    f"{elapsed / NUM_RUNS * 1000:.3f} ms"
)

print(
    f"Decodes/s  : "
    f"{NUM_RUNS / elapsed:.2f}"
)

print("=" * 60)


import time
import numpy as np

from config import *
from utils import (
    get_frozen_bits,
    append_crc,
    get_noise_sigma,
    qpsk_modulate,
    qpsk_demodulate_llr,
)
from polar import PolarCode


# ============================================================
# Configuration
# ============================================================

K = K_LIST[0]
NUM_TESTS = 1000
NUM_RUNS = 50


# ============================================================
# Create Polar Code
# ============================================================

frozen_bits = get_frozen_bits(
    N,
    K,
    design_snr_dB=0.0
)

polar = PolarCode(
    N,
    frozen_bits
)

crc_len = len(CRC_POLY) - 1
info_len = K - crc_len


# ============================================================
# Generate test frame
# ============================================================

np.random.seed(42)

info_bits = np.random.randint(
    0,
    2,
    info_len
)

u_encoded = append_crc(
    info_bits,
    CRC_POLY
)

x = polar.encode(u_encoded)

s = qpsk_modulate(x)

snr = SNR_DB[0]

sigma, sigma2 = get_noise_sigma(
    snr,
    K,
    N,
    bits_per_symbol=BITS_PER_SYMBOL
)

noise = (
    np.random.randn(len(s))
    + 1j * np.random.randn(len(s))
) * sigma

y = s + noise

llr = qpsk_demodulate_llr(
    y,
    sigma2
)


# ============================================================
# Numba compilation
# ============================================================

print("=" * 60)
print("POLAR SCL DECODER TEST + BENCHMARK")
print("=" * 60)

print()
print("Compiling Numba decoder...")

polar.decode_scl_numba(
    llr,
    L=L,
    crc_poly=CRC_POLY
)

print("Compilation complete.")


# ============================================================
# Correctness test
# ============================================================

print()
print("Running correctness test...")
print(f"Tests: {NUM_TESTS}")

mismatches = 0

np.random.seed(12345)

for test in range(NUM_TESTS):

    # --------------------------------------------------------
    # Generate random information bits
    # --------------------------------------------------------

    info_bits = np.random.randint(
        0,
        2,
        info_len
    )

    u_encoded = append_crc(
        info_bits,
        CRC_POLY
    )

    x = polar.encode(u_encoded)

    s = qpsk_modulate(x)

    # --------------------------------------------------------
    # Generate random noise
    # --------------------------------------------------------

    noise = (
        np.random.randn(len(s))
        + 1j * np.random.randn(len(s))
    ) * sigma

    y = s + noise

    test_llr = qpsk_demodulate_llr(
        y,
        sigma2
    )

    # --------------------------------------------------------
    # Original decoder
    # --------------------------------------------------------

    u_original = polar.decode_scl(
        test_llr,
        L=L,
        crc_poly=CRC_POLY
    )

    # --------------------------------------------------------
    # Numba decoder
    # --------------------------------------------------------

    u_numba = polar.decode_scl_numba(
        test_llr,
        L=L,
        crc_poly=CRC_POLY
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    if not np.array_equal(
        u_original,
        u_numba
    ):
        mismatches += 1

        # Print first mismatch for debugging
        if mismatches == 1:

            print()
            print("FIRST MISMATCH")
            print("-" * 60)

            print(
                "Original:",
                u_original
            )

            print(
                "Numba   :",
                u_numba
            )

            print(
                "Different positions:",
                np.where(
                    u_original != u_numba
                )[0]
            )


print()
print("-" * 60)
print(
    f"Decoder mismatches: "
    f"{mismatches}/{NUM_TESTS}"
)
print("-" * 60)


# ============================================================
# Stop if incorrect
# ============================================================

if mismatches != 0:

    print()
    print("ERROR: Numba decoder is NOT equivalent.")
    print("Benchmark will NOT be performed.")
    print()

    raise SystemExit(1)


print()
print("SUCCESS: Numba decoder matches original decoder.")


# ============================================================
# Benchmark
# ============================================================

print()
print("Running benchmark...")
print(f"Runs: {NUM_RUNS}")


# Use the same LLR repeatedly.
# This isolates decoder performance from signal generation.
start = time.perf_counter()

for _ in range(NUM_RUNS):

    polar.decode_scl(
        llr,
        L=L,
        crc_poly=CRC_POLY
    )

original_elapsed = (
    time.perf_counter() - start
)


# ------------------------------------------------------------
# Numba decoder
# ------------------------------------------------------------

start = time.perf_counter()

for _ in range(NUM_RUNS):

    polar.decode_scl_numba(
        llr,
        L=L,
        crc_poly=CRC_POLY
    )

numba_elapsed = (
    time.perf_counter() - start
)


# ============================================================
# Results
# ============================================================

original_ms = (
    original_elapsed
    / NUM_RUNS
    * 1000
)

numba_ms = (
    numba_elapsed
    / NUM_RUNS
    * 1000
)

speedup = (
    original_elapsed
    / numba_elapsed
)

print()
print("=" * 60)
print("POLAR SCL DECODER BENCHMARK RESULTS")
print("=" * 60)

print(f"N              : {N}")
print(f"K              : {K}")
print(f"L              : {L}")
print(f"SNR            : {snr} dB")
print(f"Runs           : {NUM_RUNS}")
print()

print("Original decoder:")
print(f"  Total time   : {original_elapsed:.3f} s")
print(f"  Per decode   : {original_ms:.3f} ms")
print(f"  Decodes/s    : {NUM_RUNS / original_elapsed:.2f}")

print()

print("Numba decoder:")
print(f"  Total time   : {numba_elapsed:.3f} s")
print(f"  Per decode   : {numba_ms:.3f} ms")
print(f"  Decodes/s    : {NUM_RUNS / numba_elapsed:.2f}")

print()

print(f"SPEEDUP        : {speedup:.2f}x")

print("=" * 60)
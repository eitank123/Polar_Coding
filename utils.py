import os
import numpy as np


def bit_reversal_permutation(N):
    n = int(np.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def get_frozen_bits(N, K, design_snr_dB=0.0):
    """Generates frozen bit set using Bhattacharyya bounds for AWGN.

    The returned mask is in natural (non-bit-reversed) order, matching the
    G = F^{(x)n} encoder / SC tree used by Sionna's polar blocks.
    """
    snr_linear = 10 ** (design_snr_dB / 10)
    z = np.exp(-snr_linear)
    n = int(np.log2(N))
    Z = np.zeros(N)
    Z[0] = z

    for i in range(1, n + 1):
        stride = 2**(i-1)
        for j in range(stride):
            z_prev = Z[j]
            Z[j] = 2 * z_prev - z_prev**2
            Z[j + stride] = z_prev**2

    # The recursion above polarizes the LSB first, i.e. it yields Z in
    # bit-reversed order; the natural-order encoder polarizes the MSB first.
    Z = Z[bit_reversal_permutation(N)]

    indices = np.argsort(Z)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[indices[:K]] = False
    return frozen_bits


def get_frozen_positions(N, K, design_snr_dB=0.0):
    """Frozen-bit indices (the format Sionna's polar blocks expect)."""
    return np.where(get_frozen_bits(N, K, design_snr_dB))[0].astype(int)


def available_cpus():
    """CPUs actually granted to this container (cgroup quota), not the host count."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))

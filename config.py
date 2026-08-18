import numpy as np

# Polar Code Parameters
N = 512
K_LIST = [128, 256]

# SCL Decoder Parameters
L = 8
# 6-bit CRC Polynomial: x^6 + x + 1
CRC_POLY = np.array([1, 0, 0, 0, 0, 1, 1], dtype=int)

# Modulation Parameters
MODULATION = 'QPSK'
BITS_PER_SYMBOL = 2

# Simulation Parameters
SNR_DB = np.arange(0, 11, 1)  # SNR region [-5, 10]
ITERATIONS = 480           # Monte-Carlo trials per SNR point

# Blind Frame Synchronization Parameters
MAX_SYMBOL_OFFSET = 10        # Maximum timing offset in QPSK symbols

NUM_H0_TRIALS = 16000      # Need >=1000 to resolve P_false = 1e-3 accurately
NUM_H1_TRIALS = 8000
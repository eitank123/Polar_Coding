"""Legacy NumPy helpers (pre-Sionna). Kept as a reference oracle for tests/."""
import numpy as np

# Legacy 6-bit CRC polynomial x^6 + x + 1
CRC_POLY = np.array([1, 0, 0, 0, 0, 1, 1], dtype=int)


def get_frozen_bits(N, K, design_snr_dB=0.0):
    """Generates frozen bit set using Bhattacharyya bounds for AWGN."""
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
            
    indices = np.argsort(Z)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[indices[:K]] = False
    return frozen_bits

def append_crc(data, crc_poly):
    """Appends CRC bits to data."""
    crc_len = len(crc_poly) - 1
    padded = np.pad(data, (0, crc_len), 'constant')
    for i in range(len(data)):
        if padded[i] == 1:
            padded[i:i+crc_len+1] ^= crc_poly
    crc = padded[-crc_len:]
    return np.concatenate((data, crc))

def check_crc(data_with_crc, crc_poly):
    """Verifies CRC consistency."""
    crc_len = len(crc_poly) - 1
    padded = np.copy(data_with_crc)
    for i in range(len(data_with_crc) - crc_len):
        if padded[i] == 1:
            padded[i:i+crc_len+1] ^= crc_poly
    return np.sum(padded[-crc_len:]) == 0

def qpsk_modulate(x):
    """Modulates length-N binary vector x into N/2 QPSK symbols."""
    x_in_phase = x[0::2]
    x_quad = x[1::2]
    s_i = (1 - 2 * x_in_phase) / np.sqrt(2)
    s_q = (1 - 2 * x_quad) / np.sqrt(2)
    return s_i + 1j * s_q

def qpsk_demodulate_llr(y, sigma2):
    """Computes LLRs for N bits from N/2 complex QPSK symbols y."""
    llr_i = 2 * np.real(y) * np.sqrt(2) / sigma2
    llr_q = 2 * np.imag(y) * np.sqrt(2) / sigma2
    llr = np.empty(2 * len(y), dtype=np.float64)
    llr[0::2] = llr_i
    llr[1::2] = llr_q
    return llr

def build_burst_stream(packet_symbols, num_noise_packets):
    """Embeds one real packet's QPSK symbols into a random slot of a burst period.

    The period has (num_noise_packets + 1) slots, each the length of packet_symbols;
    all slots besides the chosen one are left at zero (no transmitted signal).
    """
    period = num_noise_packets + 1
    slot_len = len(packet_symbols)
    true_slot = np.random.randint(0, period)
    stream = np.zeros(period * slot_len, dtype=complex)
    stream[true_slot * slot_len : (true_slot + 1) * slot_len] = packet_symbols
    return stream, true_slot


def get_noise_sigma(snr_dB, K, N, bits_per_symbol=2):
    """Calculates AWGN noise standard deviation for a given Eb/N0."""
    snr_linear = 10 ** (snr_dB / 10)
    rate = K / N
    eb_n0_linear = rate * bits_per_symbol * snr_linear
    sigma2 = 1 / (2 * eb_n0_linear)
    return np.sqrt(sigma2), sigma2
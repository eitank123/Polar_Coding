import numpy as np

def run_blind_synchronization(polar, received_llr_stream, num_noise_packets, L, crc_poly):
    """Evaluates each candidate burst slot to identify which one holds the real packet."""
    best_slot = 0
    best_metric = -np.inf

    for slot in range(num_noise_packets + 1):
        bit_offset = slot * polar.N
        llr_window = received_llr_stream[bit_offset : bit_offset + polar.N]

        _, sync_metric = polar.decode_scl_numba(llr_window, L=L, crc_poly=crc_poly, return_sync_metric=True)

        if sync_metric > best_metric:
            best_metric = sync_metric
            best_slot = slot

    return best_slot
import numpy as np

def run_blind_synchronization(polar, received_llrs, max_symbol_offset, bits_per_symbol, L, crc_poly):
    """Evaluates candidate offsets to identify frame start using decoder soft info."""
    best_offset = 0
    best_metric = -np.inf
    
    for sym_offset in range(max_symbol_offset + 1):
        bit_offset = sym_offset * bits_per_symbol
        llr_window = received_llrs[bit_offset : bit_offset + polar.N]
        
        _, sync_metric = polar.decode_scl_numba(llr_window, L=L, crc_poly=crc_poly, return_sync_metric=True)
        
        if sync_metric > best_metric:
            best_metric = sync_metric
            best_offset = sym_offset
            
    return best_offset
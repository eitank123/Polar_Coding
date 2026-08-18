import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import os
from config import *
from utils import get_frozen_bits, append_crc, qpsk_modulate, qpsk_demodulate_llr, get_noise_sigma
from polar import PolarCode
from sync import run_blind_synchronization

NUM_CPUS = 8  # Limit execution to exactly 8 workers

def _run_part1_trial(args):
    seed, polar, info_len, sigma, sigma2 = args
    np.random.seed(seed)
    
    info_bits = np.random.randint(0, 2, info_len)
    u_encoded = append_crc(info_bits, CRC_POLY)
    
    x = polar.encode(u_encoded)
    s = qpsk_modulate(x)
    
    noise = (np.random.randn(polar.N // 2) + 1j * np.random.randn(polar.N // 2)) * sigma
    y = s + noise
    
    llr = qpsk_demodulate_llr(y, sigma2)

    # SC Decoding
    u_hat_sc = polar.decode_sc(llr)
    b_err_sc = np.sum(info_bits != u_hat_sc[:info_len])

    # CA-SCL Decoding - Numba
    u_hat_scl = polar.decode_scl_numba(
        llr,
        L=L,
        crc_poly=CRC_POLY
    )

    b_err_scl = np.sum(
        info_bits != u_hat_scl[:info_len]
    )
    return b_err_sc, int(b_err_sc > 0), b_err_scl, int(b_err_scl > 0)


def _run_part2_trial(args):
    seed, polar, info_len, sigma, sigma2 = args
    np.random.seed(seed)
    
    info_bits = np.random.randint(0, 2, info_len)
    u_encoded = append_crc(info_bits, CRC_POLY)
    x = polar.encode(u_encoded)
    
    true_sym_offset = np.random.randint(0, MAX_SYMBOL_OFFSET + 1)
    true_bit_offset = true_sym_offset * BITS_PER_SYMBOL
    
    pre_bits = np.random.randint(0, 2, true_bit_offset)
    post_bits = np.random.randint(0, 2, (MAX_SYMBOL_OFFSET - true_sym_offset) * BITS_PER_SYMBOL)
    
    stream_bits = np.concatenate([pre_bits, x, post_bits])
    
    s = qpsk_modulate(stream_bits)
    noise = (np.random.randn(len(s)) + 1j * np.random.randn(len(s))) * sigma
    y = s + noise
    
    llr_stream = qpsk_demodulate_llr(y, sigma2)
    
    est_sc = run_blind_synchronization(polar, llr_stream, MAX_SYMBOL_OFFSET, BITS_PER_SYMBOL, L=1, crc_poly=None)
    est_scl = run_blind_synchronization(polar, llr_stream, MAX_SYMBOL_OFFSET, BITS_PER_SYMBOL, L=L, crc_poly=CRC_POLY)
    
    return int(est_sc == true_sym_offset), int(est_scl == true_sym_offset)

def _run_h0_trial(args):
    """Worker function for H0 (Noise Only) hypothesis."""
    seed, polar, sigma, sigma2 = args
    np.random.seed(seed)
    
    # Generate pure complex Gaussian noise (no signal)
    noise = (np.random.randn(polar.N // 2) + 1j * np.random.randn(polar.N // 2)) * sigma
    llr_stream = qpsk_demodulate_llr(noise, sigma2)
    
    # Decode and extract soft metric
    _, metric = polar.decode_scl_numba(llr_stream, L=L, crc_poly=CRC_POLY if L > 1 else None, return_sync_metric=True)
    return metric

def _run_h1_trial(args):
    """Worker function for H1 (Polar Frame + Noise) hypothesis."""
    seed, polar, info_len, sigma, sigma2 = args
    np.random.seed(seed)
    
    # Generate valid polar frame
    info_bits = np.random.randint(0, 2, info_len)
    u_encoded = append_crc(info_bits, CRC_POLY)
    x = polar.encode(u_encoded)
    s = qpsk_modulate(x)
    
    # Add noise
    noise = (np.random.randn(polar.N // 2) + 1j * np.random.randn(polar.N // 2)) * sigma
    y = s + noise
    llr_stream = qpsk_demodulate_llr(y, sigma2)
    
    # Decode and extract soft metric
    _, metric = polar.decode_scl_numba(llr_stream, L=L, crc_poly=CRC_POLY if L > 1 else None, return_sync_metric=True)
    return metric

def evaluate_part1_ber_bler():
    ber_sc = {K: [] for K in K_LIST}
    bler_sc = {K: [] for K in K_LIST}
    ber_scl = {K: [] for K in K_LIST}
    bler_scl = {K: [] for K in K_LIST}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for K in K_LIST:
            print(f"\n--- PART 1: BER/BLER Evaluation (N={N}, K={K}, Modulation={MODULATION}) ---")
            frozen_bits = get_frozen_bits(N, K, design_snr_dB=0.0)
            polar = PolarCode(N, frozen_bits)
            crc_len = len(CRC_POLY) - 1
            info_len = K - crc_len

            for snr in SNR_DB:
                sigma, sigma2 = get_noise_sigma(snr, K, N, bits_per_symbol=BITS_PER_SYMBOL)
                
                err_bits_sc, err_blocks_sc = 0, 0
                err_bits_scl, err_blocks_scl = 0, 0

                # Loop through batches of size NUM_CPUS (8 tasks per outer loop iteration)
                for batch_start in range(0, ITERATIONS, NUM_CPUS):
                    if (batch_start % (10 * NUM_CPUS) == 0):
                        print(f"ITERATION: {batch_start} / {ITERATIONS}")
                    batch_size = min(NUM_CPUS, ITERATIONS - batch_start)
                    base_seed = int.from_bytes(os.urandom(4), byteorder='little')
                    
                    tasks = [( (base_seed + i) % (2**32), polar, info_len, sigma, sigma2 ) for i in range(batch_size)]
                    results = list(executor.map(_run_part1_trial, tasks))

                    for res in results:
                        err_bits_sc += res[0]
                        err_blocks_sc += res[1]
                        err_bits_scl += res[2]
                        err_blocks_scl += res[3]

                total_bits = ITERATIONS * info_len
                total_blocks = ITERATIONS

                ber_sc[K].append(err_bits_sc / total_bits)
                bler_sc[K].append(err_blocks_sc / total_blocks)
                ber_scl[K].append(err_bits_scl / total_bits)
                bler_scl[K].append(err_blocks_scl / total_blocks)
                
                print(f"SNR: {snr:2d} dB | SC BLER: {bler_sc[K][-1]:.4e} | CA-SCL BLER: {bler_scl[K][-1]:.4e}")

    # Plot Part 1: BER
    plt.figure(figsize=(10, 6))
    for K in K_LIST:
        plt.semilogy(SNR_DB, ber_sc[K], '--o', label=f'SC BER (K={K})')
        plt.semilogy(SNR_DB, ber_scl[K], '-s', label=f'CA-SCL BER (K={K}, L={L})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.ylim([1e-7, 1])
    plt.legend()
    plt.title(f'Polar Code BER vs SNR (N={N}, QPSK)')
    plt.savefig('polar_ber.png', dpi=300)
    plt.close()

    # Plot Part 1: BLER
    plt.figure(figsize=(10, 6))
    for K in K_LIST:
        plt.semilogy(SNR_DB, bler_sc[K], '--o', label=f'SC BLER (K={K})')
        plt.semilogy(SNR_DB, bler_scl[K], '-s', label=f'CA-SCL BLER (K={K}, L={L})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('Block Error Rate (BLER)')
    plt.ylim([1e-5, 1])
    plt.legend()
    plt.title(f'Polar Code BLER vs SNR (N={N}, QPSK)')
    plt.savefig('polar_bler.png', dpi=300)
    plt.close()


def evaluate_part2_frame_sync():
    K = 128  
    print(f"\n--- PART 2: Blind Frame Sync Evaluation (N={N}, K={K}, Modulation={MODULATION}) ---")
    frozen_bits = get_frozen_bits(N, K, design_snr_dB=0.0)
    polar = PolarCode(N, frozen_bits)
    crc_len = len(CRC_POLY) - 1
    info_len = K - crc_len

    prob_sync_sc = []
    prob_sync_scl = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for snr in SNR_DB:
            sigma, sigma2 = get_noise_sigma(snr, K, N, bits_per_symbol=BITS_PER_SYMBOL)
            correct_sc, correct_scl = 0, 0

            # Loop through batches of size NUM_CPUS (8 tasks per outer loop iteration)
            for batch_start in range(0, ITERATIONS, NUM_CPUS):
                if (batch_start % (5 * NUM_CPUS) == 0):
                        print(f"ITERATION: {batch_start} / {ITERATIONS}")
                batch_size = min(NUM_CPUS, ITERATIONS - batch_start)
                base_seed = int.from_bytes(os.urandom(4), byteorder='little')
                
                tasks = [( (base_seed + i) % (2**32), polar, info_len, sigma, sigma2 ) for i in range(batch_size)]
                results = list(executor.map(_run_part2_trial, tasks))

                for res in results:
                    correct_sc += res[0]
                    correct_scl += res[1]
                    
            prob_sync_sc.append(correct_sc / ITERATIONS)
            prob_sync_scl.append(correct_scl / ITERATIONS)
            print(f"SNR: {snr:2d} dB | SC Sync Prob: {prob_sync_sc[-1]:.4f} | SCL Sync Prob: {prob_sync_scl[-1]:.4f}")

    # Plot Part 2: Frame Sync Probability
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_DB, prob_sync_sc, '--o', label=f'SC Frame Sync (K={K})')
    plt.plot(SNR_DB, prob_sync_scl, '-s', label=f'CA-SCL Frame Sync (K={K}, L={L})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('Probability of Correct Synchronization')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.title(f'Blind Frame Synchronization Probability vs SNR (N={N}, QPSK)')
    plt.savefig('polar_sync_prob.png', dpi=300)
    plt.close()

def evaluate_part3_hypothesis_test():
    """PART 3: Binary Hypothesis Testing (H0 vs H1)."""
    print(f"\n--- PART 3: Hypothesis Testing (N={N}) ---")
    
    K_RATES = [128, 171, 256] # Corresponds to Rates: 1/4, ~1/3, 1/2 for N=512
    
    results_pmiss = {K: [] for K in K_RATES}
    results_pfalse = {K: [] for K in K_RATES}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUS) as executor:
        for K in K_RATES:
            rate_str = "1/4" if K == 128 else ("1/3" if K == 171 else "1/2")
            print(f"\nEvaluating Rate {rate_str} (K={K})")
            
            frozen_bits = get_frozen_bits(N, K, design_snr_dB=0.0)
            polar = PolarCode(N, frozen_bits)
            crc_len = len(CRC_POLY) - 1
            info_len = K - crc_len
            
            for snr in SNR_DB:
                sigma, sigma2 = get_noise_sigma(snr, K, N, bits_per_symbol=BITS_PER_SYMBOL)
                
                # --- Collect Metrics for H0 (Noise Only) ---
                metrics_h0 = []
                for batch_start in range(0, NUM_H0_TRIALS, NUM_CPUS):
                    if (batch_start % (100 * NUM_CPUS) == 0):
                        print(f"ITERATION: {batch_start} / {NUM_H0_TRIALS}")
                    batch_size = min(NUM_CPUS, NUM_H0_TRIALS - batch_start)
                    base_seed = int.from_bytes(os.urandom(4), byteorder='little')
                    tasks = [((base_seed + i) % (2**32), polar, sigma, sigma2) for i in range(batch_size)]
                    metrics_h0.extend(list(executor.map(_run_h0_trial, tasks)))
                
                # --- Collect Metrics for H1 (Polar Frame) ---
                metrics_h1 = []
                for batch_start in range(0, NUM_H1_TRIALS, NUM_CPUS):
                    if (batch_start % (100 * NUM_CPUS) == 0):
                        print(f"ITERATION: {batch_start} / {NUM_H1_TRIALS}")
                    batch_size = min(NUM_CPUS, NUM_H1_TRIALS - batch_start)
                    base_seed = int.from_bytes(os.urandom(4), byteorder='little')
                    tasks = [((base_seed + i) % (2**32), polar, info_len, sigma, sigma2) for i in range(batch_size)]
                    metrics_h1.extend(list(executor.map(_run_h1_trial, tasks)))
                    
                metrics_h0 = np.array(metrics_h0)
                metrics_h1 = np.array(metrics_h1)
                
                # --- Scenario A: Fixed P_false <= 10^-3 ---
                # Threshold gamma is set to the 99.9th percentile of H0 metrics.
                # Only 0.1% (10^-3) of noise samples will exceed this threshold.
                gamma_fixed_pfalse = np.percentile(metrics_h0, 99.9) 
                p_miss = np.sum(metrics_h1 <= gamma_fixed_pfalse) / NUM_H1_TRIALS
                results_pmiss[K].append(p_miss)
                
                # --- Scenario B: Zero P_miss ---
                # Threshold gamma is set slightly below the minimum observed H1 metric.
                # No H1 frames will fall below this threshold (P_miss = 0).
                gamma_zero_pmiss = np.min(metrics_h1) - 1e-6
                p_false = np.sum(metrics_h0 > gamma_zero_pmiss) / NUM_H0_TRIALS
                results_pfalse[K].append(p_false)
                
                print(f"SNR: {snr:2d} dB | P_miss (P_false=1e-3): {p_miss:.4e} | P_false (P_miss=0): {p_false:.4e}")

    # Plot 1: P_miss vs SNR (Fixed P_false)
    plt.figure(figsize=(10, 6))
    for K in K_RATES:
        rate_str = "1/4" if K == 128 else ("1/3" if K == 171 else "1/2")
        plt.semilogy(SNR_DB, results_pmiss[K], '-o', label=f'Rate {rate_str} (K={K})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('$P_{miss}$ (for fixed $P_{false} = 10^{-3}$)')
    plt.ylim([1e-7, 1.1])
    plt.legend()
    plt.title(f'Probability of Missed Detection vs SNR (N={N})')
    plt.savefig('polar_hypo_pmiss.png', dpi=300)
    plt.close()

    # Plot 2: P_false vs SNR (Zero P_miss)
    plt.figure(figsize=(10, 6))
    for K in K_RATES:
        rate_str = "1/4" if K == 128 else ("1/3" if K == 171 else "1/2")
        plt.semilogy(SNR_DB, results_pfalse[K], '-s', label=f'Rate {rate_str} (K={K})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel('E_b/N_0 (dB)')
    plt.ylabel('$P_{false}$ (for empirical $P_{miss} = 0$)')
    plt.ylim([1e-7, 1.1])
    plt.legend()
    plt.title(f'Probability of False Alarm vs SNR (Zero Miss Detection, N={N})')
    plt.savefig('polar_hypo_pfalse.png', dpi=300)
    plt.close()
    
if __name__ == "__main__":
    evaluate_part1_ber_bler()
    evaluate_part2_frame_sync()
    evaluate_part3_hypothesis_test()
import numpy as np
from numba import njit

def f_vec(a, b):
    a_c = np.clip(a, -100.0, 100.0)
    b_c = np.clip(b, -100.0, 100.0)
    return np.sign(a_c) * np.sign(b_c) * np.minimum(np.abs(a_c), np.abs(b_c))

def g_vec(a, b, c_left):
    a_c = np.clip(a, -100.0, 100.0)
    b_c = np.clip(b, -100.0, 100.0)
    return b_c + (1 - 2 * c_left) * a_c

@njit
def _f_numba(a, b):

    # Manual scalar clipping
    if a < -100.0:
        a_c = -100.0
    elif a > 100.0:
        a_c = 100.0
    else:
        a_c = a

    if b < -100.0:
        b_c = -100.0
    elif b > 100.0:
        b_c = 100.0
    else:
        b_c = b

    # sign(a)
    if a_c > 0.0:
        sa = 1.0
    elif a_c < 0.0:
        sa = -1.0
    else:
        sa = 0.0

    # sign(b)
    if b_c > 0.0:
        sb = 1.0
    elif b_c < 0.0:
        sb = -1.0
    else:
        sb = 0.0

    abs_a = abs(a_c)
    abs_b = abs(b_c)

    if abs_a < abs_b:
        minimum = abs_a
    else:
        minimum = abs_b

    return sa * sb * minimum


@njit
def _g_numba(a, b, c_left):

    if a < -100.0:
        a_c = -100.0
    elif a > 100.0:
        a_c = 100.0
    else:
        a_c = a

    if b < -100.0:
        b_c = -100.0
    elif b > 100.0:
        b_c = 100.0
    else:
        b_c = b

    return b_c + (1.0 - 2.0 * c_left) * a_c


@njit
def _update_c_numba(C, NODE_OFFSETS, NODE_SIZES, N, i, path, u_i):
    v = N + i

    off_v = NODE_OFFSETS[v]
    C[path, off_v] = u_i

    p = v >> 1

    while p >= 1:

        if v % 2 == 1:

            left_child = v - 1
            right_child = v

            off_p = NODE_OFFSETS[p]
            off_l = NODE_OFFSETS[left_child]
            off_r = NODE_OFFSETS[right_child]

            half = NODE_SIZES[right_child]

            for j in range(half):
                C[path, off_p + j] = (
                    C[path, off_l + j]
                    + C[path, off_r + j]
                ) % 2

            for j in range(half):
                C[path, off_p + half + j] = (
                    C[path, off_r + j]
                )

            v = p
            p = v >> 1

        else:
            break


@njit
def _decode_scl_numba_core(
    llr,
    L,
    N,
    n,
    frozen_bits,
    NODE_OFFSETS,
    NODE_SIZES
):
    mem_size = (n + 1) * N

    LLR = np.zeros(
        (L, mem_size),
        dtype=np.float64
    )

    C = np.zeros(
        (L, mem_size),
        dtype=np.int8
    )

    U_dec = np.zeros(
        (L, N),
        dtype=np.int8
    )

    PM = np.zeros(
        L,
        dtype=np.float64
    )

    Frozen_Sum = np.zeros(
        L,
        dtype=np.float64
    )

    LLR[0, :N] = llr

    num_active = 1

    for i in range(N):

        # ----------------------------------------------------
        # Exact equivalent of:
        #
        # d_start = 1 if i == 0 else \
        #     n - (i & -i).bit_length() + 1
        #
        # Numba does not support int.bit_length().
        # ----------------------------------------------------

        if i == 0:

            d_start = 1

        else:

            x = i & (-i)

            bit_length = 0

            while x > 0:
                x = x >> 1
                bit_length += 1

            d_start = n - bit_length + 1

        # ----------------------------------------------------
        # LLR tree traversal
        # ----------------------------------------------------

        for l in range(num_active):

            for d in range(d_start, n + 1):

                v = (N + i) >> (n - d)

                p = v >> 1

                off_p = NODE_OFFSETS[p]
                off_v = NODE_OFFSETS[v]

                half = NODE_SIZES[v]

                if v % 2 == 0:

                    for j in range(half):

                        a = LLR[l,off_p + j]

                        b = LLR[l,off_p + half + j]

                        LLR[l,off_v + j] = _f_numba(a, b)

                else:

                    off_left = NODE_OFFSETS[v - 1]

                    for j in range(half):

                        a = LLR[
                            l,
                            off_p + j
                        ]

                        b = LLR[
                            l,
                            off_p + half + j
                        ]

                        c = C[
                            l,
                            off_left + j
                        ]

                        LLR[
                            l,
                            off_v + j
                        ] = _g_numba(a, b, c)

        # ----------------------------------------------------
        # Leaf
        # ----------------------------------------------------

        leaf_v = N + i

        off_leaf = NODE_OFFSETS[leaf_v]

        # Maximum number of candidates = 2L
        max_candidates = 2 * L

        cand_parent = np.empty(
            max_candidates,
            dtype=np.int64
        )

        cand_bit = np.empty(
            max_candidates,
            dtype=np.int8
        )

        cand_pm = np.empty(
            max_candidates,
            dtype=np.float64
        )

        cand_f = np.empty(
            max_candidates,
            dtype=np.float64
        )

        num_candidates = 0

        # ----------------------------------------------------
        # Candidate generation
        # EXACTLY mirrors decode_scl()
        # ----------------------------------------------------

        if frozen_bits[i]:

            for l in range(num_active):

                lambda_val = LLR[
                    l,
                    off_leaf
                ]

                if lambda_val < 0.0:
                    pm_penalty = abs(lambda_val)
                else:
                    pm_penalty = 0.0

                cand_parent[num_candidates] = l
                cand_bit[num_candidates] = 0
                cand_pm[num_candidates] = (
                    PM[l] + pm_penalty
                )
                cand_f[num_candidates] = lambda_val

                num_candidates += 1

        else:

            for l in range(num_active):

                lambda_val = LLR[
                    l,
                    off_leaf
                ]

                if lambda_val < 0.0:

                    pm0 = PM[l] + abs(lambda_val)

                else:

                    pm0 = PM[l]

                if lambda_val >= 0.0:

                    pm1 = PM[l] + abs(lambda_val)

                else:

                    pm1 = PM[l]

                # Candidate 0
                cand_parent[num_candidates] = l
                cand_bit[num_candidates] = 0
                cand_pm[num_candidates] = pm0
                cand_f[num_candidates] = 0.0

                num_candidates += 1

                # Candidate 1
                cand_parent[num_candidates] = l
                cand_bit[num_candidates] = 1
                cand_pm[num_candidates] = pm1
                cand_f[num_candidates] = 0.0

                num_candidates += 1

        # ----------------------------------------------------
        # Stable insertion sort by PM
        #
        # Python:
        # candidates.sort(key=lambda x: x[2])
        #
        # Using < rather than <= preserves candidate order
        # when metrics are equal.
        # ----------------------------------------------------

        for a in range(1, num_candidates):

            parent_tmp = cand_parent[a]
            bit_tmp = cand_bit[a]
            pm_tmp = cand_pm[a]
            f_tmp = cand_f[a]

            b = a - 1

            while b >= 0 and cand_pm[b] > pm_tmp:

                cand_parent[b + 1] = cand_parent[b]
                cand_bit[b + 1] = cand_bit[b]
                cand_pm[b + 1] = cand_pm[b]
                cand_f[b + 1] = cand_f[b]

                b -= 1

            cand_parent[b + 1] = parent_tmp
            cand_bit[b + 1] = bit_tmp
            cand_pm[b + 1] = pm_tmp
            cand_f[b + 1] = f_tmp

        # ----------------------------------------------------
        # Select first L candidates
        # ----------------------------------------------------

        new_num_active = num_candidates

        if new_num_active > L:
            new_num_active = L

        LLR_next = np.empty_like(LLR)
        C_next = np.empty_like(C)
        U_next = np.empty_like(U_dec)

        PM_next = np.zeros(
            L,
            dtype=np.float64
        )

        Frozen_Next = np.zeros(
            L,
            dtype=np.float64
        )

        for idx in range(new_num_active):

            p_old = cand_parent[idx]
            u_i = cand_bit[idx]
            pm_val = cand_pm[idx]
            f_val = cand_f[idx]

            # Exact copies from original decoder
            LLR_next[idx, :] = LLR[p_old, :]
            C_next[idx, :] = C[p_old, :]
            U_next[idx, :] = U_dec[p_old, :]

            U_next[idx, i] = u_i

            PM_next[idx] = pm_val

            Frozen_Next[idx] = (
                Frozen_Sum[p_old] + f_val
            )

            _update_c_numba(
                C_next,
                NODE_OFFSETS,
                NODE_SIZES,
                N,
                i,
                idx,
                u_i
            )

        LLR = LLR_next
        C = C_next
        U_dec = U_next
        PM = PM_next
        Frozen_Sum = Frozen_Next

        num_active = new_num_active

    return (
        U_dec,
        PM,
        Frozen_Sum,
        num_active
    )

class PolarCode:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = frozen_bits

        # Pre-compute binary tree node indexing tables
        self.NODE_OFFSETS = np.zeros(2 * N, dtype=int)
        self.NODE_SIZES = np.zeros(2 * N, dtype=int)
        for v in range(1, 2 * N):
            d = v.bit_length() - 1
            j = v - (1 << d)
            size = N >> d
            self.NODE_SIZES[v] = size
            self.NODE_OFFSETS[v] = d * N + j * size

    def encode(self, u_info):
        """Polar Kronecker encoding."""
        u = np.zeros(self.N, dtype=int)
        u[~self.frozen_bits] = u_info
        x = np.copy(u)
        for d in range(self.n):
            stride = 2**d
            for i in range(0, self.N, 2*stride):
                for j in range(stride):
                    x[i+j] = (x[i+j] + x[i+j+stride]) % 2
        return x

    def decode_sc(self, llr, return_sync_metric=False):
        """Successive Cancellation decoder."""
        return self.decode_scl(llr, L=1, crc_poly=None, return_sync_metric=return_sync_metric)

    def _update_c(self, C, l, i, u_i):
        v = self.N + i
        off_v = self.NODE_OFFSETS[v]
        C[l, off_v] = u_i
        p = v >> 1
        while p >= 1:
            if v % 2 == 1: 
                left_child = v - 1
                right_child = v
                off_p = self.NODE_OFFSETS[p]
                off_l = self.NODE_OFFSETS[left_child]
                off_r = self.NODE_OFFSETS[right_child]
                half = self.NODE_SIZES[right_child]
                
                C[l, off_p : off_p + half] = (C[l, off_l : off_l + half] + C[l, off_r : off_r + half]) % 2
                C[l, off_p + half : off_p + 2 * half] = C[l, off_r : off_r + half]
                v = p
                p = v >> 1
            else:
                break

    def decode_scl(self, llr, L, crc_poly=None, return_sync_metric=False):
        """Vectorized SCL Decoder supporting metric extraction for frame sync."""
        mem_size = (self.n + 1) * self.N
        LLR = np.zeros((L, mem_size), dtype=np.float64)
        C = np.zeros((L, mem_size), dtype=np.int8)
        U_dec = np.zeros((L, self.N), dtype=np.int8)
        PM = np.zeros(L, dtype=np.float64)
        Frozen_Sum = np.zeros(L, dtype=np.float64)

        LLR[0, :self.N] = llr
        num_active = 1

        for i in range(self.N):
            d_start = 1 if i == 0 else self.n - (i & -i).bit_length() + 1

            for l in range(num_active):
                for d in range(d_start, self.n + 1):
                    v = (self.N + i) >> (self.n - d)
                    p = v >> 1
                    off_p = self.NODE_OFFSETS[p]
                    off_v = self.NODE_OFFSETS[v]
                    half = self.NODE_SIZES[v]

                    if v % 2 == 0:  
                        LLR[l, off_v : off_v + half] = f_vec(
                            LLR[l, off_p : off_p + half],
                            LLR[l, off_p + half : off_p + 2 * half]
                        )
                    else:  
                        off_left = self.NODE_OFFSETS[v - 1]
                        LLR[l, off_v : off_v + half] = g_vec(
                            LLR[l, off_p : off_p + half],
                            LLR[l, off_p + half : off_p + 2 * half],
                            C[l, off_left : off_left + half]
                        )

            leaf_v = self.N + i
            off_leaf = self.NODE_OFFSETS[leaf_v]

            candidates = []
            if self.frozen_bits[i]:
                for l in range(num_active):
                    lambda_val = LLR[l, off_leaf]
                    pm_penalty = abs(lambda_val) if lambda_val < 0 else 0.0
                    candidates.append((l, 0, PM[l] + pm_penalty, lambda_val))
            else:
                for l in range(num_active):
                    lambda_val = LLR[l, off_leaf]
                    pm0 = PM[l] + (abs(lambda_val) if lambda_val < 0 else 0.0)
                    pm1 = PM[l] + (abs(lambda_val) if lambda_val >= 0 else 0.0)
                    candidates.append((l, 0, pm0, 0.0))
                    candidates.append((l, 1, pm1, 0.0))

            candidates.sort(key=lambda x: x[2])
            selected = candidates[:L]
            num_active = len(selected)

            LLR_next = np.empty_like(LLR)
            C_next = np.empty_like(C)
            U_next = np.empty_like(U_dec)
            PM_next = np.zeros(L, dtype=np.float64)
            Frozen_Next = np.zeros(L, dtype=np.float64)

            for idx, (p_old, u_i, pm_val, f_val) in enumerate(selected):
                LLR_next[idx] = LLR[p_old]
                C_next[idx] = C[p_old]
                U_next[idx] = U_dec[p_old]
                U_next[idx, i] = u_i
                PM_next[idx] = pm_val
                Frozen_Next[idx] = Frozen_Sum[p_old] + f_val
                self._update_c(C_next, idx, i, u_i)

            LLR, C, U_dec, PM, Frozen_Sum = LLR_next, C_next, U_next, PM_next, Frozen_Next

        # Candidate path selection
        best_pm_idx = np.argmin(PM[:num_active])
        if crc_poly is not None:
            from .utils_numpy import check_crc
            best_pm_crc = float('inf')
            for l in range(num_active):
                u_cand = U_dec[l, ~self.frozen_bits]
                if check_crc(u_cand, crc_poly):
                    if PM[l] < best_pm_crc:
                        best_pm_crc = PM[l]
                        best_pm_idx = l
                        
        best_u = U_dec[best_pm_idx, ~self.frozen_bits]

        if return_sync_metric:
            if L == 1:
                # SC Frozen-bit metric: Sum of LLRs
                return best_u, Frozen_Sum[0]
            else:
                # SCL Path metric: Inverted so that max value corresponds to correct sync
                return best_u, -PM[best_pm_idx]
                
        return best_u
    
    def decode_scl_numba(
        self,
        llr,
        L,
        crc_poly=None,
        return_sync_metric=False
    ):
        """
        Numba-accelerated SCL decoder.

        Algorithmically matches decode_scl().
        """

        U_dec, PM, Frozen_Sum, num_active = _decode_scl_numba_core(
            llr,
            L,
            self.N,
            self.n,
            self.frozen_bits,
            self.NODE_OFFSETS,
            self.NODE_SIZES
        )

        # --------------------------------------------------------
        # Candidate path selection
        # --------------------------------------------------------

        best_pm_idx = 0

        for l in range(1, num_active):

            if PM[l] < PM[best_pm_idx]:
                best_pm_idx = l

        # --------------------------------------------------------
        # CRC-aided selection
        #
        # CRC checking remains in Python because your existing
        # check_crc() is already implemented there.
        # --------------------------------------------------------

        if crc_poly is not None:

            from .utils_numpy import check_crc

            best_pm_crc = float("inf")

            for l in range(num_active):

                u_cand = U_dec[
                    l,
                    ~self.frozen_bits
                ]

                if check_crc(u_cand, crc_poly):

                    if PM[l] < best_pm_crc:

                        best_pm_crc = PM[l]
                        best_pm_idx = l

        # --------------------------------------------------------
        # Extract information bits
        # --------------------------------------------------------

        best_u = U_dec[
            best_pm_idx,
            ~self.frozen_bits
        ]

        # --------------------------------------------------------
        # Sync metric
        #
        # EXACT same behavior as decode_scl()
        # --------------------------------------------------------

        if return_sync_metric:

            if L == 1:

                return (
                    best_u,
                    Frozen_Sum[0]
                )

            else:

                return (
                    best_u,
                    -PM[best_pm_idx]
                )

        return best_u
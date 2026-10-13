"""Polar-coded QPSK/AWGN link built from Sionna (PyTorch) blocks.

LLR conventions: Sionna passes *logits* log p(b=1)/p(b=0) between blocks. The
sync/detection metrics returned by the decoders below use the classic LLR
convention log p(b=0)/p(b=1), matching the original (pre-Sionna) definitions:

* SC  metric: sum of the frozen-leaf decision LLRs along the SC path
  (large and positive when the frozen bits look like zeros, i.e. a valid frame).
* SCL metric: -PM of the selected path (the CRC-passing path with the smallest
  path metric, or the smallest-PM path if none passes), so larger = better.
"""
import torch
from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN
from sionna.phy.fec.crc import CRCEncoder
from sionna.phy.fec.polar import PolarEncoder, PolarSCDecoder, PolarSCLDecoder
from sionna.phy.mapping import BinarySource, Demapper, Mapper
from sionna.phy.utils import ebnodb2no

from utils import get_frozen_positions


class PolarSCDecoderWithMetric(PolarSCDecoder):
    """Sionna SC decoder that additionally returns the frozen-bit sync metric."""

    def _polar_decode_sc(self, llr_ch, frozen_ind):
        # Sionna's recursion calls back into this override at every tree node,
        # so the frozen leaves see exactly the LLRs the SC decision is based on.
        if frozen_ind.shape[0] == 1 and bool(frozen_ind[0] == 1):
            self._frozen_llr_sum += llr_ch[..., 0]
        return super()._polar_decode_sc(llr_ch, frozen_ind)

    def call(self, llr_ch):
        batch_shape = llr_ch.shape[:-1]
        self._frozen_llr_sum = torch.zeros(
            batch_shape.numel(), dtype=self.dtype, device=llr_ch.device)
        u_hat = super().call(llr_ch)
        metric = self._frozen_llr_sum.reshape(batch_shape)
        del self._frozen_llr_sum
        return u_hat, metric


class PolarSCLDecoderWithMetric(PolarSCLDecoder):
    """Sionna CA-SCL decoder that additionally returns -PM of the selected path.

    Path selection replicates ``PolarSCLDecoder.call`` (Sionna 2.2); the
    ``test_scl_wrapper_matches_sionna`` test guards against upstream drift.
    """

    def call(self, llr_ch):
        if self._iil:
            raise NotImplementedError("CRC input-bit interleaving is not supported.")
        input_shape = llr_ch.shape
        llr_ch = llr_ch.reshape(-1, self._n)
        llr_ch = torch.clamp(-llr_ch, -self._llr_max, self._llr_max)

        bits, pm = self._decode(llr_ch)  # [B, 2L, n], [B, 2L]

        batch_idx = torch.arange(bits.shape[0], device=bits.device)
        if self._use_crc:
            u_list = bits[:, :, self._info_pos_t].to(llr_ch.dtype)
            _, crc_valid = self._crc_decoder(u_list)
            crc_valid = crc_valid.squeeze(-1)
            penalty = (1.0 - crc_valid.to(pm.dtype)) * self._llr_max * self._k
            cand = torch.argmin(pm + penalty, dim=-1)
            crc_ok = crc_valid[batch_idx, cand]
        else:
            cand = torch.argmin(pm, dim=-1)
            crc_ok = torch.ones_like(cand, dtype=torch.bool)

        u_hat = bits[batch_idx, cand][:, self._info_pos_t].to(llr_ch.dtype)
        metric = -pm[batch_idx, cand]

        batch_shape = list(input_shape[:-1])
        return (u_hat.reshape(batch_shape + [self._k]),
                metric.reshape(batch_shape),
                crc_ok.reshape(batch_shape))


class PolarLink:
    """CRC + polar encoder + QPSK + AWGN + SC / CA-SCL decoders for one (N, K)."""

    def __init__(self, N, K, list_size, crc_degree, bits_per_symbol=2,
                 design_snr_db=0.0, precision=None):
        self.N, self.K = N, K
        self.list_size, self.precision = list_size, precision
        self.bits_per_symbol = bits_per_symbol
        self.frozen_pos = get_frozen_positions(N, K, design_snr_db)

        self.source = BinarySource(precision=precision)
        self.crc_encoder = CRCEncoder(crc_degree, precision=precision)
        self.info_len = K - self.crc_encoder.crc_length
        self.encoder = PolarEncoder(self.frozen_pos, N, precision=precision)
        self.mapper = Mapper("qam", bits_per_symbol, precision=precision)
        self.channel = AWGN(precision=precision)
        self.demapper = Demapper("app", "qam", bits_per_symbol, precision=precision)
        self.sc = PolarSCDecoderWithMetric(self.frozen_pos, N, precision=precision)
        self.scl = PolarSCLDecoderWithMetric(
            self.frozen_pos, N, list_size=list_size, crc_degree=crc_degree,
            precision=precision)

    def noise_var(self, ebno_db):
        """Complex noise variance N0 for Eb/N0 [dB], Es = 1.

        Eb is the energy per *information* bit: the CRC bits are overhead, so the
        rate is info_len / N (not K / N). Preamble overhead is added in window_stream.
        """
        return ebnodb2no(float(ebno_db), self.bits_per_symbol, self.info_len / self.N)

    def frame(self, num_trials, ebno_db):
        """One packet per trial, perfectly aligned. Returns info bits and logits [T, N]."""
        no = self.noise_var(ebno_db)
        info_bits = self.source([num_trials, self.info_len])
        symbols = self.mapper(self.encoder(self.crc_encoder(info_bits)))
        return info_bits, self.demapper(self.channel(symbols, no), no)

    def window_stream(self, num_trials, max_offset_sym, ebno_db, preamble=None, signal=True):
        """Observation stream for sliding-window sync (Parts 4/5).

        The transmitted block [preamble | QPSK data] starts at a uniform random
        symbol offset in [0, max_offset_sym]; silence elsewhere. With signal=False
        the stream is noise only (H0). Eb counts *all* transmitted energy
        (preamble + data, Es = 1 per symbol), so a preamble raises N0 by
        (n_pre + n_data) / n_data at the same Eb/N0.

        Returns info bits [T, info_len] (None for H0), received symbols
        [T, max_offset_sym + n_pre + n_data], N0, and the true offset [T].
        """
        n_data = self.N // self.bits_per_symbol
        n_pre = 0 if preamble is None else preamble.shape[-1]
        no = self.noise_var(ebno_db) * (n_pre + n_data) / n_data
        offset = torch.randint(max_offset_sym + 1, (num_trials,), generator=sionna_config.torch_rng())
        stream = torch.zeros((num_trials, max_offset_sym + n_pre + n_data), dtype=self.mapper.cdtype)
        info_bits = None
        if signal:
            info_bits = self.source([num_trials, self.info_len])
            tx = self.mapper(self.encoder(self.crc_encoder(info_bits)))
            if preamble is not None:
                tx = torch.cat([preamble.to(tx.dtype).expand(num_trials, -1), tx], dim=1)
            idx = offset[:, None] + torch.arange(n_pre + n_data)
            stream.scatter_(1, idx, tx)
        return info_bits, self.channel(stream, no), no, offset

    def burst(self, num_trials, max_start, ebno_db, step=1):
        """One packet starting at a uniformly random bit in {0, step, 2*step, ..., max_start}.

        step=1: any bit. step=N: N-bit slots (with max_start=4N: one packet in a
        random slot of a 5-slot burst). Everything outside the packet is silence.
        The receive window is max_start + N coded bits long. QPSK is two
        independent BPSK rails (Gray, bit 2i -> I, bit 2i+1 -> Q), so an odd
        start simply puts the packet's first bit on the Q rail; empty bit
        positions carry zero amplitude (noise only).

        Returns info bits [T, info_len], channel logits [T, max_start + N]
        and the true start bit [T].
        """
        if (max_start + self.N) % self.bits_per_symbol:
            raise ValueError("max_start + N must be a multiple of bits_per_symbol.")
        if max_start % step:
            raise ValueError("max_start must be a multiple of step.")
        if self.bits_per_symbol != 2:
            raise NotImplementedError("Bit-granular offsets are implemented for QPSK only.")
        no = self.noise_var(ebno_db)
        info_bits = self.source([num_trials, self.info_len])
        symbols = self.mapper(self.encoder(self.crc_encoder(info_bits)))  # [T, N/2]
        amplitudes = torch.view_as_real(symbols).reshape(num_trials, self.N)  # per-bit rail amplitude

        start = step * torch.randint(max_start // step + 1, (num_trials,),
                                     generator=sionna_config.torch_rng(), device=symbols.device)
        rails = amplitudes.new_zeros((num_trials, max_start + self.N))
        idx = start[:, None] + torch.arange(self.N, device=symbols.device)
        rails.scatter_(1, idx, amplitudes)
        stream = torch.complex(rails[:, 0::2], rails[:, 1::2])

        llr = self.demapper(self.channel(stream, no), no)
        return info_bits, llr, start

    def noise_only(self, num_trials, ebno_db):
        """Channel logits [T, N] when nothing is transmitted (H0)."""
        no = self.noise_var(ebno_db)
        stream = torch.zeros((num_trials, self.N // self.bits_per_symbol),
                             dtype=self.mapper.cdtype)
        return self.demapper(self.channel(stream, no), no)

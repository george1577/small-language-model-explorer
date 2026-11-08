"""Simulated weight quantization (NF4 and int8), written out in plain PyTorch.

The example's model_quantization.py uses bitsandbytes, which needs an NVIDIA GPU. Here we do the
same arithmetic by hand: split weights into blocks, scale each block by its largest absolute value
(absmax), snap every weight to the nearest of the format's levels, then turn the codes back into
numbers ("de-quantize") so the model can run. That reproduces the *accuracy* effect exactly, but
the weights are stored back in full precision, so there is no real memory or speed saving at run
time. The memory numbers shown are calculated, not measured. See docs/adr/0002-simulated-quantization.md.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

# The 16 NF4 levels used by bitsandbytes (from the QLoRA paper). They are the quantiles of a normal
# distribution scaled to [-1, 1], so levels are packed tightly near 0, where most weights are.
NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
])
INT8_LEVELS = torch.arange(-127, 128, dtype=torch.float32) / 127   # 255 evenly spaced levels

BLOCK_SIZE = 64   # bitsandbytes' default block size for 4-bit weights
SCHEMES = ["original", "int8", "nf4"]


@dataclass
class Quantized:
    codes: torch.Tensor    # uint8: index of the chosen level for every weight (padded to whole blocks)
    absmax: torch.Tensor   # float32: one scale per block
    levels: torch.Tensor
    shape: torch.Size
    numel: int


def quantize(w: torch.Tensor, levels: torch.Tensor, block_size: int = BLOCK_SIZE) -> Quantized:
    flat = w.detach().float().flatten()
    pad = (-flat.numel()) % block_size
    blocks = torch.cat([flat, flat.new_zeros(pad)]).view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normalised = blocks / absmax[:, None]                 # every block now lies in [-1, 1]
    midpoints = (levels[1:] + levels[:-1]) / 2
    codes = torch.bucketize(normalised, midpoints)        # index of the nearest level
    return Quantized(codes.to(torch.uint8), absmax, levels, w.shape, flat.numel())


def dequantize(q: Quantized) -> torch.Tensor:
    values = q.levels[q.codes.long()] * q.absmax[:, None]
    return values.flatten()[: q.numel].view(q.shape)


def levels_for(scheme: str) -> torch.Tensor:
    return {"nf4": NF4_LEVELS, "int8": INT8_LEVELS}[scheme]


def fake_quantize(w: torch.Tensor, scheme: str, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Quantize then de-quantize: the weights the quantized model effectively computes with."""
    if scheme == "original":
        return w
    return dequantize(quantize(w, levels_for(scheme), block_size)).to(w.dtype)


# ---- which weights get quantized ----------------------------------------------------------------


def quantizable_linears(model: torch.nn.Module) -> dict[str, torch.nn.Linear]:
    """The Linear layers bitsandbytes would convert: every one except the output head (lm_head).

    Embeddings and layer norms also stay in 16-bit, as in bitsandbytes.
    """
    return {
        name: m for name, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and not name.endswith("lm_head")
    }


@dataclass
class QuantStats:
    scheme: str
    num_quantized: int          # weights that were quantized
    num_total: int              # all parameters in the model
    mean_abs_error: float       # average |original - de-quantized| over quantized weights
    mean_abs_weight: float      # average |original| over quantized weights, for scale


@contextmanager
def quantized_weights(model: torch.nn.Module, scheme: str):
    """Temporarily replace every quantizable weight with its quantized-and-de-quantized version."""
    linears = quantizable_linears(model)
    saved = {}
    err_sum = abs_sum = 0.0
    count = 0
    try:
        with torch.no_grad():
            for name, layer in linears.items():
                original = layer.weight.data
                new = fake_quantize(original, scheme)
                err_sum += float((original.float() - new.float()).abs().sum())
                abs_sum += float(original.float().abs().sum())
                count += original.numel()
                saved[name] = original
                layer.weight.data = new
        total = sum(p.numel() for p in model.parameters())
        yield QuantStats(scheme, count, total, err_sum / max(count, 1), abs_sum / max(count, 1))
    finally:
        for name, original in saved.items():
            linears[name].weight.data = original


# ---- memory arithmetic ----------------------------------------------------------------------------


def bits_per_weight(scheme: str, block_size: int = BLOCK_SIZE, double_quant: bool = False) -> float:
    """Storage bits per quantized weight, including the per-block scale (absmax).

    One float32 absmax per block costs 32 / block_size bits per weight (0.5 bits for blocks of 64).
    Double quantization (bitsandbytes' `bnb_4bit_use_double_quant`) stores the absmax values
    themselves in 8 bits, with one float32 scale per 256 of them: 8/64 + 32/(64*256) ≈ 0.127 bits.
    """
    base = {"original": 16.0, "int8": 8.0, "nf4": 4.0}[scheme]
    if scheme == "original":
        return base
    scale_bits = (8 / block_size + 32 / (block_size * 256)) if double_quant else 32 / block_size
    return base + scale_bits


@dataclass
class MemoryRow:
    label: str
    quantized_params: int
    other_params: int
    bits_quantized: float
    bits_other: float

    @property
    def bytes(self) -> float:
        return (self.quantized_params * self.bits_quantized + self.other_params * self.bits_other) / 8


def memory_table(model: torch.nn.Module) -> list[MemoryRow]:
    q = sum(m.weight.numel() for m in quantizable_linears(model).values())
    total = sum(p.numel() for p in model.parameters())
    other = total - q
    return [
        MemoryRow("fp32 (32-bit floats)", q, other, 32, 32),
        MemoryRow("fp16 / bf16 (16-bit floats)", q, other, 16, 16),
        MemoryRow("int8 (blocks of 64)", q, other, bits_per_weight("int8"), 16),
        MemoryRow("NF4 (blocks of 64)", q, other, bits_per_weight("nf4"), 16),
        MemoryRow("NF4 + double quantization", q, other, bits_per_weight("nf4", double_quant=True), 16),
    ]

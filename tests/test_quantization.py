import torch

from slm_lab.quantization import (
    INT8_LEVELS, NF4_LEVELS, bits_per_weight, dequantize, fake_quantize, quantize, quantized_weights,
)


def test_levels_are_round_tripped_exactly():
    w = NF4_LEVELS.repeat(4) * 0.37      # every block's absmax is 0.37
    assert torch.allclose(dequantize(quantize(w, NF4_LEVELS)), w, atol=1e-6)


def test_picks_nearest_level():
    w = torch.tensor([1.0, 0.03, 0.05, -0.65])
    q = quantize(w, NF4_LEVELS, block_size=4)
    assert NF4_LEVELS[q.codes.long()].flatten().tolist() == [1.0, 0.0, NF4_LEVELS[8].item(), NF4_LEVELS[1].item()]


def test_shape_and_padding_preserved():
    w = torch.randn(7, 13)
    assert fake_quantize(w, "nf4").shape == w.shape


def test_int8_is_more_accurate_than_nf4():
    torch.manual_seed(0)
    w = torch.randn(256, 256) * 0.02
    e8 = (fake_quantize(w, "int8") - w).abs().mean()
    e4 = (fake_quantize(w, "nf4") - w).abs().mean()
    assert e8 < e4 < w.abs().mean() * 0.2


def test_bits_per_weight():
    assert bits_per_weight("nf4") == 4.5
    assert bits_per_weight("int8") == 8.5
    assert abs(bits_per_weight("nf4", double_quant=True) - 4.127) < 0.001


def test_context_manager_restores_weights():
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8))
    before = [p.clone() for p in model.parameters()]
    with quantized_weights(model, "nf4") as stats:
        assert stats.num_quantized == 64 * 64 + 64 * 8
        assert not torch.equal(model[0].weight, before[0])
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), before))

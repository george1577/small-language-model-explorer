import torch

from slm_lab.distillation import (
    ToyConfig, combined_loss, distillation_loss, per_position_losses, run_toy_experiment, soften, task_loss,
)


def test_temperature_flattens():
    logits = torch.tensor([4.0, 1.0, 0.0])
    assert soften(logits, 4.0).max() < soften(logits, 1.0).max()


def test_distillation_loss_zero_when_student_matches_teacher():
    logits = torch.randn(5, 10)
    assert torch.allclose(distillation_loss(logits, logits, 2.0), torch.zeros(5), atol=1e-6)


def test_distillation_loss_matches_pytorch_kl_div():
    torch.manual_seed(0)
    s, t, T = torch.randn(3, 10), torch.randn(3, 10), 2.0
    ref = torch.nn.functional.kl_div(
        torch.log_softmax(s / T, -1), torch.softmax(t / T, -1), reduction="sum"
    ) * T**2
    assert torch.allclose(distillation_loss(s, t, T).sum(), ref, atol=1e-5)


def test_combined_loss_endpoints():
    a, b = torch.tensor([1.0]), torch.tensor([3.0])
    assert combined_loss(a, b, 0.0).item() == 1.0
    assert combined_loss(a, b, 1.0).item() == 3.0


def test_per_position_losses_predict_next_token():
    s, t = torch.randn(4, 6), torch.randn(4, 6)
    rows = per_position_losses(s, t, [0, 1, 2, 3], 2.0, 0.5)
    assert [r.label for r in rows] == [1, 2, 3]
    assert abs(rows[0].task - task_loss(s[:1], torch.tensor([1])).item()) < 1e-5


def test_toy_distilled_student_beats_hard_label_student():
    r = run_toy_experiment(ToyConfig(epochs=200))
    hard, distilled = list(r.heldout_kl)
    assert r.heldout_kl[distilled][-1] < r.heldout_kl[hard][-1] * 0.5

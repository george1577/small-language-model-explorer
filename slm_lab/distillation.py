"""Knowledge distillation losses, plus a tiny experiment that shows why soft labels help.

The loss is the same as `DistillationTrainer.compute_loss` in the example's knowledge_distillation.py:

    loss = (1 - alpha) * task_loss + alpha * distillation_loss
    task_loss         = cross-entropy(student, hard label)
    distillation_loss = T^2 * KL( softmax(teacher / T) || softmax(student / T) )

Here every loss is reported per token position so the app can show where the student and teacher
disagree. The example script averages differently (see the guide, section "Where the app differs").
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def soften(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax with temperature: divide logits by T first. T > 1 flattens, T < 1 sharpens."""
    return torch.softmax(logits / temperature, dim=-1)


def task_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the hard label at each position. Shape [n]."""
    return F.cross_entropy(student_logits, labels, reduction="none")


def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """T^2 * KL(teacher_T || student_T) at each position. Shape [n].

    Multiplying by T^2 keeps the size of the gradients roughly the same whatever T is, so changing T
    doesn't secretly change how much the distillation term counts.
    """
    log_student = torch.log_softmax(student_logits / temperature, dim=-1)
    log_teacher = torch.log_softmax(teacher_logits / temperature, dim=-1)
    kl = (log_teacher.exp() * (log_teacher - log_student)).sum(dim=-1)
    return kl * temperature**2


def combined_loss(task: torch.Tensor, distill: torch.Tensor, alpha: float) -> torch.Tensor:
    return (1 - alpha) * task + alpha * distill


# ---- inspecting real teacher and student logits on one text ---------------------------------


@dataclass
class PositionLoss:
    position: int            # index of the token being predicted
    label: int               # the hard label: the token actually in the text
    student_p_label: float   # student's probability of the hard label (T = 1)
    teacher_p_label: float   # teacher's probability of the hard label (T = 1)
    task: float
    distill: float
    combined: float


def per_position_losses(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, ids: list[int], temperature: float, alpha: float
) -> list[PositionLoss]:
    """Losses for predicting ids[1:] from logits rows 0..n-2."""
    s, t = student_logits[:-1], teacher_logits[:-1]
    labels = torch.tensor(ids[1:])
    task = task_loss(s, labels)
    dist = distillation_loss(s, t, temperature)
    comb = combined_loss(task, dist, alpha)
    sp = torch.softmax(s, dim=-1).gather(1, labels[:, None])[:, 0]
    tp = torch.softmax(t, dim=-1).gather(1, labels[:, None])[:, 0]
    return [
        PositionLoss(i + 1, int(labels[i]), float(sp[i]), float(tp[i]), float(task[i]), float(dist[i]), float(comb[i]))
        for i in range(len(labels))
    ]


# ---- the toy experiment ------------------------------------------------------------------------


@dataclass
class ToyConfig:
    n_train: int = 20         # training examples each student sees
    temperature: float = 2.0
    alpha: float = 0.5        # the distilled student's weight on distillation loss
    epochs: int = 300
    vocab: int = 20           # made-up tokens, in groups of similar tokens
    groups: int = 4
    dim: int = 8              # size of each example's feature vector (the "context")
    lr: float = 0.05
    seed: int = 0


@dataclass
class ToyResult:
    epochs: list[int]
    # for each student: name -> curve over epochs
    heldout_kl: dict[str, list[float]]       # KL(teacher || student) on unseen contexts, T = 1
    heldout_agree: dict[str, list[float]]    # share of unseen contexts where top-1 tokens match
    teacher_example: list[float]             # the teacher's distribution for one unseen context
    student_example: dict[str, list[float]]  # each student's distribution for the same context
    hard_labels_example: list[int]           # the hard labels sampled from the teacher (first few)


def _make_teacher(cfg: ToyConfig, gen: torch.Generator) -> torch.Tensor:
    """A linear 'teacher' whose similar tokens get similar scores (tokens come in groups)."""
    centres = torch.randn(cfg.groups, cfg.dim, generator=gen)
    group_of = torch.arange(cfg.vocab) % cfg.groups
    tokens = centres[group_of] + 0.35 * torch.randn(cfg.vocab, cfg.dim, generator=gen)
    return 1.6 * tokens   # [vocab, dim]; logits = context @ W.T


def run_toy_experiment(cfg: ToyConfig) -> ToyResult:
    """Train two tiny students from the same few examples: one on hard labels only, one distilled.

    The students have exactly the same shape as the teacher, so either could in principle match it
    perfectly; the only difference is how much each training example tells them.
    """
    gen = torch.Generator().manual_seed(cfg.seed)
    w_teacher = _make_teacher(cfg, gen)
    x_train = torch.randn(cfg.n_train, cfg.dim, generator=gen)
    x_test = torch.randn(500, cfg.dim, generator=gen)
    teacher_train_logits = x_train @ w_teacher.T
    hard = torch.multinomial(torch.softmax(teacher_train_logits, -1), 1, generator=gen)[:, 0]
    p_test = torch.softmax(x_test @ w_teacher.T, -1)

    students = {"Hard labels only (α = 0)": 0.0, f"Distilled (α = {cfg.alpha:g}, T = {cfg.temperature:g})": cfg.alpha}
    weights = {name: torch.zeros(cfg.vocab, cfg.dim, requires_grad=True) for name in students}
    opts = {name: torch.optim.Adam([w], lr=cfg.lr) for name, w in weights.items()}
    epochs, kl_curves, agree_curves = [], {n: [] for n in students}, {n: [] for n in students}

    for epoch in range(cfg.epochs + 1):
        if epoch % 5 == 0 or epoch == cfg.epochs:
            epochs.append(epoch)
            with torch.no_grad():
                for name, w in weights.items():
                    log_q = torch.log_softmax(x_test @ w.T, -1)
                    kl = (p_test * (p_test.log() - log_q)).sum(-1).mean()
                    agree = (log_q.argmax(-1) == p_test.argmax(-1)).float().mean()
                    kl_curves[name].append(float(kl))
                    agree_curves[name].append(float(agree))
        if epoch == cfg.epochs:
            break
        for name, alpha in students.items():
            logits = x_train @ weights[name].T
            loss = combined_loss(
                task_loss(logits, hard), distillation_loss(logits, teacher_train_logits, cfg.temperature), alpha
            ).mean()
            opts[name].zero_grad()
            loss.backward()
            opts[name].step()

    with torch.no_grad():
        example = {name: torch.softmax(x_test[0] @ w.T, -1).tolist() for name, w in weights.items()}
    return ToyResult(epochs, kl_curves, agree_curves, p_test[0].tolist(), example, hard[:10].tolist())

"""Glue between the real models and the pure logic in the other modules."""

import math
import time
from dataclasses import dataclass

import torch

from .lm import LanguageModel
from .speculative import DecodeResult, speculative_generate, target_only_generate


# ---- ① small vs large ----------------------------------------------------------------------------


@dataclass
class TimedGeneration:
    model: str
    prompt_tokens: int
    token_ids: list[int]
    text: str
    first_token_seconds: float   # reading the prompt + choosing the first token
    rest_seconds: float          # every token after the first

    @property
    def total_seconds(self) -> float:
        return self.first_token_seconds + self.rest_seconds

    @property
    def tokens_per_second(self) -> float:
        """Generation speed after the first token (the number usually quoted as throughput)."""
        n = len(self.token_ids) - 1
        return n / self.rest_seconds if n > 0 and self.rest_seconds > 0 else 0.0


def generate_timed(lm: LanguageModel, prompt_ids: list[int], max_new: int) -> TimedGeneration:
    """Greedy generation with a KV cache, timing the first token separately."""
    runner = lm.runner()
    out: list[int] = []
    start = time.perf_counter()
    first = 0.0
    while len(out) < max_new:
        token = int(runner.logits(prompt_ids + out)[-1].argmax())
        out.append(token)
        if len(out) == 1:
            first = time.perf_counter() - start
        if token in lm.eos_ids:
            break
    total = time.perf_counter() - start
    return TimedGeneration(lm.short_name, len(prompt_ids), out, lm.decode(out), first, total - first)


@dataclass
class TokenProb:
    token_id: int
    prob: float


def top_next_tokens(lm: LanguageModel, text: str, k: int = 10) -> list[TokenProb]:
    probs = torch.softmax(lm.sequence_logits(lm.encode(text))[-1], dim=-1)
    top = torch.topk(probs, k)
    return [TokenProb(int(i), float(p)) for p, i in zip(top.values, top.indices)]


# ---- ③ distillation ----------------------------------------------------------------------------


@dataclass
class DistillationInputs:
    ids: list[int]
    student_logits: torch.Tensor   # [n, vocab]
    teacher_logits: torch.Tensor   # [n, vocab]


def distillation_inputs(student: LanguageModel, teacher: LanguageModel, text: str, max_tokens: int) -> DistillationInputs:
    """Run the same text through both models (as the training loop does for every batch)."""
    ids = student.encode(text)[:max_tokens]
    s = student.sequence_logits(ids)
    t = teacher.sequence_logits(ids)
    vocab = min(s.shape[-1], t.shape[-1])   # same trick as the script for mismatched vocab sizes
    return DistillationInputs(ids, s[:, :vocab], t[:, :vocab])


# ---- ④ quantization ----------------------------------------------------------------------------


@dataclass
class Perplexity:
    value: float
    num_tokens: int
    seconds: float


def perplexity(lm: LanguageModel, texts: list[str], max_tokens: int) -> Perplexity:
    """exp(average negative log-probability of each actual next token) over all the texts."""
    nll, count = 0.0, 0
    start = time.perf_counter()
    for text in texts:
        ids = lm.encode(text)[:max_tokens]
        logits = lm.sequence_logits(ids)[:-1]
        nll += float(torch.nn.functional.cross_entropy(logits, torch.tensor(ids[1:]), reduction="sum"))
        count += len(ids) - 1
    return Perplexity(math.exp(nll / count), count, time.perf_counter() - start)


# ---- ⑤ speculative decoding --------------------------------------------------------------------


@dataclass
class SpeculativeComparison:
    prompt_ids: list[int]
    baseline: DecodeResult
    speculative: DecodeResult


def compare_speculative(
    draft: LanguageModel, target: LanguageModel, prompt: str, k: int, max_new: int
) -> SpeculativeComparison:
    prompt_ids = target.encode(prompt)
    eos = target.eos_ids | draft.eos_ids
    # warm-up so the first timed run doesn't pay one-off start-up costs
    target.runner().logits(prompt_ids)
    draft.runner().logits(prompt_ids)
    baseline = target_only_generate(target.runner().verify, prompt_ids, max_new, eos)
    spec = speculative_generate(draft.runner().propose, target.runner().verify, prompt_ids, k, max_new, eos)
    return SpeculativeComparison(prompt_ids, baseline, spec)

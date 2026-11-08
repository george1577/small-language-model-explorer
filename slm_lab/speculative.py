"""Speculative decoding, written out so every round can be inspected.

Works on two plain functions rather than real models, so it can be tested with fakes:

- propose(context, k) -> [(token, draft probability), ...]       (the draft model)
- verify(context, drafted) -> (target's greedy pick at each of the k+1 positions,
                               target's probability of each drafted token)   (the target model)

Verification here is greedy: a drafted token is accepted when it is exactly the token the target
model would have picked itself. That guarantees the output is identical to running the target model
alone with greedy decoding. (The sampling version of the rule is `sampling_acceptance_probability`.)
"""

import time
from dataclasses import dataclass, field
from typing import Callable

Propose = Callable[[list[int], int], list[tuple[int, float]]]
Verify = Callable[[list[int], list[int]], tuple[list[int], list[float]]]


@dataclass
class Round:
    drafted: list[int]            # the k tokens the draft model proposed
    draft_probs: list[float]      # the draft model's probability of each one (q)
    target_preds: list[int]       # the target's own greedy pick at each of the k+1 positions
    target_probs: list[float]     # the target's probability of each drafted token (p)
    num_accepted: int             # how many drafted tokens matched, counted from the left
    new_token: int                # the target's own token added at the end of the round
    draft_seconds: float
    target_seconds: float

    @property
    def added(self) -> list[int]:
        """Tokens this round contributed to the output: accepted drafts plus the target's token."""
        return self.drafted[: self.num_accepted] + [self.new_token]


@dataclass
class DecodeResult:
    token_ids: list[int]
    seconds: float
    target_calls: int
    rounds: list[Round] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds > 0 else 0.0

    @property
    def acceptance_rate(self) -> float:
        drafted = sum(len(r.drafted) for r in self.rounds)
        return sum(r.num_accepted for r in self.rounds) / drafted if drafted else 0.0


def count_accepted(drafted: list[int], target_preds: list[int]) -> int:
    """Greedy acceptance: accept drafted tokens from the left until the first mismatch."""
    n = 0
    for d, t in zip(drafted, target_preds):
        if d != t:
            break
        n += 1
    return n


def sampling_acceptance_probability(p_target: float, q_draft: float) -> float:
    """The sampling-mode rule: accept a drafted token with probability min(1, p/q).

    Not used by the app's greedy loop; shown in the app and the guide for comparison.
    """
    if q_draft <= 0:
        return 1.0
    return min(1.0, p_target / q_draft)


def _finish(tokens: list[int], max_new: int, eos_ids: set[int]) -> tuple[list[int], bool]:
    """Cut at max_new tokens or just after the first end-of-text token."""
    for i, t in enumerate(tokens[:max_new]):
        if t in eos_ids:
            return tokens[: i + 1], True
    return tokens[:max_new], len(tokens) >= max_new


def target_only_generate(verify: Verify, prompt_ids: list[int], max_new: int, eos_ids: set[int]) -> DecodeResult:
    """The baseline: the target model writes alone, one forward pass per token."""
    out: list[int] = []
    calls = 0
    start = time.perf_counter()
    while True:
        preds, _ = verify(prompt_ids + out, [])
        calls += 1
        out.append(preds[0])
        out, done = _finish(out, max_new, eos_ids)
        if done:
            break
    return DecodeResult(out, time.perf_counter() - start, calls)


def speculative_generate(
    propose: Propose, verify: Verify, prompt_ids: list[int], k: int, max_new: int, eos_ids: set[int]
) -> DecodeResult:
    """Draft k tokens, verify them in one target pass, keep the matching ones plus one target token."""
    out: list[int] = []
    rounds: list[Round] = []
    start = time.perf_counter()
    while True:
        context = prompt_ids + out
        t0 = time.perf_counter()
        drafted_pairs = propose(context, k)
        t1 = time.perf_counter()
        drafted = [t for t, _ in drafted_pairs]
        preds, p_drafted = verify(context, drafted)
        t2 = time.perf_counter()

        n = count_accepted(drafted, preds)
        r = Round(
            drafted=drafted,
            draft_probs=[q for _, q in drafted_pairs],
            target_preds=preds,
            target_probs=p_drafted,
            num_accepted=n,
            new_token=preds[n],   # the correction at the first mismatch, or a bonus token if all matched
            draft_seconds=t1 - t0,
            target_seconds=t2 - t1,
        )
        rounds.append(r)
        out.extend(r.added)
        out, done = _finish(out, max_new, eos_ids)
        if done:
            break
    return DecodeResult(out, time.perf_counter() - start, len(rounds), rounds)

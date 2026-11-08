from slm_lab.speculative import (
    Round, count_accepted, round_starts, sampling_acceptance_probability, speculative_generate,
    target_only_generate, verify_rows,
)

EOS = {99}


def target_next(context):
    """A fake target model: the next token is (sum of context) mod 7, plus 1."""
    return sum(context) % 7 + 1


def make_verify():
    def verify(context, drafted):
        preds, ctx = [], list(context)
        for t in drafted + [None]:
            preds.append(target_next(ctx))
            if t is not None:
                ctx.append(t)
        return preds, [1.0 if p == d else 0.0 for p, d in zip(preds, drafted)]
    return verify


def make_propose(wrong_every):
    """A fake draft model: agrees with the target except on every `wrong_every`-th position."""
    def propose(context, k):
        out, ctx = [], list(context)
        for _ in range(k):
            t = target_next(ctx)
            if len(ctx) % wrong_every == 0:
                t = 0
            out.append((t, 0.9))
            ctx.append(t)
        return out
    return propose


def test_count_accepted_stops_at_first_mismatch():
    assert count_accepted([1, 2, 3], [1, 2, 3, 4]) == 3
    assert count_accepted([1, 5, 3], [1, 2, 3, 4]) == 1
    assert count_accepted([5, 2, 3], [1, 2, 3, 4]) == 0


def test_speculative_output_equals_target_alone():
    baseline = target_only_generate(make_verify(), [3, 4], 30, EOS)
    for wrong_every in (1, 2, 3, 5, 1000):
        for k in (1, 3, 5):
            spec = speculative_generate(make_propose(wrong_every), make_verify(), [3, 4], k, 30, EOS)
            assert spec.token_ids == baseline.token_ids


def test_perfect_draft_accepts_everything_and_saves_target_calls():
    spec = speculative_generate(make_propose(10**9), make_verify(), [3, 4], 3, 20, EOS)
    assert spec.acceptance_rate == 1.0
    assert spec.target_calls == 5          # each round adds 3 accepted + 1 bonus token
    assert all(len(r.added) == 4 for r in spec.rounds)


def test_always_wrong_draft_still_makes_one_token_per_round():
    spec = speculative_generate(make_propose(1), make_verify(), [3, 4], 3, 10, EOS)
    assert spec.acceptance_rate == 0.0
    assert spec.target_calls == 10


def test_stops_at_end_of_text():
    verify = lambda ctx, d: ([99] * (len(d) + 1), [1.0] * len(d))
    propose = lambda ctx, k: [(99, 1.0)] * k
    assert speculative_generate(propose, verify, [1], 3, 50, EOS).token_ids == [99]
    assert target_only_generate(verify, [1], 50, EOS).token_ids == [99]


def test_sampling_acceptance_probability():
    assert sampling_acceptance_probability(0.6, 0.3) == 1.0
    assert sampling_acceptance_probability(0.2, 0.8) == 0.25


def _round(drafted, preds, n):
    return Round(drafted, [0.5] * len(drafted), preds, [0.5] * len(drafted), n, preds[n], 0.0, 0.0)


def test_verify_rows_after_a_rejection():
    rows = verify_rows(_round([10, 11, 12], [10, 20, 12, 13], 1))
    assert [r.status for r in rows] == ["accepted", "rejected", "discarded", "unused"]
    assert [r.kept for r in rows] == [True, True, False, False]
    assert [r.drafted_before for r in rows] == [[], [10], [10, 11], [10, 11, 12]]
    assert rows[-1].proposed is None and rows[-1].target_pick == 13


def test_verify_rows_all_accepted_uses_the_bonus():
    rows = verify_rows(_round([10, 11, 12], [10, 11, 12, 13], 3))
    assert [r.status for r in rows] == ["accepted", "accepted", "accepted", "bonus"]
    assert all(r.kept for r in rows)


def test_verify_rows_first_draft_rejected_keeps_only_the_correction():
    rows = verify_rows(_round([10, 11, 12], [20, 11, 12, 13], 0))
    assert [r.status for r in rows] == ["rejected", "discarded", "discarded", "unused"]
    assert [r.target_pick for r in rows if r.kept] == [20]


def test_round_starts_follow_tokens_added():
    spec = speculative_generate(make_propose(3), make_verify(), [3, 4], 3, 20, EOS)
    starts = round_starts(spec.rounds)
    assert starts[0] == 0
    for i in range(1, len(starts)):
        assert starts[i] == starts[i - 1] + len(spec.rounds[i - 1].added)

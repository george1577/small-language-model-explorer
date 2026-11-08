"""The only module that touches the real models. Everything else works on plain token ids and logits."""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small model: student, draft model, quantization subject
LARGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # large model: teacher, target model

# On a laptop CPU, bf16 matrix maths is extremely slow (seconds per token on an M2), and in fp32 the two
# models need ~8 GB, which makes an 8 GB Mac swap constantly. Both run in fp16 (~4 GB together).
# See docs/adr/0003-fp16-on-cpu.md.
DTYPES = {SMALL_MODEL: torch.float16, LARGE_MODEL: torch.float16}


def pick_device() -> str:
    # Apple's MPS backend is opt-in: on macOS 14 recent torch builds hang or crash on it.
    if os.environ.get("SLM_DEVICE"):
        return os.environ["SLM_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


class LanguageModel:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = pick_device()
        self.dtype = DTYPES.get(model_id, torch.float32)
        start = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=self.dtype).to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - start

        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])

    @property
    def short_name(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def num_params(self) -> int:
        # tied embeddings (input embedding == output head) are counted once
        return sum(p.numel() for p in self.model.parameters())

    @property
    def weight_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    # ---- text <-> ids ---------------------------------------------------------------

    def chat_prompt_ids(self, user_message: str) -> list[int]:
        """Token ids for a one-message chat, ready for the assistant's reply."""
        messages = [{"role": "user", "content": user_message}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.encode(text)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_label(self, token_id: int) -> str:
        """A token's text with whitespace made visible."""
        if token_id in self.eos_ids:
            return "⟨end of text⟩"
        text = self.tokenizer.decode([token_id])
        return text.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")

    # ---- model calls ----------------------------------------------------------------

    @torch.no_grad()
    def sequence_logits(self, ids: list[int]) -> torch.Tensor:
        """Logits at every position of `ids` (row i predicts token i+1), as float32 on the CPU."""
        out = self.model(input_ids=torch.tensor([ids], device=self.device))
        return out.logits[0].float().cpu()

    def runner(self) -> "CachedRunner":
        return CachedRunner(self)


@dataclass
class CachedRunner:
    """Feeds a growing token sequence to a model, reusing the KV cache.

    The KV cache stores the model's work for tokens it has already read. When the sequence changes
    (speculative decoding throws away rejected tokens), the cache is cut back to the longest prefix
    that is still valid and only the new tokens are fed in.
    """

    lm: LanguageModel
    cached_ids: list[int] = field(default_factory=list)
    past: Optional[object] = None
    forward_calls: int = 0

    @torch.no_grad()
    def logits(self, ids: list[int], need_from: Optional[int] = None) -> torch.Tensor:
        """Next-token logits after each prefix ids[:i+1] for i = need_from .. len(ids)-1.

        `need_from` defaults to the last position, i.e. just the logits for the next token.
        """
        need_from = len(ids) - 1 if need_from is None else need_from
        common = 0
        for a, b in zip(self.cached_ids, ids):
            if a != b:
                break
            common += 1
        keep = min(common, need_from)
        if self.past is not None and keep < len(self.cached_ids):
            # a negative value means "remove this many tokens from the end" (works across versions)
            self.past.crop(-(len(self.cached_ids) - keep))
        if keep == 0:
            self.past = None
        new = ids[keep:]
        out = self.lm.model(
            input_ids=torch.tensor([new], device=self.lm.device),
            past_key_values=self.past,
            use_cache=True,
        )
        self.forward_calls += 1
        self.past = out.past_key_values
        self.cached_ids = list(ids)
        return out.logits[0, need_from - keep:].float().cpu()

    # ---- the two roles in speculative decoding (see speculative.py) ----------------

    def propose(self, context: list[int], k: int) -> list[tuple[int, float]]:
        """Draft model role: greedily propose k tokens, each with the draft model's probability."""
        drafted, ctx = [], list(context)
        for _ in range(k):
            probs = torch.softmax(self.logits(ctx)[-1], dim=-1)
            token = int(probs.argmax())
            drafted.append((token, float(probs[token])))
            ctx.append(token)
        return drafted

    def verify(self, context: list[int], drafted: list[int]) -> tuple[list[int], list[float]]:
        """Target model role: one forward pass over context + drafted.

        Returns the target's own greedy choice at each of the len(drafted)+1 positions, and the
        target's probability of each drafted token.
        """
        rows = torch.softmax(self.logits(context + drafted, need_from=len(context) - 1), dim=-1)
        preds = [int(r.argmax()) for r in rows]
        p_drafted = [float(rows[i, t]) for i, t in enumerate(drafted)]
        return preds, p_drafted

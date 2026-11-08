# Small Language Model Explorer

An interactive Streamlit app for **seeing** how a small language model can stand in for, or speed
up, a large one. It runs two small open models (Qwen2.5 0.5B and 1.5B) on your laptop, so you can
inspect the probabilities, the losses, the quantized weights and every speculative-decoding round
instead of just reading about them.

It's a companion to the *Small Language Model* examples in
[Generative AI Design Patterns](https://github.com/lakshmanok/generative-ai-design-patterns)
(`examples/24_small_language_model`) and covers all five scripts.

> **New to this?** Read [docs/how-small-language-models-work.md](docs/how-small-language-models-work.md)
> first. It explains every idea in plain language with diagrams, worked numbers and real examples
> from this app. Terms are defined in [GLOSSARY.md](GLOSSARY.md).

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/), about 4 GB of free RAM, and an internet connection
for the first run (to download the two models, about 4 GB).

```bash
uv sync                          # installs Python 3.12 + dependencies into .venv
uv run streamlit run app.py      # opens http://localhost:8501
```

The small model loads when the app opens (around 5 to 10 seconds); the large one loads the first
time a tab needs it (around 15 to 30 seconds). Run the tests:

```bash
uv run pytest
```

### Configuration

| Setting | How | Default |
|---|---|---|
| Small model (student, draft model, gets quantized) | `SMALL_MODEL` in `slm_lab/lm.py` | `Qwen/Qwen2.5-0.5B-Instruct`, fp16 |
| Large model (teacher, target model) | `LARGE_MODEL` in `slm_lab/lm.py` | `Qwen/Qwen2.5-1.5B-Instruct`, fp16 |
| Device | `SLM_DEVICE=mps uv run streamlit run app.py` (or `cuda`) | `cpu` (or `cuda` if available) |

> **Why these settings?** The example scripts use Gemma 3 models of 1B to 27B parameters on a
> large GPU. On an 8 GB Apple M2 laptop, bf16 ran at about 7 s per token, fp16 at about 0.07 s,
> and Apple's MPS backend hung on macOS 14. See [ADR 0003](docs/adr/0003-fp16-on-cpu.md).
> On an 8 GB machine, close other heavy apps: swapping to disk makes everything much slower.

---

## How the app works

There are five tabs, meant to be followed in order. Each tab matches one script in the example
folder.

| Tab | Example script | What you see |
|---|---|---|
| ① Small vs large | `model_comparison.py` | Same prompt through both models: outputs, speed, memory; then both models' next-token probabilities side by side |
| ② Training data | `generate_training_set.py` | The 1000 Claude-generated snippets: how they were made, quality checks, a browser |
| ③ Knowledge distillation | `knowledge_distillation.py` | The distillation loss calculated on real teacher/student outputs, plus a live toy experiment |
| ④ Quantization | `model_quantization.py` | NF4 and int8 on one block of real weights, the memory arithmetic, and the quantized model's perplexity and output |
| ⑤ Speculative decoding | `speculative_decoding.py` | Draft + target vs target alone: speed, acceptance rate, and every round's accepted and rejected tokens |

### ① Small vs large

**You provide:** a prompt (default: the script's "generate documentation for this code" prompt,
with a shorter snippet) and a maximum length.

**What happens when you click *Run both models*:** each model formats the prompt with its chat
template and writes greedily (always the most likely token), reusing its KV cache. The app times
the first token (reading the prompt) separately from the rest (`slm_lab/pipelines.py`,
`generate_timed`).

**What you see:** parameters, weight memory, load time, time to first token, tokens per second,
and both outputs. Below that, *Compare next-token predictions* shows each model's top 10 next
tokens for the same text. This is the distribution distillation tries to copy.

### ② Training data

**No API calls.** The tab shows the generation prompt, the training template every snippet is
wrapped in, and the steps from Claude to tokenized dataset. Quality checks: duplicates, how many
snippets are complete and valid Python (most were cut off), the token-length histogram, and the most
common class names (most snippets are task managers). A browser with search lets you read any
snippet.

### ③ Knowledge distillation

**A. The loss on a real example.** Pick a training snippet and how many tokens to use, then click
*Run teacher and student*. Both models read the same text once (`distillation_inputs`). Moving the
**T** and **α** sliders recalculates instantly (`slm_lab/distillation.py`):

- average task loss, distillation loss, combined loss, and how often the student's top token matches the teacher's;
- a chart of both losses at every position;
- a position slider with the teacher's and student's probabilities at T = 1 and at your T.

**B. The toy experiment.** Two tiny students learn a made-up teacher from the same few examples,
one from hard labels only (α = 0), one distilled (your α and T). Charts show each student's
distance from the teacher on 500 unseen contexts as training goes on. It runs in about a second.

### ④ Quantization

Done by hand in PyTorch because bitsandbytes needs an NVIDIA GPU
([ADR 0002](docs/adr/0002-simulated-quantization.md)).

**A.** Pick any Linear layer and block: a table walks through absmax → scale → nearest level →
code → de-quantized value → error for all 64 weights, with a chart of the weights against the
NF4 (or int8) levels and a histogram of the whole layer.

**B.** A calculated memory table for fp32, fp16, int8, NF4 and NF4 + double quantization.

**C.** *Quantize and compare* swaps every quantizable weight for its quantized version, measures
perplexity on training snippets, and generates an answer to the script's prompt for original,
int8 and NF4 weights. The original weights are restored afterwards.

### ⑤ Speculative decoding

**You provide:** one of the script's prompts (or your own), the speculation length **k** (default
3, like the script), and a maximum length.

**What happens when you click *Run both ways*** (`slm_lab/speculative.py`, our own loop: see
[ADR 0001](docs/adr/0001-custom-speculative-decoding-loop.md)):

1. **Target alone:** the 1.5B model writes greedily, one forward pass per token.
2. **Speculative:** each round, the 0.5B model drafts k tokens; the 1.5B model checks all of them in
   one pass; drafts are accepted from the left while they match the target's own choice; the target
   then adds one token (a correction, or a bonus if every draft matched).

**What you see:** both speeds, the speed-up, the acceptance rate, whether the two outputs are
identical, the output coloured by token origin (accepted / correction / bonus), a chart of accepted
drafts per round, and a round inspector showing each draft's probability under both models and
what the sampling rule `min(1, p/q)` would have done.

---

## Project layout

```
app.py                     Streamlit UI (one section per tab)
slm_lab/
  lm.py                    the only module that touches the models; KV-cached runner
  pipelines.py             glue: timed generation, perplexity, distillation inputs, speculative comparison
  speculative.py           speculative decoding loop (pure, tested with fake models)
  distillation.py          loss functions and the toy experiment
  quantization.py          NF4/int8 quantize/de-quantize, weight swapping, memory arithmetic
  training_data.py         loading and checking the synthetic dataset
  prompts.py               prompts from the example scripts
data/python_code_examples.json   the 1000 snippets from generate_training_set.py
docs/how-small-language-models-work.md   the beginner's guide
docs/adr/                  why the app departs from the example scripts
tests/                     pytest suite for the pure logic
```

"""Small Language Model Explorer: a Streamlit app for seeing how small models replace or speed up large ones.

Run with:  uv run streamlit run app.py
"""

import html
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer

from slm_lab import prompts
from slm_lab.distillation import ToyConfig, per_position_losses, run_toy_experiment, soften
from slm_lab.lm import LARGE_MODEL, SMALL_MODEL, LanguageModel
from slm_lab.pipelines import (
    compare_speculative,
    distillation_inputs,
    generate_timed,
    perplexity,
    top_next_tokens,
)
from slm_lab.quantization import (
    BLOCK_SIZE,
    NF4_LEVELS,
    SCHEMES,
    levels_for,
    memory_table,
    quantizable_linears,
    quantize,
    quantized_weights,
)
from slm_lab.speculative import round_starts, sampling_acceptance_probability, verify_rows
from slm_lab.training_data import (
    GENERATION_SYSTEM,
    GENERATION_USER,
    TEST_SPLIT,
    TRAINING_TEMPLATE,
    TRUNCATION_LENGTH,
    class_names,
    load_examples,
    parses_as_python,
    report,
)

DATA = Path(__file__).parent / "data"
DOC_LINK = "docs/how-small-language-models-work.md"
PROB = st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)

st.set_page_config(page_title="Small Language Model Explorer", page_icon="🐣", layout="wide")

# Results kept in session_state are slm_lab objects. When that code changes while the app is
# running, old results keep their old shape and could crash a tab, so drop them on code changes.
RESULT_KEYS = ("cmp", "cmp_next", "dist", "toy", "quant", "spec")


def code_version() -> str:
    files = sorted((Path(__file__).parent / "slm_lab").glob("*.py"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)


if st.session_state.get("_code_version") != (version := code_version()):
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_code_version"] = version


@st.cache_resource(show_spinner="Loading the small model (Qwen2.5-0.5B)…")
def small_model() -> LanguageModel:
    return LanguageModel(SMALL_MODEL)


@st.cache_resource(show_spinner="Loading the large model (Qwen2.5-1.5B)…")
def large_model() -> LanguageModel:
    return LanguageModel(LARGE_MODEL)


@st.cache_resource
def tokenizer():
    return AutoTokenizer.from_pretrained(SMALL_MODEL)


@st.cache_data
def examples() -> tuple[list[str], dict]:
    return load_examples(DATA / "python_code_examples.json")


@st.cache_data
def token_lengths() -> tuple[list[int], list[int]]:
    """Length of each raw example, and of each example wrapped in the training template."""
    tok = tokenizer()
    ex, _ = examples()
    raw = [len(tok(e)["input_ids"]) for e in ex]
    wrapped = [len(tok(TRAINING_TEMPLATE.format(code=e))["input_ids"]) for e in ex]
    return raw, wrapped


def label(token_id: int) -> str:
    return small_model().token_label(token_id)


def visible(text: str) -> str:
    return text.replace("\n", "⏎ ")


def fmt_bytes(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def prob_table(rows: list[tuple[str, float]], col: str = "probability") -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["token", col])


# ---- sidebar ----------------------------------------------------------------------------

with st.sidebar:
    st.title("🐣 Small Language Model Explorer")
    st.markdown(
        f"**Small model:** `{SMALL_MODEL.split('/')[-1]}`  \n"
        "student · draft model · gets quantized\n\n"
        f"**Large model:** `{LARGE_MODEL.split('/')[-1]}`  \n"
        "teacher · target model"
    )
    st.caption(
        "The example scripts use Gemma 3 models (1B student, 12B teacher, 27B target with a 4B draft). "
        "Those need a big GPU; these two Qwen models play the same roles on a laptop."
    )
    st.markdown(
        "**Learning path**\n"
        "1. Small vs large\n"
        "2. Training data\n"
        "3. Knowledge distillation\n"
        "4. Quantization\n"
        "5. Speculative decoding\n\n"
        f"Read the companion guide: `{DOC_LINK}`"
    )
    st.caption("Tip: on an 8 GB laptop, close other heavy apps. Both models together need about 4 GB; if the Mac starts swapping, everything (especially speculative decoding) slows down a lot.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "① Small vs large", "② Training data", "③ Knowledge distillation", "④ Quantization", "⑤ Speculative decoding",
])

# ---- ① small vs large -------------------------------------------------------------------

with tab1:
    st.header("① Small vs large")
    st.markdown(
        "A **small language model** has fewer parameters (the numbers learned in training). Fewer "
        "parameters means less memory and less arithmetic per token, so it is faster and cheaper, but "
        "it usually knows less and writes worse. This tab runs the same prompt through both models "
        "so you can see the trade-off. It follows `model_comparison.py`, which asks for documentation "
        "of a Python snippet. Both models use greedy decoding (always the most likely token)."
    )

    c1, c2 = st.columns([3, 1])
    cmp_prompt = c1.text_area("Prompt", prompts.COMPARISON_PROMPT, height=220, key="cmp_prompt")
    cmp_max = c2.slider("Max new tokens", 16, 256, 128, step=16, key="cmp_max")
    c2.caption("The large model takes roughly 10 to 40 s to read this prompt on a laptop CPU, "
               "then about 3 to 8 tokens per second.")
    if c2.button("Run both models", type="primary", key="cmp_go"):
        results = []
        for get in (small_model, large_model):
            lm = get()
            with st.spinner(f"{lm.short_name} is writing…"):
                results.append((lm, generate_timed(lm, lm.chat_prompt_ids(cmp_prompt), cmp_max)))
        st.session_state.cmp = results

    if res := st.session_state.get("cmp"):
        cols = st.columns(2)
        for col, (lm, g) in zip(cols, res):
            col.subheader(lm.short_name)
            m1, m2, m3 = col.columns(3)
            m1.metric("Parameters", f"{lm.num_params / 1e9:.2f} B")
            m2.metric("Weights in memory", fmt_bytes(lm.weight_bytes), help=f"Stored as {str(lm.dtype)[6:]}")
            m3.metric("Load time", f"{lm.load_seconds:.1f} s")
            m4, m5, m6 = col.columns(3)
            m4.metric("Prompt tokens", g.prompt_tokens)
            m5.metric("Time to first token", f"{g.first_token_seconds:.1f} s",
                      help="Reading the whole prompt and picking the first token")
            m6.metric("Tokens / second", f"{g.tokens_per_second:.1f}", help="Speed after the first token")
            col.markdown("**Output**")
            col.code(g.text, language="markdown", wrap_lines=True)
        (_, gs), (_, gl) = res
        if gl.tokens_per_second and gs.tokens_per_second:
            st.info(
                f"The small model generated **{gs.tokens_per_second / gl.tokens_per_second:.1f}×** as many "
                f"tokens per second and read the prompt **{gl.first_token_seconds / max(gs.first_token_seconds, 1e-6):.1f}×** "
                "faster. Compare the outputs: is the large model's documentation more accurate or complete?"
            )

    st.divider()
    st.subheader("Same text, two opinions: the next-token distributions")
    st.markdown(
        "Every model outputs a probability for every token in its vocabulary. The two Qwen models share "
        "the same vocabulary of about 151,000 tokens, so their predictions can be compared one to one. "
        "This is exactly what knowledge distillation (tab ③) trains the small model to copy."
    )
    nt_text = st.text_area("Text", prompts.NEXT_TOKEN_TEXT, height=100, key="nt_text")
    if st.button("Compare next-token predictions", key="nt_go"):
        st.session_state.cmp_next = {
            lm.short_name: top_next_tokens(lm, nt_text, 10) for lm in (small_model(), large_model())
        }
    if nxt := st.session_state.get("cmp_next"):
        cols = st.columns(2)
        for col, (name, toks) in zip(cols, nxt.items()):
            col.markdown(f"**{name}**: top 10 next tokens")
            col.dataframe(prob_table([(label(t.token_id), t.prob) for t in toks]),
                          column_config={"probability": PROB}, hide_index=True, width="stretch")

# ---- ② training data --------------------------------------------------------------------

with tab2:
    st.header("② Training data")
    st.markdown(
        "Before you can distill a teacher into a student you need text to run through both models. "
        "`generate_training_set.py` asked Claude for **1000 Python snippets** and saved them to "
        "`python_code_examples.json` (bundled here). This tab makes **no API calls**; it lets you look at "
        "what was generated, and checks how good the data is."
    )
    ex, meta = examples()
    st.markdown("#### How the data was made")
    st.code(
        f"For each of {meta.get('num_examples', len(ex))} examples:\n"
        f"  1. Ask {meta.get('model_used', 'Claude')} (temperature 0.7):\n"
        f"       system: {GENERATION_SYSTEM}\n"
        f"       user:   {GENERATION_USER}\n"
        "  2. Strip the ```python fences, keep the code.\n"
        "Then:\n"
        "  3. Wrap every snippet in the training template (below).\n"
        f"  4. Split {100 - TEST_SPLIT * 100:.0f}% train / {TEST_SPLIT * 100:.0f}% validation "
        f"(≈ {len(ex) * (1 - TEST_SPLIT):.0f} / {len(ex) * TEST_SPLIT:.0f}).\n"
        f"  5. Tokenize, padding or cutting every text to {TRUNCATION_LENGTH} tokens.",
        language="text",
    )
    with st.expander("The training template (what the student actually reads)"):
        st.code(TRAINING_TEMPLATE, language="text")
        st.caption("Note there is no documentation after `Documentation:`. The models are trained to "
                   "predict the prompt and the code itself, token by token. See the guide.")

    st.markdown("#### Data quality checks")
    rep = report(ex)
    raw_len, wrapped_len = token_lengths()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Examples", rep.num_examples)
    m2.metric("Unique", rep.num_unique, delta=f"{rep.num_unique - rep.num_examples} duplicates", delta_color="off")
    m3.metric("Complete, valid Python", f"{rep.num_parse} ({rep.num_parse / rep.num_examples:.0%})",
              help="Parsed with Python's own parser. Failures are almost all snippets cut off mid-line.")
    m4.metric(f"Contain `class {rep.class_counts[0][0]}`", f"{rep.share_with_top_class:.0%}")
    st.caption(
        f"Wrapped texts are {min(wrapped_len)} to {max(wrapped_len)} tokens long, so none hits the "
        f"{TRUNCATION_LENGTH}-token limit; each is padded with about {TRUNCATION_LENGTH - max(wrapped_len)} "
        f"to {TRUNCATION_LENGTH - min(wrapped_len)} padding tokens."
    )

    c1, c2 = st.columns(2)
    c1.markdown("**Length of each snippet (tokens)**")
    c1.altair_chart(
        alt.Chart(pd.DataFrame({"tokens": raw_len})).mark_bar().encode(
            alt.X("tokens:Q", bin=alt.Bin(maxbins=30), title="tokens per snippet"), alt.Y("count()", title="snippets")
        ), width="stretch",
    )
    c2.markdown("**Most common class names**")
    c2.altair_chart(
        alt.Chart(pd.DataFrame(rep.class_counts, columns=["class", "snippets"])).mark_bar().encode(
            alt.X("snippets:Q"), alt.Y("class:N", sort="-x")
        ), width="stretch",
    )
    st.warning(
        "Asking one model the same question 1000 times gives **similar answers**: most snippets are a "
        "task manager, and most were cut off before the end. A student trained on this learns that "
        "narrow slice well. Real projects add variety to the prompt (topics, styles, difficulty) and "
        "filter out broken examples."
    )

    st.markdown("#### Browse the snippets")
    c1, c2 = st.columns([1, 2])
    query = c2.text_input("Only show snippets containing", "", key="td_query")
    matches = [i for i, e in enumerate(ex) if query.lower() in e.lower()] if query else list(range(len(ex)))
    if not matches:
        st.info("No snippet contains that text.")
    else:
        idx = c1.selectbox(f"Snippet ({len(matches)} match)", matches, key="td_idx")
        code = ex[idx]
        st.caption(
            f"{raw_len[idx]} tokens · classes: {', '.join(class_names(code)) or 'none'} · "
            + ("✅ valid Python" if parses_as_python(code) else "⚠️ not valid Python (probably cut off)")
        )
        st.code(code, language="python")

# ---- ③ knowledge distillation -----------------------------------------------------------

with tab3:
    st.header("③ Knowledge distillation")
    st.markdown(
        "**Knowledge distillation** trains a small **student** to imitate a large **teacher**. Ordinary "
        "training only tells the student which token came next (the **hard label**). Distillation also "
        "shows it the teacher's whole probability distribution (the **soft label**): which wrong tokens "
        "were *almost* right. That extra information helps the student learn more from each example."
    )
    st.latex(r"\text{loss} = (1-\alpha)\cdot\underbrace{\text{CE}(\text{student},\ \text{hard label})}_{\text{task loss}}"
             r"\;+\;\alpha\cdot\underbrace{T^2\cdot \text{KL}\big(\text{softmax}(\tfrac{\text{teacher}}{T})\ \|\ "
             r"\text{softmax}(\tfrac{\text{student}}{T})\big)}_{\text{distillation loss}}")
    st.caption("This is `DistillationTrainer.compute_loss` from knowledge_distillation.py "
               "(defaults T = 2, α = 0.5). Training the student for real needs a GPU, so this tab does two "
               "things instead: part A calculates the loss on real teacher and student outputs, and part B "
               "trains a tiny toy student live to show why soft labels help.")

    st.subheader("A. The loss on a real training example")
    c1, c2, c3 = st.columns([1, 1, 1])
    ex, _ = examples()
    d_idx = c1.number_input("Training snippet #", 0, len(ex) - 1, 0, key="d_idx")
    d_len = c2.slider("Tokens to use", 32, 160, 64, step=16, key="d_len")
    if c3.button("Run teacher and student", type="primary", key="d_go"):
        text = TRAINING_TEMPLATE.format(code=ex[d_idx])
        with st.spinner("Running the same text through both models…"):
            st.session_state.dist = distillation_inputs(small_model(), large_model(), text, d_len)

    if dist := st.session_state.get("dist"):
        c1, c2 = st.columns(2)
        T = c1.slider("Distillation temperature T", 0.5, 5.0, 2.0, step=0.5, key="d_T",
                      help="Logits are divided by T before softmax. Higher T flattens both distributions.")
        alpha = c2.slider("α (weight on distillation loss)", 0.0, 1.0, 0.5, step=0.05, key="d_alpha")
        rows = per_position_losses(dist.student_logits, dist.teacher_logits, dist.ids, T, alpha)
        n = len(rows)
        mean = lambda f: sum(f(r) for r in rows) / n
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Task loss (avg)", f"{mean(lambda r: r.task):.3f}", help="Cross-entropy against the hard label")
        m2.metric("Distillation loss (avg)", f"{mean(lambda r: r.distill):.3f}", help="T² × KL(teacher_T ‖ student_T)")
        m3.metric("Combined loss (avg)", f"{mean(lambda r: r.combined):.3f}")
        m4.metric("Student's top pick = teacher's",
                  f"{sum(int(dist.student_logits[r.position - 1].argmax()) == int(dist.teacher_logits[r.position - 1].argmax()) for r in rows) / n:.0%}")

        df = pd.DataFrame({
            "position": [r.position for r in rows],
            "task loss": [r.task for r in rows],
            "distillation loss": [r.distill for r in rows],
        }).melt("position", var_name="loss", value_name="value")
        st.markdown("**Loss at every position** (spikes are tokens the student finds surprising, or where it disagrees with the teacher)")
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="position:Q", y="value:Q", color="loss:N"),
                        width="stretch")

        worst = max(rows, key=lambda r: r.distill).position
        pos = st.slider("Inspect a position", 1, n, worst, key="d_pos",
                        help="Starts at the position with the highest distillation loss")
        r = rows[pos - 1]
        st.markdown(
            f"Text so far: `…{visible(small_model().decode(dist.ids[:pos])[-120:])}` → hard label "
            f"**`{label(r.label)}`**. Teacher gives it **{r.teacher_p_label:.1%}**, student **{r.student_p_label:.1%}**. "
            f"Task loss {r.task:.3f} · distillation loss {r.distill:.3f}."
        )
        s_logits, t_logits = dist.student_logits[pos - 1], dist.teacher_logits[pos - 1]
        t1, tT, s1, sT = soften(t_logits, 1), soften(t_logits, T), soften(s_logits, 1), soften(s_logits, T)
        top = torch.topk(tT + sT, 10).indices.tolist()
        table = pd.DataFrame({
            "token": [label(i) + (" ← hard label" if i == r.label else "") for i in top],
            "teacher (T=1)": [float(t1[i]) for i in top],
            f"teacher (T={T:g}) = soft label": [float(tT[i]) for i in top],
            "student (T=1)": [float(s1[i]) for i in top],
            f"student (T={T:g})": [float(sT[i]) for i in top],
        })
        st.dataframe(table, column_config={c: PROB for c in table.columns[1:]}, hide_index=True, width="stretch")
        st.caption("Raise T and watch the teacher's small probabilities grow: that is the 'almost right' "
                   "information the hard label throws away. The distillation loss measures how far the "
                   "student's column is from the soft label column.")

    st.subheader("B. Why soft labels help: a toy experiment")
    st.markdown(
        "A made-up **teacher** maps a context (8 numbers) to a distribution over 20 tokens, which come in 4 "
        "groups of similar tokens. Two **students** with exactly the teacher's shape learn from the **same "
        "few contexts**. One only sees the hard label (one token sampled from the teacher), the other also "
        "sees the teacher's soft label. We then test both on 500 contexts neither has seen."
    )
    c1, c2, c3, c4 = st.columns(4)
    toy_n = c1.slider("Training examples", 5, 200, 20, step=5, key="toy_n")
    toy_T = c2.slider("T", 0.5, 5.0, 2.0, step=0.5, key="toy_T")
    toy_a = c3.slider("α", 0.0, 1.0, 0.5, step=0.05, key="toy_a")
    toy_seed = c4.number_input("Random seed", 0, 999, 0, key="toy_seed")
    if st.button("Train both students", type="primary", key="toy_go"):
        st.session_state.toy = run_toy_experiment(ToyConfig(n_train=toy_n, temperature=toy_T, alpha=toy_a, seed=toy_seed))
    if toy := st.session_state.get("toy"):
        names = list(toy.heldout_kl)
        kl = pd.DataFrame({"epoch": toy.epochs, **toy.heldout_kl}).melt("epoch", var_name="student", value_name="KL")
        ag = pd.DataFrame({"epoch": toy.epochs, **toy.heldout_agree}).melt("epoch", var_name="student", value_name="agreement")
        c1, c2 = st.columns(2)
        c1.markdown("**Distance from the teacher on unseen contexts** (KL, lower is better)")
        c1.altair_chart(alt.Chart(kl).mark_line().encode(x="epoch:Q", y="KL:Q", color="student:N"), width="stretch")
        c2.markdown("**Top token matches the teacher's** (share of unseen contexts)")
        c2.altair_chart(alt.Chart(ag).mark_line().encode(x="epoch:Q", y=alt.Y("agreement:Q", scale=alt.Scale(domain=[0, 1])),
                                                         color="student:N"), width="stretch")
        final = {n: toy.heldout_kl[n][-1] for n in names}
        st.info(f"After training: **{names[0]}** KL = {final[names[0]]:.3f}, **{names[1]}** KL = {final[names[1]]:.3f}. "
                "Try fewer examples (the gap grows), more examples (the gap shrinks), or α = 0 (both students become identical).")
        bars = pd.DataFrame({"token": list(range(len(toy.teacher_example))), "teacher": toy.teacher_example,
                             **toy.student_example}).melt("token", var_name="model", value_name="probability")
        st.markdown("**One unseen context: the teacher's distribution vs each student's**")
        st.altair_chart(alt.Chart(bars).mark_bar().encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labels=False, ticks=False)), y="probability:Q",
            color="model:N", column=alt.Column("token:O", title="token (tokens 0, 4, 8, … are one group)")),
        )

# ---- ④ quantization ---------------------------------------------------------------------

with tab4:
    st.header("④ Quantization")
    st.markdown(
        "**Quantization** stores each weight with fewer bits. A 32-bit float becomes an 8-bit or even "
        "4-bit code, so the model takes 4 to 8 times less memory. `model_quantization.py` loads a model in "
        "**NF4** (4-bit NormalFloat) with bitsandbytes. bitsandbytes needs an NVIDIA GPU, so this tab does "
        "the same arithmetic in plain PyTorch on the small model. You see the exact accuracy effect; the "
        "memory figures are **calculated**, and there is **no real speed-up** here because the weights are "
        "turned back into ordinary floats to run."
    )
    lm_q = small_model()

    st.subheader("A. Quantizing one block of 64 weights")
    linears = list(quantizable_linears(lm_q.model))
    c1, c2, c3 = st.columns([2, 1, 1])
    layer_name = c1.selectbox("Layer", linears, index=linears.index("model.layers.0.mlp.down_proj"), key="q_layer")
    w = quantizable_linears(lm_q.model)[layer_name].weight.detach().float().flatten()
    n_blocks = (w.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    block = c2.number_input(f"Block (of {n_blocks:,})", 0, n_blocks - 1, 0, key="q_block")
    scheme_a = c3.radio("Format", ["nf4", "int8"], horizontal=True, key="q_scheme_a")
    blk = w[block * BLOCK_SIZE:(block + 1) * BLOCK_SIZE]
    q = quantize(blk, levels_for(scheme_a), block_size=BLOCK_SIZE)
    absmax = float(q.absmax[0])
    codes = q.codes.flatten()[: blk.numel()].long()
    deq = q.levels[codes] * absmax
    st.markdown(
        f"Step 1: the block's largest absolute value is **absmax = {absmax:.5f}**. Step 2: divide every "
        f"weight by it, so they all fall between −1 and 1. Step 3: replace each with the nearest of the "
        f"{len(q.levels)} levels and store only that level's number (its **code**, {4 if scheme_a == 'nf4' else 8} bits). "
        "Step 4 (when the model runs): code → level × absmax."
    )
    block_df = pd.DataFrame({
        "original weight": blk.tolist(),
        "÷ absmax": (blk / absmax).tolist(),
        "code": codes.tolist(),
        "level": q.levels[codes].tolist(),
        "de-quantized": deq.tolist(),
        "error": (deq - blk).tolist(),
    })
    c1, c2 = st.columns([3, 2])
    c1.dataframe(block_df.style.format("{:.5f}", subset=["original weight", "÷ absmax", "level", "de-quantized", "error"]),
                 hide_index=True, width="stretch", height=320)
    pts = alt.Chart(pd.DataFrame({"value ÷ absmax": (blk / absmax).tolist(), "kind": "weight"})).mark_tick(
        thickness=2, size=30, color="#4c78a8").encode(x=alt.X("value ÷ absmax:Q", scale=alt.Scale(domain=[-1.05, 1.05])))
    lv = alt.Chart(pd.DataFrame({"value ÷ absmax": q.levels.tolist()})).mark_rule(color="#e45756", strokeDash=[3, 3]).encode(x="value ÷ absmax:Q")
    c2.markdown("**The block's weights (blue) and the available levels (red dashes)**")
    c2.altair_chart((lv + pts).properties(height=120), width="stretch")
    if scheme_a == "nf4":
        c2.caption("NF4's 16 levels are packed tightly near 0, where most weights are, and spread out "
                   "near ±1, where few are. That's why it beats evenly spaced 4-bit levels.")
    sample = w[torch.randperm(w.numel(), generator=torch.Generator().manual_seed(0))[:20000]]
    c2.markdown("**All weights in this layer** (a sample of 20,000): bell-shaped, centred on 0")
    c2.altair_chart(alt.Chart(pd.DataFrame({"weight": sample.tolist()})).mark_bar().encode(
        alt.X("weight:Q", bin=alt.Bin(maxbins=60)), alt.Y("count()")).properties(height=150), width="stretch")

    st.subheader("B. How much memory each format needs")
    rows = memory_table(lm_q.model)
    mem = pd.DataFrame({
        "format": [r.label for r in rows],
        "bits per quantized weight": [round(r.bits_quantized, 3) for r in rows],
        "estimated size": [fmt_bytes(r.bytes) for r in rows],
        "vs fp32": [f"{r.bytes / rows[0].bytes:.0%}" for r in rows],
    })
    st.dataframe(mem, hide_index=True, width="stretch")
    r0 = rows[0]
    st.caption(
        f"{lm_q.short_name}: {r0.quantized_params / 1e6:.0f} M weights live in the Linear layers that get quantized; "
        f"{r0.other_params / 1e6:.0f} M (mostly the token embedding table) stay in 16-bit, as bitsandbytes does. "
        "Each block of 64 also stores its absmax (32 bits, +0.5 bit per weight); double quantization stores "
        "those in 8 bits. The example script uses NF4 + double quantization."
    )

    st.subheader("C. Does the quantized model still work?")
    st.markdown(
        "**Perplexity** measures how surprised a model is by text: lower is better. A perplexity of 3 means the "
        "model is, on average, about as unsure as if it were choosing between 3 equally likely tokens. We measure it on the training "
        "snippets, and also generate an answer to the script's prompt."
    )
    c1, c2, c3 = st.columns(3)
    q_n = c1.slider("Snippets for perplexity", 5, 50, 10, step=5, key="q_n",
                     help="Each snippet is cut to its first 128 tokens. About 1 to 2 s per snippet per format.")
    q_prompt = c2.text_input("Prompt", prompts.QUANTIZATION_PROMPT, key="q_prompt")
    q_max = c3.slider("Max new tokens", 16, 256, 128, step=16, key="q_max")
    if st.button("Quantize and compare", type="primary", key="q_go"):
        ex, _ = examples()
        out = []
        for scheme in SCHEMES:
            with st.spinner(f"Running the {scheme} model…"), quantized_weights(lm_q.model, scheme) as stats:
                ppl = perplexity(lm_q, ex[-q_n:], 128)
                gen = generate_timed(lm_q, lm_q.chat_prompt_ids(q_prompt), q_max)
            out.append((stats, ppl, gen))
        st.session_state.quant = out
    if res := st.session_state.get("quant"):
        bits = {"original": 32, "int8": 8.5, "nf4": 4.5}
        st.dataframe(pd.DataFrame({
            "weights": [s.scheme for s, _, _ in res],
            "perplexity": [round(p.value, 3) for _, p, _ in res],
            "vs original": [f"{p.value / res[0][1].value - 1:+.1%}" for _, p, _ in res],
            "avg weight error": [f"{s.mean_abs_error / s.mean_abs_weight:.1%} of avg |weight|" for s, _, _ in res],
            "bits per quantized weight": [bits[s.scheme] for s, _, _ in res],
            "tokens scored": [p.num_tokens for _, p, _ in res],
        }), hide_index=True, width="stretch")
        cols = st.columns(len(res))
        for col, (s, _, g) in zip(cols, res):
            col.markdown(f"**{s.scheme}**")
            col.code(g.text, language="markdown", wrap_lines=True)
        st.caption("The snippets used for perplexity are the last ones in the file. Greedy decoding is used, "
                   "so any difference between outputs comes from the weights alone.")

# ---- ⑤ speculative decoding -------------------------------------------------------------

with tab5:
    st.header("⑤ Speculative decoding")
    st.markdown(
        "A large model is slow mainly because each token needs a full pass through all its weights, and "
        "checking several tokens in one pass costs little more than generating one. **Speculative decoding** "
        "takes advantage of this. The small **draft model** quickly guesses the next *k* tokens, then the "
        "**target model** checks all of them in a single pass. Tokens it agrees with are kept for free; at "
        "the first disagreement it substitutes its own token. With greedy checking the output is **exactly "
        "what the target model would have written alone**, just sooner. `speculative_decoding.py` does this "
        "with vLLM (27B target, 4B draft, k = 3)."
    )
    c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
    preset = c1.selectbox("Prompt (from the script)", prompts.SPECULATIVE_PROMPTS + ["Custom…"], key="s_preset")
    s_prompt = c2.text_input("Custom prompt", "def quicksort(arr):", key="s_custom",
                             disabled=preset != "Custom…") if preset == "Custom…" else preset
    if preset != "Custom…":
        c2.text_input("Custom prompt", "", key="s_custom_off", disabled=True, placeholder="choose Custom…")
    s_k = c3.slider("Speculation length k", 1, 8, 3, key="s_k")
    s_max = c4.slider("Max new tokens", 16, 256, 128, step=16, key="s_max")
    if st.button("Run both ways", type="primary", key="s_go"):
        with st.spinner("Target alone, then draft + target…"):
            st.session_state.spec = compare_speculative(small_model(), large_model(), s_prompt, s_k, s_max)

    if res := st.session_state.get("spec"):
        b, s = res.baseline, res.speculative
        tgt = large_model()
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Target alone", f"{b.tokens_per_second:.2f} tok/s", help=f"{b.target_calls} target passes")
        m2.metric("Speculative", f"{s.tokens_per_second:.2f} tok/s", help=f"{s.target_calls} target passes")
        m3.metric("Speed-up", f"{s.tokens_per_second / b.tokens_per_second:.2f}×" if b.tokens_per_second else "–")
        m4.metric("Acceptance rate", f"{s.acceptance_rate:.0%}", help="Share of drafted tokens the target agreed with")
        m5.metric("Target passes", f"{s.target_calls} vs {b.target_calls}")
        if b.tokens_per_second and s.tokens_per_second < b.tokens_per_second:
            st.warning(
                "Speculative decoding came out **slower** here. On a GPU that almost never happens; on a laptop "
                "it usually means the computer is swapping memory to disk. Speculative decoding switches between "
                "the two models every round, so if both don't fit in RAM, each switch has to reload weights. "
                f"The hardware-independent gain is the **target passes**: {s.target_calls} instead of {b.target_calls}. "
                "Close other apps and run again."
            )
        same = b.token_ids == s.token_ids
        (st.success if same else st.warning)(
            "✅ Identical output to the target model alone." if same else
            "⚠️ The outputs differ slightly. In fp16, checking several tokens in one pass can round differently "
            "from one-at-a-time, which occasionally flips a near-tie. See the guide."
        )

        rounds = s.rounds
        starts = round_starts(rounds)
        GREEN, ORANGE, BLUE, RED = "#c7f0c2", "#ffd9b3", "#cfe0ff", "#ffc9c9"

        def chip(text: str, bg: str, style: str = "", tip: str = "") -> str:
            t = f' title="{html.escape(tip)}"' if tip else ""
            return (f'<span{t} style="background:{bg};color:#111;border-radius:3px;padding:1px 2px;'
                    f'margin-right:1px;{style}">{html.escape(text)}</span>')

        st.markdown("**Step through the rounds**")
        rnum = st.slider("Round", 1, len(rounds), 1, key="s_round") if len(rounds) > 1 else 1
        ri = rnum - 1
        rd = rounds[ri]
        ctx_ids = res.prompt_ids + s.token_ids[: starts[ri]]
        added = s.token_ids[starts[ri]: starts[ri] + len(rd.added)]

        # The whole text so far: prompt, earlier rounds (the current "prompt"), this round, later rounds.
        spans = []
        for i, r in enumerate(rounds):
            for t in r.drafted[: r.num_accepted]:
                spans.append((GREEN, t, f"round {i + 1}: drafted and accepted", i))
            bonus = r.num_accepted == len(r.drafted)
            spans.append((BLUE if bonus else ORANGE, r.new_token,
                          f"round {i + 1}: {'bonus (all drafts accepted)' if bonus else 'target’s correction'}", i))
        spans = spans[: len(s.token_ids)]
        parts = [chip(tgt.decode(res.prompt_ids), "#e6e6e6", "color:#555", "the original prompt")]
        marked = False
        for c, t, tip, i in spans:
            if i == ri and not marked:
                parts.append('<b style="color:#e0245e">▶</b>')
                marked = True
            style = "outline:2px solid #e0245e;font-weight:bold" if i == ri else "opacity:0.25" if i > ri else ""
            parts.append(chip(tgt.decode([t]), c, style, tip))
        parts = [p.replace("\n", "⏎<br>") for p in parts]
        st.markdown(f'<div style="line-height:2;font-family:monospace">{"".join(parts)}</div>',
                    unsafe_allow_html=True)
        st.caption(f"Round {rnum}: the target's **prompt** is the grey prompt plus everything before ▶ "
                   f"({len(ctx_ids)} tokens). The outlined tokens are what this round added: **{len(added)} "
                   f"token{'s' if len(added) != 1 else ''} for one target pass**. Faded tokens come from later "
                   "rounds. 🟩 drafted and accepted · 🟧 the target's correction at the first mismatch · "
                   "🟦 bonus token when all k drafts were accepted. Hover a token for its round.")

        # Step 1: the draft writes k tokens, one after another.
        rows = verify_rows(rd)
        k = len(rd.drafted)
        tail = visible(tgt.decode(ctx_ids))
        tail_short = ("…" + tail[-60:]) if len(tail) > 60 else tail
        lead_in = ("…" + tail[-25:]) if len(tail) > 25 else tail
        strip = " → ".join([f'<span style="opacity:.6">{html.escape(lead_in)}</span>']
                           + [chip(label(t), "#fff3b0") for t in rd.drafted])
        st.markdown(f"**① Draft model: {k} guess{'es' if k != 1 else ''}, one after another** "
                    f"({k} small pass{'es' if k != 1 else ''}; each guess needs the one before it)")
        st.markdown(f'<div style="font-family:monospace;line-height:2">{strip}</div>', unsafe_allow_html=True)

        # Step 2: the target checks all k + 1 rows in one pass.
        st.markdown(f"**② Target model: checks all {k + 1} prompts in ONE pass, in parallel**")
        result_text = {
            "accepted": "✅ accepted (matches)",
            "rejected": "❌ rejected → keep the target's pick (correction)",
            "discarded": "– discarded: its prompt contains a rejected draft",
            "bonus": "🟦 bonus: all drafts accepted → keep the target's pick",
            "unused": "– unused: an earlier draft was rejected",
        }
        draft_bg = {"accepted": GREEN, "rejected": RED, "discarded": "#e6e6e6"}
        pick_bg = {"accepted": GREEN, "rejected": ORANGE, "bonus": BLUE}

        def prob_cell(x: float | None) -> str:
            if x is None:
                return ""
            pct = round(100 * x)
            return (f'<div style="background:linear-gradient(90deg,rgba(90,140,255,.35) {pct}%,transparent {pct}%);'
                    f'padding:0 4px">{x:.3f}</div>')

        td = 'style="padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.25);vertical-align:middle"'
        head = ["", "row", "prompt given to the target (its last tokens)", "draft proposed", "target's pick",
                "result (greedy)", "draft prob q", "target prob of draft p", "sampling rule min(1, p/q)"]
        trs = []
        for j, row in enumerate(rows):
            prompt_cell = (f'<span style="opacity:.6" title="{html.escape(tail)}">{html.escape(tail_short)}</span>'
                           + "".join(chip(label(t), draft_bg[rows[m].status],
                                          "text-decoration:line-through" if rows[m].status == "discarded" else "")
                                     for m, t in enumerate(row.drafted_before)))
            proposed = chip(label(row.proposed), "#fff3b0") if row.proposed is not None else \
                '<span style="opacity:.6">— (no draft: bonus slot)</span>'
            pick = chip(label(row.target_pick), pick_bg[row.status], "font-weight:bold") if row.kept else \
                chip(label(row.target_pick), "#e6e6e6", "opacity:.6")
            q = rd.draft_probs[j] if j < k else None
            p = rd.target_probs[j] if j < k else None
            rule = sampling_acceptance_probability(p, q) if j < k else None
            lead = (f'<td rowspan="{k + 1}" style="padding:6px 10px;background:rgba(224,36,94,.08);'
                    f'border-left:4px solid #e0245e;font-weight:bold;text-align:center;vertical-align:middle">'
                    f'⚡ ONE<br>target<br>pass<br><span style="font-weight:normal;font-size:.85em">'
                    f'{k + 1} rows<br>in parallel</span></td>') if j == 0 else ""
            cells = [str(j + 1), prompt_cell, proposed, pick, result_text[row.status],
                     prob_cell(q), prob_cell(p), prob_cell(rule)]
            trs.append("<tr>" + lead + "".join(f"<td {td}>{c}</td>" for c in cells) + "</tr>")
        th = 'style="padding:4px 8px;border-bottom:1px solid rgba(128,128,128,.5);text-align:left"'
        ths = "".join(f"<th {th}>{h}</th>" for h in head)
        st.markdown(f'<div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;'
                    f'font-size:.9em"><thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>',
                    unsafe_allow_html=True)

        kind = "bonus" if rd.num_accepted == k else "correction"
        st.markdown(
            f"**③ Result:** {rd.num_accepted} accepted draft{'s' if rd.num_accepted != 1 else ''} + the target's "
            f"{kind} `{label(rd.new_token)}` = **{len(added)} token{'s' if len(added) != 1 else ''} from 1 target "
            f"pass**. The target alone would need {len(added)} pass{'es' if len(added) != 1 else ''}, one token at "
            "a time, because each token's prompt includes the token before it."
        )
        st.caption("Row j's prompt is the text so far plus the first j−1 drafted tokens. The target can run all "
                   "rows in one pass because the drafts are already known, and its causal mask stops each row "
                   "from seeing the tokens after it. Rows after the first rejection were computed from text "
                   "that will never exist, so they're thrown away. The last three columns show what the sampling "
                   "version of speculative decoding would do (the app uses the greedy rule).")

        acc = pd.DataFrame({"accepted drafts": [r.num_accepted for r in rounds]})
        c1, _ = st.columns([1, 2])
        c1.markdown("**Accepted drafts per round**")
        c1.altair_chart(alt.Chart(acc).mark_bar().encode(alt.X("accepted drafts:O"), alt.Y("count()", title="rounds")),
                        width="stretch")
        c1.caption(f"Each round adds accepted + 1 tokens. Draft time {sum(r.draft_seconds for r in rounds):.1f} s, "
                   f"target time {sum(r.target_seconds for r in rounds):.1f} s in total.")
        st.caption("On a laptop CPU the speed-up is smaller than on a GPU: a CPU pays more per extra token "
                   "checked, and the draft model here is a third of the target's size (the script's is about "
                   "a seventh). Timings on a busy laptop are noisy, so run it twice before drawing conclusions. "
                   "Try different k and prompts: predictable text gets more drafts accepted.")

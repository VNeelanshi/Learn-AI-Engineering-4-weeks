# Fine-Tuning LLMs: When to Do It, LoRA/QLoRA Theory, PEFT, and Unsloth

A hands-on beginner's guide. No prior knowledge assumed. Every term is defined the first time it appears. Roughly 70% explanation, 30% practice — each section ends with an exercise, and every section closes with five interview questions and answers.

A few definitions to anchor us:

- **LLM (Large Language Model):** a program (like GPT, Claude, Llama, or Qwen) that takes text in and produces text out.
- **Weights / parameters:** the millions or billions of numbers inside the model that determine its behavior. Training is the process of adjusting these numbers; a "7B" model has ~7 billion of them.
- **Prompt:** the text instructions you send to the model at the moment you ask it something.
- **Fine-tuning:** continuing to train an already-trained model on your own examples, so its weights shift to better fit your task. This *permanently changes the model's numbers*, unlike prompting, which changes only the instructions you send.

---

## Section 1 — When to Fine-Tune: Fine-Tuning vs. RAG vs. Prompting

### The problem: fine-tuning is the expensive tool people reach for first

Fine-tuning sounds like the "serious" way to customize an LLM, so beginners often jump to it. That's usually a mistake. Fine-tuning costs money and time, requires a dataset and a GPU, produces a model you now have to host and maintain, and — done wrong — performs *worse* than the base model with a good prompt. Before you train anything, you need a decision framework for when fine-tuning is actually the right fix, versus two cheaper approaches. Guidance from OpenAI's fine-tuning documentation and from practitioners like Hamel Husain converges on the same message: **start simple, measure, and escalate only when needed.**

### The one question that decides it

Almost every "should I fine-tune?" question reduces to: **do I need new facts, or new behavior?**

- If the model is missing *knowledge* (your company's docs, recent events, private data), fine-tuning is the wrong tool. Fine-tuning does **not** reliably teach facts.
- If the model needs to *behave differently* (a specific tone, a rigid output format, a narrow skill), that's what fine-tuning is for.

Let's define the three approaches, cheapest first.

**Prompting (prompt engineering).** You improve results purely by changing the text you send: a clear **system prompt** (standing instructions given to the model), and **few-shot examples** (a handful of input→output examples included in the prompt to show the model the pattern). Nothing about the model changes. It's the fastest to ship and free to iterate.

**RAG (Retrieval-Augmented Generation).** Before the model answers, a **retriever** searches a collection of your documents for relevant chunks and inserts them into the prompt as context. (Retrieval is usually done with **embeddings** — numeric representations of text where similar meanings sit close together — stored in a **vector database**.) It's like turning a closed-book exam into an open-book one: the model looks facts up instead of memorizing them. RAG is the right tool for knowledge that is private, large, or changes over time, because you update the documents, not the model.

**Fine-tuning.** You train the model's weights on many input→output examples so the new behavior is baked in. This is **SFT (Supervised Fine-Tuning)** — "supervised" because each training example includes the desired output (the supervision signal).

### The decision framework

Work in this order and don't skip steps:

1. **Try prompting first.** Write a clear system prompt, add a few examples, and test on ~50 real inputs. If accuracy is acceptable, ship it.
2. **If knowledge is missing, add RAG.** Index the relevant documents, wire up retrieval, test again.
3. **If behavior/style/format is still wrong, or the task is narrow and high-volume, consider fine-tuning.**

**Fine-tune when:**
- You need a **specific style, tone, or voice** the model can't reliably follow from instructions.
- You need a **rigid output format** (e.g., always a particular JSON shape) with high reliability.
- You have a **narrow, high-volume task** where a long prompt with many examples is redundant overhead — fine-tuning lets you use a short prompt (and often a smaller model), saving cost at scale.
- You've already optimized prompting and RAG and *still* miss your accuracy targets.

**Don't fine-tune when:**
- The task is **knowledge-heavy** or your data changes often — use RAG, which is cheaper to update (retraining every time facts change is painful).
- You **haven't tried prompting hard yet.**
- You need one model to do **many unrelated tasks** — prompt-engineer, or train separate models.

> **Gotcha — fine-tuning learns your data's flaws, including its balance.** The training data's distribution becomes the model's worldview. If 60% of your training examples are refusals but real traffic is 5% refusals, the model will refuse far too often. If 10% of your example answers contain typos, expect typos ~10% of the time. Curate and balance the data deliberately.

> **Gotcha — measure, or you'll pay for a placebo.** Build an eval set *before* you train and hold out 10–20% as a test set. After training, run the same inputs through both the base model and the fine-tuned model and compare on your eval. If you can't measure a difference, you didn't need to fine-tune — the "it feels better" improvement is often imaginary.

The three approaches are complementary, not rivals. Mature systems commonly use all three: prompting shapes behavior, RAG supplies knowledge, and a fine-tuned (often small) model makes a high-volume path cheap.

### Exercise 1 (hands-on, no training required)

1. Write down a real task you'd like an LLM to do better.
2. Answer the deciding question in one sentence: does it need **new facts** or **new behavior**?
3. Draft a **prompting** attempt: a system prompt plus 2–3 few-shot examples.
4. Decide whether RAG is needed (is knowledge missing?) and, if so, name the document set you'd index.
5. Write a two-sentence justification for whether fine-tuning is warranted *yet*, referencing the "fine-tune when / don't fine-tune when" lists. If you conclude "not yet," you've just saved yourself a training run.

### Interview questions — Section 1

**Q1 (conceptual). What is the single most useful question for deciding between fine-tuning and RAG, and how does the answer route you?**
"Do I need new facts or new behavior?" New facts → RAG (or put docs in the prompt), because fine-tuning doesn't reliably teach knowledge and is costly to update. New behavior (style, format, a narrow skill) → fine-tuning, which bakes behavior into the weights.

**Q2 (conceptual). In what order should you try prompting, RAG, and fine-tuning, and why that order?**
Prompting first (fastest, free to iterate, no infra), then RAG if knowledge is missing (accurate and easy to update), then fine-tuning only if behavior/format is still off or the task is narrow/high-volume. The order escalates cost and complexity, so you stop as soon as something works.

**Q3 (practical). A support bot needs to always answer in a strict JSON schema and adopt the company's terse house style. Prompting gets it ~85% right. Is this a fine-tuning candidate? Why?**
Yes — this is behavior/format, not knowledge, and prompting has plateaued below the reliability bar. A rigid format plus a specific voice on a repetitive task is exactly what SFT is good at, and it can shorten the prompt (cost savings at volume).

**Q4 (gotcha). Your fine-tuned model refuses far more often than the base model, though you never intended that. What likely went wrong?**
A data-balance problem: your training set probably over-represented refusals relative to production, so the model learned that refusing is normal. The training distribution becomes the model's behavior — you must balance the dataset to match the behavior you actually want.

**Q5 (gotcha). After fine-tuning, the model "feels" better to you. What must you do before trusting that, and what's the risk if you don't?**
Compare base vs. fine-tuned on a held-out eval set with objective metrics. Without that, "feels better" is often a placebo — you may have spent time and money for no measurable gain (or even a regression you can't see).

---

## Section 2 — LoRA and QLoRA Theory

### The problem: full fine-tuning is too expensive for most people

**Full fine-tuning** means updating *every* weight in the model. For a 7-billion-parameter model, that requires holding the model, its gradients (the signals used to adjust weights), and the optimizer's bookkeeping in **GPU memory** — the fast memory on a graphics card, measured in **VRAM** (Video RAM). Full fine-tuning of even a 7B model needs far more VRAM than a **consumer GPU** (the kind in a gaming PC or free cloud notebook, e.g., 16 GB) has. You'd need expensive datacenter hardware. LoRA and QLoRA exist to make fine-tuning fit on the hardware ordinary people have.

### LoRA: train a tiny add-on instead of the whole model

**LoRA (Low-Rank Adaptation)** is based on a key insight from its 2021 paper: when you adapt a big model to a new task, the *change* to its weights has low "intrinsic rank" — meaning the update, though it spans a huge matrix, can be well-approximated by something much smaller. (A **matrix** is a grid of numbers; a weight layer is a big matrix. **Rank** loosely measures how much independent information a matrix carries — a low-rank matrix can be reconstructed from far fewer numbers.)

Here's the mechanism, kept concrete. Take one weight matrix `W` in the model (say it's `d × d`). Normal fine-tuning would learn a full-size update `ΔW` and set the new weight to `W + ΔW`. LoRA instead:

- **Freezes `W`** — the original weights never change.
- Represents the update as the product of two *skinny* matrices: `ΔW = B · A`, where `A` is `r × d` and `B` is `d × r`, and `r` (the **rank**) is small — often 8, 16, or 32.
- The adapted weight is `W + (α/r) · B·A`, where `α` (**alpha**) is a scaling factor.

Only `A` and `B` are trained. Because `r` is tiny, `A` and `B` together have a few million parameters instead of billions — LoRA can cut trainable parameters by around 10,000×. When you wrap a 3B model, you might train only ~0.1% of its parameters. Two more benefits: the frozen base can be shared across many tasks (each task is just a small set of `A`/`B` **adapters**), and you can swap adapters in and out without touching the base.

> **Gotcha — LoRA shrinks *trainable* memory, not the base model's memory.** The base weights still have to sit in VRAM (usually in 16-bit), and that's the biggest memory cost. So plain LoRA reduces fine-tuning memory only modestly (roughly 3×). To fit a big model on a small GPU you need to shrink the base itself — which is what QLoRA does.

### The hyperparameters that actually matter

- **`r` (rank):** how much capacity the adapter has. Higher `r` = more expressive but more parameters and more overfitting risk. Common values: 8, 16, 32, 64. (**Overfitting** = the model memorizes the training examples instead of learning the general pattern, so it does worse on new data.)
- **`α` (lora_alpha):** the scaling factor; the effective update is scaled by `α/r`. A common heuristic is `α = r` or `α = 2r`. What matters is the *ratio* to `r`.
- **`target_modules`:** which weight matrices get adapters. Typically the attention projection layers (named `q_proj`, `k_proj`, `v_proj`, `o_proj`) and often the feed-forward/MLP layers (`gate_proj`, `up_proj`, `down_proj`). Adapting more modules = more capacity. A useful finding: if you apply LoRA to *all* linear layers, the exact value of `r` matters less.
- **`lora_dropout`:** randomly ignores a fraction of adapter connections during training to reduce overfitting.
- **learning rate:** how big each weight-adjustment step is. LoRA typically uses a higher learning rate than full fine-tuning.

### QLoRA: shrink the base so it fits

**QLoRA (Quantized LoRA)**, from its 2023 paper, adds one idea to LoRA: **quantize** the frozen base model. **Quantization** means storing numbers with fewer bits — here, compressing the base weights from 16-bit down to **4-bit**, cutting their memory footprint ~4×. The adapters are still trained in higher precision (16-bit). This is what lets a 33B model fine-tune on a single 24 GB GPU. QLoRA has three named innovations worth knowing:

1. **4-bit NormalFloat (NF4):** a 4-bit number format designed specifically for data that follows a bell curve (a normal distribution), which trained neural-network weights tend to. NF4 preserves more information than ordinary 4-bit integers for this kind of data.
2. **Double Quantization:** quantizing even the small "constants" used by the quantization process itself, squeezing out a bit more memory.
3. **Paged Optimizers:** a safety mechanism that spills optimizer memory to CPU RAM during sudden spikes, preventing the out-of-memory crashes that otherwise kill long training runs.

During training, the 4-bit base weight for a layer is **de-quantized on the fly** back to 16-bit just for that layer's computation, combined with the adapter output, and the result flows forward. The base is never permanently changed.

> **Gotcha — 4-bit ≠ free lunch.** Quantization introduces small errors. NF4 and double quantization are what keep the quality loss minimal; naive 4-bit quantization plus fine-tuning would degrade results noticeably. In practice QLoRA gets close to 16-bit LoRA quality — one famous QLoRA-trained model reached ~99% of a leading commercial model's score on a benchmark — but "close" is not "identical." If you have the VRAM for plain LoRA in 16-bit, that's the safer default.

> **Gotcha — precision knobs must match your GPU.** Newer GPUs (Ampere generation and later) support **bfloat16** (a 16-bit format with wide range); older ones (like the free-tier T4) use **float16**. Selecting the wrong compute type causes errors or slowdowns. Auto-detection or matching the format to the card avoids this.

### Exercise 2 (paper-and-pencil)

1. For a weight matrix that is 4096 × 4096, count the parameters in a full update (`ΔW`) vs. a LoRA update with `r = 8` (`B` is 4096×8, `A` is 8×4096). Compute the ratio. (You should find LoRA uses a tiny fraction.)
2. Explain in one sentence why LoRA alone doesn't let you fit a 65B model on a 24 GB GPU, but QLoRA does.
3. Match each QLoRA component to the problem it solves: NF4 → ?, Double Quantization → ?, Paged Optimizers → ?
4. Given a task where you suspect the adapter needs more capacity, name two hyperparameters you'd change and in which direction.

### Interview questions — Section 2

**Q1 (conceptual). What is the core insight behind LoRA, and how does it change what gets trained?**
The insight is that the *weight update* during task adaptation has low intrinsic rank, so it can be approximated by two small matrices `B` and `A` (`ΔW = B·A`) instead of a full-size update. The base weights are frozen; only `A` and `B` (a few million parameters) are trained, cutting trainable parameters by orders of magnitude.

**Q2 (conceptual). What does QLoRA add to LoRA, and name its three innovations.**
QLoRA quantizes the frozen base model to 4-bit to shrink its memory footprint (adapters stay in 16-bit). Its three innovations are 4-bit NormalFloat (NF4) quantization, Double Quantization (quantizing the quantization constants), and Paged Optimizers (spilling optimizer memory to avoid out-of-memory spikes).

**Q3 (practical). You have a 16 GB consumer GPU and want to fine-tune a 7B model. Would you choose LoRA or QLoRA, and why?**
QLoRA. Plain LoRA still keeps the 7B base in 16-bit VRAM, which is the dominant memory cost and likely won't fit comfortably in 16 GB with training overhead. QLoRA's 4-bit base makes it fit, at a small, usually acceptable quality cost.

**Q4 (practical). Which LoRA hyperparameters would you tune first for a task where the adapter seems underpowered, and what's the risk of pushing them too far?**
Increase `r` (adapter capacity) and possibly widen `target_modules` to more layers. The risk is overfitting and more memory/compute; a higher `r` also interacts with `α`, since the effective scaling is `α/r`.

**Q5 (gotcha). A teammate says "we quantized to 4-bit, so quality will obviously tank." Why is that not necessarily true for QLoRA?**
Because QLoRA uses NF4 (a format tuned for normally-distributed weights) and double quantization to preserve information, and de-quantizes to 16-bit for actual computation. This keeps quality close to 16-bit LoRA — naive 4-bit quantization would indeed degrade quality, but QLoRA is specifically designed to avoid that.

---

## Section 3 — The PEFT Library

### The problem: you need a clean way to attach adapters to a real model

The theory in Section 2 is useless without tooling that actually inserts the `A`/`B` adapters into a model, freezes the base, and hands you a trainable object. **PEFT (Parameter-Efficient Fine-Tuning)** is Hugging Face's library that does exactly this. (**Hugging Face** is the standard hub and ecosystem for open models and the `transformers` library that loads them.) "Parameter-efficient" is the umbrella term for methods like LoRA that train a small fraction of parameters. PEFT is where the concepts become three lines of code.

### The three-step pattern

Every PEFT + LoRA workflow is the same shape:

1. **Load a base model** with `transformers`.
2. **Define a `LoraConfig`** — the object holding your hyperparameters.
3. **Wrap the model** with `get_peft_model(...)`, which freezes the base and injects trainable adapters.

Here's a minimal, annotated example for a **causal LM** (a text-generation model — "causal" means it predicts the next token from the ones before it):

```python
# pip install torch transformers peft accelerate datasets bitsandbytes
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_id = "Qwen/Qwen2.5-3B-Instruct"

# --- QLoRA: load the base in 4-bit ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # store base weights in 4-bit
    bnb_4bit_quant_type="nf4",       # the NF4 format from the QLoRA paper
    bnb_4bit_compute_dtype=torch.bfloat16,  # de-quantize to bf16 for compute (float16 on older GPUs)
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Prepares a quantized model for training (handles some numerical setup)
model = prepare_model_for_kbit_training(model)

# --- Define the LoRA adapters ---
lora_config = LoraConfig(
    r=16,                 # rank
    lora_alpha=32,        # scaling (effective scale = alpha / r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],  # attention + MLP layers
    lora_dropout=0.05,
    bias="none",          # don't train bias terms
    task_type="CAUSAL_LM",
)

# --- Wrap: freeze base, inject adapters ---
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# e.g. "trainable params: 3,686,400 || all params: 3,089,625,088 || trainable%: 0.12"
```

That `print_trainable_parameters()` line is your sanity check: if the trainable percentage is a fraction of a percent, LoRA is wired up correctly. From here you'd hand `model` to a normal training loop (e.g., Hugging Face's `Trainer` or TRL's `SFTTrainer`), then save just the small adapter with `model.save_pretrained(...)`.

> **Gotcha — wrong `target_modules` names = silent no-op or crash.** The module names differ between model architectures (a T5 model uses `q`/`v`; a Llama/Qwen model uses `q_proj`/`v_proj`, etc.). If you pass names that don't exist, PEFT may attach nothing or error out. Check the model's actual layer names, or use PEFT's option to target all linear layers, rather than copying names from an unrelated tutorial.

> **Gotcha — for 4-bit training, call `prepare_model_for_kbit_training`.** Skipping this step on a quantized model commonly causes training instability or errors. It's a required bit of setup, not optional.

> **Gotcha — `task_type` must match the model.** `CAUSAL_LM` for text generation, `SEQ_2_SEQ_LM` for encoder-decoder models, `SEQ_CLS` for classification. The wrong `task_type` produces a model that trains but behaves incorrectly.

### Exercise 3 (hands-on)

1. In a notebook with a GPU, `pip install` the stack above and load a small causal LM in 4-bit (a 1–3B model keeps it fast).
2. Create a `LoraConfig` and wrap the model with `get_peft_model`.
3. Call `model.print_trainable_parameters()` and record the trainable percentage.
4. Change `r` from 16 to 64 and re-wrap; note how the trainable parameter count changes.
5. Deliberately pass a nonsense entry in `target_modules` (e.g., `"not_a_layer"`) and observe the error or the change in trainable params — this teaches you what a misconfigured target looks like. Then fix it.

### Interview questions — Section 3

**Q1 (conceptual). What does PEFT's `get_peft_model` actually do to the model?**
It freezes the base model's weights and injects trainable low-rank adapter matrices into the layers named in `target_modules`, returning a model where only those adapters are trainable — so a standard training loop updates the adapters, not the base.

**Q2 (practical). You loaded a model in 4-bit and want to fine-tune it. Which extra PEFT step is required and why?**
Call `prepare_model_for_kbit_training(model)` before wrapping with LoRA. It performs numerical setup needed for stable training on a quantized model; skipping it commonly causes instability or errors.

**Q3 (practical). How do you quickly verify LoRA is correctly configured after wrapping?**
Call `model.print_trainable_parameters()`. If the trainable percentage is a small fraction of a percent (with the bulk of parameters frozen), LoRA is wired up correctly; a high trainable percentage suggests the base wasn't frozen or the config is wrong.

**Q4 (gotcha). Your training runs but the model doesn't seem to learn, and trainable params look suspiciously low or zero. What's a likely cause?**
Incorrect `target_modules` names for this architecture, so no adapters (or too few) were attached. Layer names vary by model; verify the actual names or target all linear layers instead of copying names from a different model.

**Q5 (conceptual). What's the difference between `task_type="CAUSAL_LM"` and `"SEQ_2_SEQ_LM"`, and why does it matter?**
`CAUSAL_LM` is for decoder-only text-generation models (predict next token); `SEQ_2_SEQ_LM` is for encoder-decoder models. PEFT uses `task_type` to wire adapters and heads correctly, so a mismatch yields a model that trains but behaves incorrectly for the architecture.

---

## Section 4 — Unsloth Setup for Consumer GPUs

### The problem: PEFT works, but it's slow and memory-hungry on a free GPU

PEFT gives you correct LoRA/QLoRA, but on a free cloud notebook or a modest gaming GPU, training can be slow and can still run out of memory. **Unsloth** is an open-source library that re-implements the training internals to be faster and lighter — roughly **2× faster** with about **30% less VRAM** — while keeping the same LoRA/QLoRA math. It wraps everything behind one class, `FastLanguageModel`, and integrates with the same Hugging Face ecosystem (models, datasets, and trainers). Its target audience is exactly people fine-tuning on a **free Colab** notebook or a single consumer GPU. (**Google Colab** is a free, browser-based notebook service that gives you a GPU for a limited session, commonly an older T4 card.)

### The setup, step by step

Unsloth mirrors the PEFT pattern but folds loading and quantization into one call:

```python
# In Colab: select Runtime -> Change runtime type -> GPU (e.g., T4) first.
# pip install unsloth
from unsloth import FastLanguageModel

max_seq_length = 2048   # context length; 2048 is a good testing default

# --- Load base model, quantized, in one call ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",   # also e.g. Qwen models; "unsloth/..." are pre-quantized
    max_seq_length=max_seq_length,
    dtype=None,             # None = auto-detect (float16 on T4/V100, bfloat16 on newer GPUs)
    load_in_4bit=True,      # QLoRA-style 4-bit; set False for 16-bit LoRA if you have the VRAM
    # token="hf_...",       # needed only for gated models
)

# --- Attach LoRA adapters ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                                # suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,                      # 0 is optimized in Unsloth
    bias="none",                         # "none" is optimized
    use_gradient_checkpointing="unsloth",  # extra memory savings for long context
    random_state=3407,                   # a seed, for reproducibility
)
```

Two knobs worth understanding:

- **`use_gradient_checkpointing`:** **gradient checkpointing** trades a little extra compute for a lot less memory by recomputing some intermediate values during training instead of storing them all. Unsloth's `"unsloth"` mode is an optimized version that enables longer context on the same card.
- **`random_state`:** a fixed **seed** so a run is reproducible; the specific number doesn't matter, only that you keep it fixed if you want repeatable results.

From here you pair the model with a trainer — commonly TRL's `SFTTrainer` — pass your dataset, and call `trainer.train()`. When done, Unsloth can save the adapters locally or push the fine-tuned model to the Hugging Face Hub.

> **Gotcha — forgetting to select a GPU runtime.** On a free notebook, the default runtime may have *no* GPU. If you skip switching the runtime to GPU, Unsloth will either fail or fall back to painfully slow CPU execution. Always set the GPU runtime first.

> **Gotcha — `dtype` vs. your GPU generation.** A free-tier T4 needs float16; newer (Ampere+) cards prefer bfloat16. Setting `dtype=None` lets Unsloth auto-detect, which is the safe choice; hard-coding bfloat16 on a T4 will error.

> **Gotcha — use the pre-quantized `unsloth/...` model names for the fastest path.** Unsloth publishes 4-bit versions of popular models (e.g., `unsloth/Meta-Llama-3.1-8B`). Loading those avoids quantizing on the fly and is the smoothest experience; loading a raw upstream model still works but may be slower to set up.

> **Gotcha — the free session is time-limited and resets.** A free Colab GPU session can disconnect after a period of inactivity or time, wiping your files. Save adapters/checkpoints to persistent storage (or push to the Hub) *before* you lose the session, and keep runs short enough to finish within it.

### Exercise 4 (hands-on) — the capstone

1. Open a free GPU notebook and **set the runtime to GPU** (confirm with a quick check that a GPU is visible).
2. `pip install unsloth` and load a small pre-quantized model (a 1–3B `unsloth/...` model trains fastest) with `load_in_4bit=True` and `dtype=None`.
3. Attach LoRA adapters with `FastLanguageModel.get_peft_model`, using `r=16` and the attention+MLP `target_modules`, and `use_gradient_checkpointing="unsloth"`.
4. Print the trainable parameter count (the same sanity check as with PEFT) and confirm it's a small fraction.
5. Without necessarily running a full training job, save the adapter (`model.save_pretrained("my-adapter")`) to confirm the save path works, and note how you would push it to the Hub. Write two sentences on what you'd do differently to keep a real training run safely within a time-limited free session.

### Interview questions — Section 4

**Q1 (conceptual). What is Unsloth and what does it change relative to using PEFT directly?**
Unsloth is an open-source library that re-implements LoRA/QLoRA training internals to be faster (~2×) and more memory-efficient (~30% less VRAM), while keeping the same math and Hugging Face compatibility. It wraps loading, quantization, and adapter attachment behind the `FastLanguageModel` class, aimed at fine-tuning on consumer GPUs and free notebooks.

**Q2 (practical). Walk through the two `FastLanguageModel` calls needed to set up a QLoRA fine-tune in Unsloth.**
`FastLanguageModel.from_pretrained(...)` loads the (ideally pre-quantized) base with `load_in_4bit=True`, a `max_seq_length`, and `dtype=None` for auto-detection. Then `FastLanguageModel.get_peft_model(...)` attaches LoRA adapters with `r`, `target_modules`, `lora_alpha`, `lora_dropout=0`, `bias="none"`, and gradient checkpointing.

**Q3 (conceptual). What does gradient checkpointing do, and why enable it on a small GPU?**
It reduces memory by not storing all intermediate activations during the forward pass and instead recomputing them during the backward pass — trading a bit of extra compute for substantially less memory, which lets you train longer contexts or larger batches on a limited GPU.

**Q4 (gotcha). Your Unsloth notebook is extremely slow and no training progress bar advances meaningfully. What's the first thing to check?**
Whether a GPU runtime is actually selected. On free notebooks the default may have no GPU, causing a CPU fallback that's dramatically slower; switch the runtime to GPU and reload.

**Q5 (gotcha). You set `dtype="bfloat16"` explicitly and get an error on a free-tier T4. Why, and what's the fix?**
Older GPUs like the T4 support float16, not bfloat16, so hard-coding bfloat16 fails. Set `dtype=None` to let Unsloth auto-detect the correct type for the card (float16 on T4/V100, bfloat16 on Ampere+).

---

## Gotchas summary table

| # | Where | Gotcha | Why it bites | Fix / mindset |
|---|-------|--------|-------------|----------------|
| 1 | When to FT | Fine-tuning to add knowledge | FT doesn't reliably teach facts; costly to update | Use RAG (or prompt) for knowledge |
| 2 | When to FT | Data imbalance shapes behavior | 60% refusals in data → constant refusals | Curate & balance data to match desired behavior |
| 3 | When to FT | Trusting "it feels better" | Often a placebo; may even regress | Build an eval set first; compare base vs FT on held-out data |
| 4 | When to FT | Fine-tuning before trying prompting/RAG | Skips cheaper fixes | Prompt → RAG → fine-tune, in order |
| 5 | LoRA theory | Expecting LoRA to fit huge models on tiny GPUs | Base still lives in 16-bit VRAM | Use QLoRA (4-bit base) for big models / small GPUs |
| 6 | LoRA theory | Ignoring the α/r ratio | Effective scale is α/r, not α alone | Tune r and α together (e.g., α = r or 2r) |
| 7 | QLoRA theory | Assuming 4-bit tanks quality | Naive 4-bit would; QLoRA won't | NF4 + double quant keep quality near 16-bit |
| 8 | QLoRA theory | Wrong compute dtype for GPU | Errors/slowdowns | bf16 on Ampere+, fp16 on T4; auto-detect if unsure |
| 9 | PEFT | Wrong `target_modules` names | Attaches nothing or crashes | Use the architecture's real layer names / target all linear |
| 10 | PEFT | Skipping `prepare_model_for_kbit_training` | Instability/errors on 4-bit models | Always call it before wrapping a quantized model |
| 11 | PEFT | Wrong `task_type` | Model trains but misbehaves | Match: CAUSAL_LM / SEQ_2_SEQ_LM / SEQ_CLS |
| 12 | Unsloth | No GPU runtime selected | Silent CPU fallback, painfully slow | Set runtime to GPU first; confirm a GPU is visible |
| 13 | Unsloth | Hard-coded bfloat16 on old GPU | T4 doesn't support it | `dtype=None` for auto-detection |
| 14 | Unsloth | Free session resets, wiping files | Lose your trained adapter | Save/push adapters before the session ends; keep runs short |
| 15 | All | No sanity check on trainable % | Misconfigured LoRA goes unnoticed | Call `print_trainable_parameters()`; expect a small fraction |

---

## Quick reference card

**When to fine-tune (decide with: facts or behavior?)**
- New **facts** → RAG (or docs in prompt). FT does not reliably teach knowledge.
- New **behavior** (style, format, narrow skill) → fine-tune.
- Order: **prompting → RAG → fine-tuning.** Don't skip steps.
- Fine-tune when: specific style/voice, rigid format, narrow high-volume task (short prompt/smaller model saves cost), or prompting+RAG plateaued.
- Don't when: knowledge-heavy/changing data, many unrelated tasks, or you haven't prompted hard.
- Always build an eval set first; compare base vs FT on held-out data. Balance your training data.

**LoRA** = freeze base `W`, learn `ΔW = B·A` (rank `r` small); adapted weight `W + (α/r)·B·A`. Trains ~0.1% of params. Shrinks *trainable* memory, not the base.
- Key hyperparams: `r` (capacity), `α` (scale via α/r), `target_modules` (attention `q/k/v/o_proj` + MLP `gate/up/down_proj`), `lora_dropout`, learning rate.

**QLoRA** = LoRA + 4-bit quantized frozen base (adapters stay 16-bit). Fits big models on small GPUs.
- Three innovations: **NF4** (4-bit format for bell-curve weights), **Double Quantization** (quantize the constants), **Paged Optimizers** (avoid OOM spikes). De-quantizes to 16-bit for compute.

**PEFT (Hugging Face)** — three steps:
```python
model = AutoModelForCausalLM.from_pretrained(id, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
model = prepare_model_for_kbit_training(model)          # required for 4-bit
cfg = LoraConfig(r=16, lora_alpha=32, target_modules=[...], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, cfg); model.print_trainable_parameters()
```
Install: `torch transformers peft accelerate datasets bitsandbytes`.

**Unsloth** (~2× faster, ~30% less VRAM; consumer GPUs / free Colab):
```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B", max_seq_length=2048, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=[...], lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
```
Select GPU runtime first; `dtype=None` auto-detects (fp16 T4 / bf16 Ampere+); save/push adapters before the session resets.

**Reference material:** OpenAI's fine-tuning guide and Hamel Husain's fine-tuning posts (decision framework); the LoRA paper (arXiv 2106.09685) and QLoRA paper (arXiv 2305.14314); Hugging Face PEFT documentation; the Unsloth GitHub README and Colab notebooks.

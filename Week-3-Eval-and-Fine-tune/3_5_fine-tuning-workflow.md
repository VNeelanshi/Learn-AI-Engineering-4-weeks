# Fine-Tuning End to End: Dataset Prep, Training, Evaluation, and Publishing

A hands-on beginner's guide. No prior knowledge assumed. Every term is defined the first time it appears. Roughly 70% explanation, 30% practice — each section ends with an exercise, and every section closes with five interview questions and answers.

A few definitions to anchor us:

- **LLM (Large Language Model):** a program (like Llama or Qwen) that takes text in and produces text out.
- **Token:** the small chunk of text (a word or word-piece) that a model reads and writes one at a time.
- **Fine-tuning:** continuing to train an already-trained model on your own examples so its behavior shifts to fit your task.
- **SFT (Supervised Fine-Tuning):** the most common kind of fine-tuning, where each training example is a paired input and the *desired* output (the "supervision"), and the model learns to produce that output.

This guide walks the full practical pipeline: turn raw examples into a training-ready dataset, run an actual fine-tune of a small open model, measure whether it actually improved, and publish it so others (or your own applications) can use it.

---

## Section 1 — Dataset Prep: SFT Data, Chat Templates, and the Eval Split

### The problem: models don't read your spreadsheet; they read one exact text format

You might have training examples in a spreadsheet or JSON file. The model can't learn from that directly. During training it sees a single stream of tokens, and — critically — it must see that stream formatted *exactly* the way it will be formatted at inference time (when you actually use it). If you train on one format and serve with another, the model underperforms or produces garbage. So dataset prep is really two jobs: getting your examples into a recognized *data format*, and rendering them through the correct *chat template*.

### The two data formats

Fine-tuning tools (like Hugging Face's `datasets` library and the training frameworks built on it) recognize two shapes:

- **Standard format** — each example is plain text, e.g. a `prompt` string and a `completion` string, or a single `text` field. Used for non-chat tasks.
- **Conversational format** — each example is a list of messages, where each message is an object with a **role** and **content**: `{"role": "user", "content": "..."}`, `{"role": "assistant", "content": "..."}`, and optionally `{"role": "system", "content": "..."}` (a system message is standing instructions). This is the format for chat/instruction models and the one you'll use most.

A conversational example looks like this:

```json
{"messages": [
  {"role": "system", "content": "You are a terse support assistant."},
  {"role": "user", "content": "How do I reset my password?"},
  {"role": "assistant", "content": "Open Settings > Security > Reset password, then follow the email link."}
]}
```

> **Gotcha — the "ShareGPT" format uses different keys.** Some public datasets store conversations with `from`/`value` keys (e.g. `{"from": "human", "value": "..."}`) instead of `role`/`content`. This is the **ShareGPT** format. You must convert it to `role`/`content` before training (helper functions like `standardize_sharegpt` do this). Feeding raw ShareGPT keys to a tool expecting `role`/`content` silently mis-parses your data.

### Chat templates: the exact wording the model expects

A **chat template** is the rule that converts your list of messages into the single token stream the model was designed to read, inserting **special tokens** (reserved marker tokens) that label where each role's turn starts and ends. Different model families use different markers:

- **Llama-3 style:** turns are wrapped like `<|start_header_id|>user<|end_header_id|>\n\n...content...<|eot_id|>` (where `<|eot_id|>` means "end of turn").
- **ChatML / Qwen style:** turns look like `<|im_start|>user\n...content...<|im_end|>`.

You rarely write these by hand. The model's **tokenizer** (the component that converts text to tokens) carries the template, and you apply it with a call like `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`. Training frameworks apply it automatically when you give them a conversational dataset. (`add_generation_prompt=True` appends the opening of the assistant's turn, which is what you want when you later ask the model to *generate* a reply.)

> **Gotcha — base models have NO chat template; instruct models do.** A "base" model is the raw pretrained model; an "instruct" (or "chat") model has already been tuned to follow conversations and ships with a chat template in its tokenizer. If you fine-tune a *base* model, its tokenizer has no template, and you must set one (via a helper such as `get_chat_template(tokenizer, chat_template="llama-3.1")` or `setup_chat_format`) or your conversations won't be formatted at all. Qwen models generally ship with a template; many raw Llama base checkpoints don't.

> **Gotcha — use the SAME template at training and inference.** The single most common cause of a fine-tuned model producing broken output is a template mismatch: trained with one template, served (e.g. in another runtime) with another. Pin the template and reuse it everywhere.

> **Gotcha — align the EOS token with the template.** The **EOS (end-of-sequence) token** tells the model when to stop generating. If the template's turn-ending marker doesn't match the tokenizer's EOS token, the model may never stop generating (runaway output). Make sure they're consistent.

### Train on responses only (mask the prompt)

By default the model computes its learning signal (**loss** — a number measuring how wrong the prediction is) over the *entire* sequence, including the user's question. But you don't want the model learning to generate the *questions* — only the *answers*. **Training on responses only** (also called completion-only or assistant-only loss) masks out the prompt tokens so loss is computed only on the assistant's response. This commonly improves accuracy by a small but real margin.

In practice you enable this either with a framework flag (`assistant_only_loss=True`) or a helper that takes the exact `instruction_part` and `response_part` markers for your template (e.g., for Llama-3, `instruction_part="<|start_header_id|>user<|end_header_id|>\n\n"` and `response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"`).

> **Gotcha — wrong markers = broken masking.** If the `instruction_part`/`response_part` strings don't exactly match your template's tokens, masking silently does the wrong thing — sometimes leaking the answer into the "input," which makes training loss crash toward zero for the wrong reason. If your loss plummets in a handful of steps, suspect the masking markers.

### The eval split: hold out data you never train on

You must reserve some examples to *test* the fine-tuned model, and those examples must **never** be seen during training — otherwise you're just measuring memorization. Splitting a dataset is one call: `dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)` gives you a `train` portion and a `test` portion. `seed` fixes the randomness so the split is reproducible; `shuffle=True` ensures the split isn't biased by ordering.

> **Gotcha — split before any peeking, and keep it fixed.** Decide the split once and keep the test set locked away. If you repeatedly tweak the model while looking at the test set, you'll unconsciously tune to it (a subtle form of **data leakage** — test information bleeding into training decisions), and your reported numbers will be optimistic.

### Exercise 1 (hands-on)

1. Create 20 examples for a narrow task in **conversational format** (`messages` with `system`/`user`/`assistant`). Keep the assistant answers consistent in style — that consistency is what the model will learn.
2. Load a small instruct model's tokenizer and call `apply_chat_template` on one example with `tokenize=False`. Read the rendered text and identify the special tokens marking each role's turn.
3. Split your data with `train_test_split(test_size=0.2, seed=42, shuffle=True)`. Confirm the counts (16 train / 4 test).
4. Write down the exact `instruction_part` and `response_part` markers you'd use to train on responses only for this model's template.
5. Note one thing that would break if you trained a *base* version of this model instead of the instruct version.

### Interview questions — Section 1

**Q1 (conceptual). What is a chat template and why must training and inference use the same one?**
A chat template is the rule that converts a list of role/content messages into the exact token stream (with special role-marker tokens) the model expects. If training and inference use different templates, the model sees an unfamiliar format at serving time and its output degrades or breaks, so the template must be pinned and reused everywhere.

**Q2 (conceptual). Difference between standard and conversational dataset formats?**
Standard format is plain text (e.g., a `prompt` and `completion` string or a single `text` field), used for non-chat tasks. Conversational format is a list of messages, each with a `role` (system/user/assistant) and `content`, used for chat/instruction tuning; frameworks auto-apply the chat template to it.

**Q3 (practical). Why train on responses only, and how do you enable it?**
Because you want the model to learn to produce good answers, not to generate the user's questions — masking the prompt tokens so loss is computed only on the assistant response improves accuracy. Enable it via a flag like `assistant_only_loss=True` or a helper given the exact `instruction_part`/`response_part` markers for your template.

**Q4 (gotcha). Your training loss drops to near zero within a few steps. What's a likely dataset-prep cause?**
Incorrect response-only masking markers, so the assistant's answer leaked into the "input" the model conditions on — it's effectively copying the answer, making loss collapse for the wrong reason. Verify the `instruction_part`/`response_part` strings exactly match the template's tokens.

**Q5 (gotcha). You fine-tuned a base (not instruct) model and it never formats conversations correctly. What did you miss?**
Base models don't ship with a chat template, so you must set one (e.g., via `get_chat_template`/`setup_chat_format`) before training. Without it, your conversational data isn't rendered into the expected role-marked format at all.

---

## Section 2 — Running the Fine-Tune

### The problem: turning a prepared dataset into an actually-trained model

With data ready, you need to load a model efficiently, attach a trainable component, configure the training run, and launch it — without running out of memory on the GPU you have. (**GPU** = graphics processor that does the heavy math; its memory, **VRAM**, is the binding constraint.) We'll use two ideas that make this feasible on a modest GPU: **LoRA** and a fast training library.

**LoRA (Low-Rank Adaptation)** is a technique where you *freeze* the model's billions of original weights and train only small "adapter" matrices inserted into some layers. (**Weights** are the numbers that define the model.) You end up training well under 1% of the parameters, which slashes memory and time while keeping quality close to full fine-tuning. Loading the frozen model in **4-bit** (a compressed number format that uses a quarter of the memory of the usual 16-bit) makes even a 7–8B model fit on a consumer GPU; LoRA on top of a 4-bit base is often called **QLoRA**.

### Loading the model and attaching LoRA

Using a fast fine-tuning library (which wraps model loading and adapter setup behind a `FastLanguageModel` class and speeds training up considerably), the setup is two calls:

```python
from unsloth import FastLanguageModel

max_seq_length = 2048  # max tokens per example; longer = more memory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",  # or a Qwen instruct model
    max_seq_length = max_seq_length,
    dtype = None,          # auto-detect precision for your GPU
    load_in_4bit = True,   # QLoRA-style 4-bit base to save memory
)

model = FastLanguageModel.get_peft_model(   # freeze base, add trainable LoRA adapters
    model,
    r = 16,                                 # adapter capacity (rank)
    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # extra memory savings
    random_state = 3407,
)
```

### Configuring and launching training

You pass the model, tokenizer, and dataset to a **trainer** (here, the SFT trainer from the TRL library), with a configuration object holding the **hyperparameters** — the knobs you set before training. The ones that matter most:

- **`per_device_train_batch_size`** — how many examples the GPU processes at once. Higher is faster but uses more memory.
- **`gradient_accumulation_steps`** — accumulate gradients over this many mini-batches before updating, to *simulate* a larger batch without the memory cost. Effective batch size = batch_size × accumulation.
- **`learning_rate`** — how big each weight-update step is; LoRA typically uses ~2e-4, higher than full fine-tuning.
- **`num_train_epochs`** (or **`max_steps`**) — one **epoch** is one full pass over the training data. For small datasets, 1–3 epochs is common.
- **`warmup_steps`** — ramp the learning rate up slowly at the start to stabilize training.
- **`logging_steps`**, **`save_steps`**, **`eval_strategy`** — how often to log metrics, checkpoint, and evaluate.
- **`optim`** — the optimizer; a memory-light choice like `adamw_8bit` is common on small GPUs.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,   # effective batch size = 8
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
        save_strategy = "steps",
        save_steps = 50,                    # checkpoint every 50 steps
        seed = 3407,
    ),
)

# Optional but recommended: train only on the assistant responses
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.train()
```

### Reading the training signal

While it runs, watch the **training loss**: it should generally trend *down*. A loss that stays flat suggests the learning rate is too low or the data isn't teaching anything new; a loss that spikes wildly suggests the learning rate is too high.

> **Gotcha — a loss that crashes to ~0 almost immediately is a red flag, not a triumph.** It usually means the model can "see" the answer it's supposed to predict (broken response masking) or the data is trivially repetitive — i.e., it's memorizing, not learning. Investigate rather than celebrate.

> **Gotcha — overfitting from too many epochs.** **Overfitting** is when the model memorizes the training examples instead of the general pattern, so it does great on training data and worse on new data. On a small dataset, too many epochs causes this. Watch the *eval* loss (on your held-out set): when training loss keeps falling but eval loss starts rising, you've gone too far.

> **Gotcha — checkpoint, because free/cloud sessions die.** Set `save_steps` so training writes intermediate checkpoints to disk. If your session disconnects (common on free notebooks), you can resume from the last checkpoint instead of starting over.

### Exercise 2 (hands-on)

1. Load a small instruct model (a 1–3B model trains fastest) in 4-bit and attach LoRA adapters as shown.
2. Wire up an `SFTTrainer` with your `train_dataset` and `eval_dataset` from Section 1, using a small `num_train_epochs` (1–2) and `save_steps` for checkpointing.
3. Enable train-on-responses-only with the correct markers for your model.
4. Run `trainer.train()` and watch the loss. Record the starting and ending training loss.
5. Note whether the loss decreased steadily, plateaued, or crashed suspiciously fast — and what each would imply.

### Interview questions — Section 2

**Q1 (conceptual). What do LoRA and 4-bit loading each contribute to making fine-tuning fit on a small GPU?**
LoRA freezes the base weights and trains only tiny adapter matrices (well under 1% of parameters), cutting the memory and compute of *training*. Loading the frozen base in 4-bit compresses the model's own weights to a quarter of their 16-bit size, which is the dominant memory cost — together (QLoRA) they let large models train on consumer GPUs.

**Q2 (conceptual). What is gradient accumulation and why use it?**
It accumulates gradients over several mini-batches before performing one weight update, simulating a larger effective batch size (batch_size × accumulation_steps) without the memory cost of a truly large batch — useful when your GPU can't fit the batch size you want.

**Q3 (practical). Your training loss is essentially flat across the run. What would you check or change?**
Likely the learning rate is too low or the data isn't teaching the model anything new. Try raising the learning rate (LoRA often uses ~2e-4), verify the data actually contains the target behavior, confirm the adapters are trainable, and check that the chat template/masking are correct.

**Q4 (gotcha). How do you detect overfitting during a fine-tuning run, and what causes it on small datasets?**
Watch eval loss on the held-out set alongside training loss: when training loss keeps dropping but eval loss starts rising, the model is overfitting. On small datasets it's typically caused by too many epochs (or too high a rank/learning rate); reduce epochs or add regularization.

**Q5 (gotcha). Why configure `save_steps`, and what does it protect against?**
It writes checkpoints periodically so you can resume from the latest one if the session crashes or disconnects (common on free/cloud notebooks), rather than losing all progress and restarting training from scratch.

---

## Section 3 — Evaluating Base vs. Fine-Tuned

### The problem: "it feels better" is not evidence

After training, the tempting move is to type a couple of prompts, nod approvingly, and ship. This is how people waste money on fine-tunes that changed nothing — a fine-tune can *feel* better while performing identically (or worse) on real measurements. The only honest way to know whether fine-tuning helped is to run the **same eval suite** on both the original **base** model and your **fine-tuned** model, on the **held-out test set** you never trained on, and compare the numbers.

(An **eval suite** here means a fixed set of test inputs plus an automated way to score each output — for example, exact-match or a rule for structured outputs, a similarity check against a reference answer, or an **LLM-as-a-judge**: using another model to grade outputs against a rubric.)

### The comparison procedure

1. **Freeze the test set and the scoring method.** Same inputs, same grader, for both models — otherwise the comparison is meaningless.
2. **Run the base model** on every test input, score the outputs, record the headline metric(s).
3. **Run the fine-tuned model** on the same inputs with the *same chat template*, score, record.
4. **Compare**, and report not just accuracy but also whatever else matters for your use case — often **latency** (how long a response takes) and **cost** (a fine-tuned smaller model can be cheaper per call at volume).

For faster inference when scoring, many libraries have an inference mode (e.g., `FastLanguageModel.for_inference(model)`) that speeds up generation. Use it for the eval pass.

> **Gotcha — evaluate with the exact serving template.** If you score the fine-tuned model with a different chat template than it was trained on, you'll measure a broken configuration and wrongly conclude the fine-tune failed. Match the template used in training.

> **Gotcha — a single metric hides regressions.** Fine-tuning for one behavior can quietly break another (this is related to **catastrophic forgetting** — a model losing previously-good behavior after training on narrow new data). Include some general-capability checks in your eval, not only your target task, so you notice if the model got better at your task but worse elsewhere.

> **Gotcha — small test sets give noisy verdicts.** With only a handful of test items, a 2-point difference could be pure chance. Prefer enough examples to trust the gap, and be honest that a tiny difference on a tiny test set isn't proof of anything.

### Interpreting the result

There are three honest outcomes: the fine-tune clearly beats the base on your metric (ship it, noting any regressions), it's within noise of the base (you probably didn't need to fine-tune — save the cost), or it's worse (revisit the data, hyperparameters, or template). All three are useful findings; only "I didn't measure" is a failure.

### Exercise 3 (hands-on)

1. Take your held-out test set from Section 1 and define one clear scoring rule (exact match, keyword presence, similarity to a reference, or a judge prompt).
2. Load the **base** model (before your LoRA adapters) and run it over the test set; record the score, plus rough latency.
3. Load your **fine-tuned** model (base + adapters, same chat template) and run it over the same test set; record the score and latency.
4. Add 3–5 *general* prompts unrelated to your task and check the fine-tuned model didn't obviously regress on them.
5. Write a 3–4 sentence verdict: did fine-tuning help on the target metric, at what latency/cost trade-off, and did you spot any regression? Cite your numbers.

### Interview questions — Section 3

**Q1 (conceptual). Why is comparing base vs. fine-tuned on a held-out set the essential evaluation step?**
Because it's the only way to know whether fine-tuning actually improved anything: subjective "it feels better" impressions are often placebos, and a held-out set (never seen in training) measures generalization rather than memorization. Same inputs and same grader for both models make the comparison valid.

**Q2 (practical). Beyond accuracy, what else should you compare between the two models and why?**
Latency and cost per call, since a key reason to fine-tune a smaller model is cheaper, faster inference at volume — a small accuracy gain may or may not justify the trade-offs, and a fine-tuned small model that matches a big one while being cheaper is often the real win.

**Q3 (practical). You fine-tuned for a narrow task. Why include general-capability prompts in your eval?**
To catch catastrophic forgetting — improving the target task can degrade unrelated abilities. General prompts reveal whether the model got better at your task while getting worse elsewhere, which a task-only eval would miss.

**Q4 (gotcha). Your fine-tuned model scores terribly on the eval, but generated fine when you chatted with it manually. What's a likely cause?**
A chat-template mismatch during evaluation — scoring with a different template than the model was trained/served with produces broken outputs. Re-run the eval using the exact template used in training and manual chat.

**Q5 (gotcha). The fine-tuned model scores 2 points higher than the base on a 20-item test set. Can you claim it's better?**
Not confidently — on such a small test set a 2-point gap is within noise and could be chance. You'd want a larger test set (or a confidence estimate) before claiming a real improvement.

---

## Section 4 — Publishing to the Hugging Face Hub with a Model Card

### The problem: a trained model on your disk helps no one

Once you've trained and validated a model, it needs to be *shareable and reusable* — by your applications, your teammates, or the community. The **Hugging Face Hub** is the standard place to host open models: it stores the files, versions them, and serves them so anyone can load your model by name. Publishing well means two things: uploading the right *artifact*, and writing a *model card* that documents it.

### Authenticate first

You need a Hugging Face account and an **access token** (a secret key that authorizes uploads). You provide it by logging in (`huggingface-cli login`, or `login(token=...)` in code, or a `HF_TOKEN` environment variable).

> **Gotcha — never hard-code or commit your token.** Treat it like a password: use an environment variable or the login prompt, not a string pasted into a script you might share. A leaked write-token lets anyone modify your Hub repositories.

### What to upload: adapter vs. merged model

LoRA training produces small **adapter** files (the trained matrices) that sit on top of the base model. You have a few choices for what to publish:

- **Adapter only.** Tiny (megabytes). Users load it *on top of* the base model. Efficient, but users need the base too. Save with `model.save_pretrained(...)` / `model.push_to_hub("user/repo")`.
- **Merged 16-bit model.** The adapters are fused into the base weights, producing a standalone full model in 16-bit precision that anyone can load directly. Common for general sharing: `model.push_to_hub_merged("user/repo", tokenizer, save_method="merged_16bit", token=...)`.
- **GGUF / quantized.** A compressed single-file format for running locally in lightweight runtimes. Produced with a GGUF save method and a quantization choice (e.g. `q4_k_m`, `q8_0`).

> **Gotcha — always push the tokenizer too.** The tokenizer (with its chat template) is required to use the model correctly. If you upload only the model weights, others can't reproduce the right input formatting. Push both to the same repo (`tokenizer.push_to_hub("user/repo")`), or use a merged-push method that includes it.

> **Gotcha — merging to 4-bit is usually the wrong choice for general use.** Merging adapters directly into a 4-bit model is discouraged unless you have a specific downstream need, because it bakes in quantization loss. For sharing, prefer merged 16-bit (or publish the adapter and let users quantize as they wish).

### The model card

A **model card** is the documentation for your model — the `README.md` file in the repository. It has two parts:

1. **YAML metadata** (a block of `key: value` fields at the very top, between `---` lines) that the Hub reads to categorize and enable features. Common fields: `license`, `base_model`, `language`, `tags`, `datasets`, `library_name`.
2. **Human-readable sections** describing the model.

A minimal model card:

```markdown
---
license: apache-2.0
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
language:
  - en
tags:
  - lora
  - fine-tuned
datasets:
  - your-username/your-dataset
library_name: peft
---

# My Support-Reply Model

A LoRA fine-tune of Llama-3.1-8B-Instruct for generating terse customer-support replies.

## Intended use and limitations
Intended for drafting short support replies in English. Not for medical, legal, or
factual-lookup use. May reflect biases and errors present in the training data.

## Training data
Fine-tuned on ~2,000 anonymized support conversations in conversational format.

## Training procedure
QLoRA (4-bit base), rank r=16, alpha=16, learning rate 2e-4, 2 epochs,
trained on assistant responses only.

## Evaluation
On a held-out test set of N examples, exact-format compliance rose from X% (base)
to Y% (fine-tuned); general-capability spot checks showed no obvious regression.
```

> **Gotcha — an accurate model card is part of responsible release, not paperwork.** Document the **intended use**, **limitations**, and **known biases** honestly. People will run your model on inputs you never imagined; the card is how you set expectations and warn about failure modes. Also state the base model's and data's **licenses** — you can't relicense a model more permissively than its base or training data allow.

> **Gotcha — check the base model's license and gating before publishing.** Some base models have usage restrictions or require accepting terms. Publishing a derivative can inherit those constraints; confirm you're allowed to redistribute before you push.

### Exercise 4 (hands-on) — the capstone

1. Create a Hugging Face account and generate a **write** access token; log in via `huggingface-cli login` (don't paste the token into code).
2. Save your fine-tuned model — start with the **adapter only** (`model.save_pretrained("my-adapter")`) and confirm the files appear locally, then push it: `model.push_to_hub("your-username/your-model")` and `tokenizer.push_to_hub("your-username/your-model")`.
3. (Optional) Also push a **merged 16-bit** version so others can load it standalone.
4. Write a `README.md` model card with the YAML metadata block (license, base_model, tags, datasets) and the four sections: description, intended use & limitations, training data, and **evaluation results** (paste the base-vs-fine-tuned numbers you produced earlier).
5. Visit your model's Hub page, confirm the card renders, and load the model back by its repo name to verify it works end to end.

### Interview questions — Section 4

**Q1 (conceptual). What's the difference between publishing a LoRA adapter and a merged 16-bit model, and when would you choose each?**
An adapter is tiny (just the trained matrices) but requires users to also load the base model; a merged 16-bit model fuses adapters into the base for a standalone, directly-loadable model. Publish the adapter for efficiency when users have the base; publish merged 16-bit for easy general sharing.

**Q2 (practical). You uploaded your model weights but users say outputs are malformed. What common artifact did you likely forget?**
The tokenizer (with its chat template). Without it, users can't reproduce the exact input formatting the model expects, so outputs break. Push the tokenizer to the same repo alongside the model.

**Q3 (conceptual). What goes in a model card and why does it matter?**
YAML metadata (license, base_model, tags, datasets) plus human-readable sections: description, intended use, limitations, known biases, training data, training procedure, and evaluation results. It sets honest expectations for users who'll run the model on unforeseen inputs and documents licensing constraints — part of a responsible release.

**Q4 (gotcha). Why check the base model's license before publishing your fine-tune?**
Because your derivative can inherit the base model's (and training data's) usage restrictions or gating; you can't relicense more permissively than they allow, and some bases require accepting terms or restrict redistribution. Confirm you're permitted to publish before pushing.

**Q5 (gotcha). Why is merging adapters into a 4-bit model generally discouraged for public release?**
Because it bakes the quantization error permanently into the shared weights, degrading quality for downstream users. For general sharing, prefer merged 16-bit (or publish the adapter and let users choose their own quantization).

---

## Gotchas summary table

| # | Where | Gotcha | Why it bites | Fix / mindset |
|---|-------|--------|-------------|----------------|
| 1 | Data format | Raw ShareGPT `from`/`value` keys | Mis-parsed by tools expecting `role`/`content` | Convert first (e.g. `standardize_sharegpt`) |
| 2 | Chat template | Different template at train vs. inference | Broken/garbage output | Pin one template, reuse everywhere |
| 3 | Chat template | Base model has no template | Conversations never formatted | Set one (`get_chat_template`/`setup_chat_format`) |
| 4 | Chat template | EOS token not aligned with template | Model never stops generating | Ensure turn-end marker matches EOS |
| 5 | Response masking | Wrong instruction/response markers | Answer leaks into input; loss crashes | Match markers exactly to the template |
| 6 | Eval split | Peeking at / tuning to the test set | Data leakage, optimistic numbers | Split once, lock the test set away |
| 7 | Training | Loss crashes to ~0 immediately | Memorization / broken masking, not learning | Investigate masking & data, don't celebrate |
| 8 | Training | Too many epochs on small data | Overfitting | Watch eval loss; stop when it rises |
| 9 | Training | No checkpoints | Session dies, progress lost | Set `save_steps`; resume from checkpoint |
| 10 | Evaluation | "It feels better" | Placebo; may not have improved | Compare base vs FT on held-out set with a fixed grader |
| 11 | Evaluation | Eval with wrong template | Measures a broken config | Evaluate with the exact serving template |
| 12 | Evaluation | Single-metric eval | Hides catastrophic forgetting | Include general-capability checks |
| 13 | Evaluation | Tiny test set | Noise mistaken for improvement | Use enough examples; don't over-read small gaps |
| 14 | Publishing | Hard-coded/leaked token | Anyone can modify your repos | Use env var / login prompt; never commit |
| 15 | Publishing | Forgot to push tokenizer | Users can't format inputs correctly | Push tokenizer to same repo |
| 16 | Publishing | Merge to 4-bit for sharing | Bakes in quantization loss | Prefer merged 16-bit or publish the adapter |
| 17 | Publishing | Ignoring base model license | May not be allowed to redistribute | Check license/gating before pushing |

---

## Quick reference card

**Dataset prep**
- Formats: **standard** (text / prompt+completion) vs **conversational** (`messages` = list of `{role, content}`).
- ShareGPT `from`/`value` → convert to `role`/`content`.
- **Chat template** renders messages → token stream with special role markers (Llama-3: `<|start_header_id|>…<|eot_id|>`; ChatML/Qwen: `<|im_start|>…<|im_end|>`). Apply with `tokenizer.apply_chat_template(...)`. Base models need one set; instruct models ship one. Same template at train + inference. Align EOS.
- **Train on responses only** (mask the prompt) via `assistant_only_loss=True` or `train_on_responses_only(trainer, instruction_part=..., response_part=...)`.
- **Eval split**: `dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)`. Lock the test set.

**Run the fine-tune** (LoRA/QLoRA + fast trainer)
```python
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct", max_seq_length=2048, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=[...], lora_alpha=16, lora_dropout=0, use_gradient_checkpointing="unsloth")
trainer = SFTTrainer(model, tokenizer, train_dataset, eval_dataset, args=SFTConfig(per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=2, learning_rate=2e-4, optim="adamw_8bit", save_steps=50, output_dir="outputs"))
trainer.train()
```
Key knobs: batch size × grad-accum = effective batch; lr ~2e-4 for LoRA; 1–3 epochs on small data. Loss should trend down (not crash). Checkpoint. Watch eval loss for overfitting.

**Evaluate base vs fine-tuned**
Same held-out test set + same grader + same chat template for both. Compare accuracy, **latency**, **cost**. Include general prompts to catch **catastrophic forgetting**. Small gaps on small sets = noise. Three honest outcomes: better (ship), tied (didn't need FT), worse (fix data/params/template).

**Publish to the Hub**
- Auth: `huggingface-cli login` / `HF_TOKEN` (never commit the token).
- Artifact: **adapter only** (tiny, needs base) / **merged 16-bit** (standalone, best for sharing) / **GGUF** (local runtimes). Always push the **tokenizer** too.
- **Model card** = `README.md` with YAML metadata (`license`, `base_model`, `tags`, `datasets`) + sections: description, intended use & limitations, training data, procedure, **evaluation results**. Check the base model's license/gating.

**Reference material:** Hugging Face `datasets` documentation (formats, chat templates, `train_test_split`); the Unsloth fine-tuning notebooks and TRL `SFTTrainer` docs (training run, response-only training, saving/merging); the Hugging Face Hub documentation (uploading models, model cards).

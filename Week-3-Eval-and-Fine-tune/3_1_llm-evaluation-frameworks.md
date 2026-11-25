# Evaluating LLM Applications: Evals, RAGAS, DeepEval, and the Tooling Landscape

A hands-on beginner's guide. No prior knowledge assumed. Every term is defined the first time it appears. Roughly 70% explanation, 30% practice — each section ends with an exercise, and every section closes with five interview questions and answers.

Some quick definitions we'll lean on throughout:

- **LLM (Large Language Model):** a program (like GPT, Claude, or Llama) that takes text in and produces text out. It is *probabilistic*, meaning the same input can produce different outputs on different runs.
- **Prompt:** the text instructions you send to an LLM.
- **Eval (evaluation):** a repeatable, automated test that measures whether an LLM's output is good, according to criteria *you* define.

---

## Section 1 — Why Evals?

### The problem: the "just tweak the prompt" instinct

When people first build something with an LLM, the loop usually looks like this: write a prompt, try a few inputs, eyeball the outputs, decide it "looks good," ship it. Then a week later something breaks — maybe half the answers are suddenly too formal, or the model starts making things up — and nobody noticed because there was no way to notice.

This is the trap. The instinct to fix problems by *tweaking the prompt and re-checking by hand* feels productive but doesn't scale. You can hold three or four test cases in your head. You cannot hold three hundred. And because an LLM is probabilistic, "I ran it once and it looked fine" tells you almost nothing about how it behaves across the range of real inputs.

Practitioner Hamel Husain, in his widely-cited essay *Your AI Product Needs Evals*, puts it bluntly: he argues that unsuccessful LLM products almost always share one root cause — a failure to build a solid evaluation system. His core claim is that, like ordinary software engineering, success with AI depends on **how fast you can iterate**, and you can only iterate fast if you can *measure* whether each change made things better or worse.

> **Why this matters before we touch any tool:** Every framework later in this guide (RAGAS, DeepEval, Promptfoo) exists to solve this one problem — turning "it looks fine" into a number you can trust and re-check automatically.

### What an eval actually is

An **eval** is a test suite for LLM output. It has three parts:

1. **A dataset** — a collection of example inputs (and often the ideal answer for each).
2. **A way to run your system** on each input to produce an output.
3. **A scoring method** — a rule or judge that decides whether each output is good.

The tricky part is step 3. With normal code, you check `output == expected` — an exact match. LLM output can't be checked that way, because the same prompt produces different wording every time. So instead of exact matches, evals check *properties*: Does the answer contain the right fact? Is it grounded in the source? Is it the right tone? Is it free of made-up information?

### Eval-driven development and the three levels

Hamel describes three **levels** of evaluation, ordered from cheap-and-fast to expensive-and-slow:

- **Level 1 — Unit tests (assertions).** These are simple, code-based checks, exactly like the assertions you'd write in a normal test. ("The output must be valid JSON." "The output must not contain a phone number." "The output must mention the customer's order number.") An **assertion** is just a statement that must be true or the test fails. These are cheap, so you run them on *every* change.
- **Level 2 — Model-based evaluation (LLM-as-a-judge).** Here you use *another* LLM to grade the output. **LLM-as-a-judge** means asking a language model, "Given this question and this answer, is the answer faithful/relevant/correct?" This handles fuzzy qualities that assertions can't, but it costs money (extra API calls) and must itself be validated, so you run it on a set cadence rather than every commit.
- **Level 3 — Human evaluation.** A person reviews outputs directly. This is the most trustworthy and the most expensive, so you reserve it for significant product changes.

The cost ordering is Level 3 > Level 2 > Level 1, and that ordering dictates how often you run each. A healthy workflow runs Level 1 constantly, Level 2 regularly, and Level 3 occasionally.

> **Gotcha — don't skip to Level 2.** Beginners love LLM-as-a-judge because it feels sophisticated. But a judge is itself "a messy stochastic thing evaluating a messy stochastic thing" (Hamel's phrase). Nail down as many cheap Level 1 assertions as you can *first*; they're deterministic, free, and catch the most embarrassing failures.

### The flywheel, and "look at your data"

Hamel's central metaphor is a **flywheel**: a heavy wheel that's hard to start turning but, once moving, keeps its momentum. A good eval system is that wheel. Once it exists, every change gets measured instantly, so you can experiment fearlessly and improve quickly.

His single most repeated piece of advice: **remove all friction from looking at your data, and look at a lot of it.** The formal name for this is **error analysis** — you read through real outputs, categorize the failures ("wrong tone," "hallucinated a fact," "ignored the question"), count how often each happens, and *then* write evals targeting the most common failures. You do not start by buying a fancy tool; you start by reading outputs.

Two more of his takeaways worth internalizing early:

- Don't rely only on generic, off-the-shelf metrics — build evals specific to *your* problem.
- If you're passing 100% of your evals, your tests are probably too easy. Good eval sets include hard and adversarial cases.

> **Gotcha — the judge is a "hack" to make you look at data.** Even Hamel says the real value isn't the automated judge itself; it's that building one *forces* you to examine your outputs carefully. If you automate everything and never read a single output, you've missed the point.

### Exercise 1 (hands-on)

You don't need any tool for this — just a notepad or spreadsheet.

1. Pick any LLM task you care about (e.g., "summarize a customer email," "answer a question about a product").
2. Run 10 different realistic inputs through an LLM (any chat interface works) and paste the 10 outputs into a spreadsheet, one row each.
3. Read all 10. In a new column, tag each output with a short **failure category** if something is wrong (e.g., `too-long`, `wrong-tone`, `made-up-fact`, `ignored-question`) or `ok` if it's fine.
4. Count how many times each failure category appears.
5. Write down, in plain English, **three Level 1 assertions** you could turn into automated checks (e.g., "output must be under 100 words," "output must not invent a price").

You've just done error analysis and drafted your first eval spec — the foundation everything else builds on.

### Interview questions — Section 1

**Q1 (conceptual). Why can't you evaluate an LLM's output with a simple `output == expected` check like normal software?**
Because LLM output is probabilistic: the same prompt yields different wording each run, so an exact-match check would fail even on perfectly good answers. Evals instead check *properties* of the output (correct facts, right tone, grounded in source) rather than exact string equality.

**Q2 (conceptual). Explain the three levels of LLM evaluation and how their cost affects how often you run each.**
Level 1 is code-based assertions (cheap, deterministic) — run on every change. Level 2 is model-based / LLM-as-a-judge (costs API calls, must be validated) — run on a regular cadence. Level 3 is human review (most trustworthy, most expensive) — run occasionally, e.g., after big changes. Cost rises from Level 1 to Level 3, so frequency falls in the same direction.

**Q3 (practical). A teammate wants to improve quality by "just tweaking the prompt until it looks better." What's the risk, and what would you propose instead?**
Manual eyeballing doesn't scale and misses regressions because outputs vary run-to-run and you can't hold hundreds of cases in your head. Propose building an eval set: collect representative inputs, define pass/fail criteria (starting with cheap Level 1 assertions), and measure every prompt change against it so you know whether a change actually helped.

**Q4 (gotcha). Your LLM-as-a-judge says your system is 95% correct. Why should you not trust that number yet?**
An LLM judge is itself a fallible, probabilistic system. Its score is meaningless until you've measured its *agreement with human judgments* on the same examples. Without that calibration, you don't know whether "95%" reflects reality or the judge's own biases.

**Q5 (gotcha). Your eval suite passes 100%. Why might that be a bad sign?**
It usually means the tests are too easy and aren't challenging the system. Real eval sets should include hard and adversarial cases; a perfect score suggests you're not probing the failure modes that actually occur in production.

---

## Section 2 — RAGAS: Evaluating Retrieval-Augmented Generation

### First, what is RAG?

**RAG (Retrieval-Augmented Generation)** is a common pattern for building question-answering systems. Instead of relying only on what the LLM memorized during training, RAG does two steps:

1. **Retrieval** — search a collection of your own documents for chunks relevant to the user's question. (This search is usually done with **embeddings**: numeric representations of text where similar meanings sit close together, so "refund policy" can match a passage about returns even without identical words.)
2. **Generation** — feed those retrieved chunks, plus the question, to an LLM and have it write an answer grounded in that text.

RAG has two things that can go wrong, and they're independent:

- The **retriever** might fetch the wrong or incomplete documents.
- The **generator** might write a bad answer even from good documents (e.g., make up facts, or ignore the retrieved text).

### The problem RAGAS solves

If your RAG system gives a bad answer, *which half failed*? RAGAS (short for "Retrieval-Augmented Generation Assessment") is a framework whose whole purpose is to score both halves separately, so you can tell a retrieval problem apart from a generation problem. Crucially, most of its metrics are **reference-free**, meaning they can score your system *without* you having to hand-write the "correct" answer for every question — the framework uses an LLM judge to reason about the pieces you already have.

To compute RAGAS metrics you generally collect four things per interaction: the **question**, the **retrieved context** (the chunks your retriever returned), the **generated answer**, and — for one metric only — a **ground truth** (the reference answer a human considers correct).

### The four core metrics

**1. Faithfulness — "Did the answer stick to the source?"**
Measures whether every claim in the generated answer can actually be traced back to the retrieved context. It breaks the answer into individual statements and checks how many are supported by the context. This is your **hallucination detector**. (**Hallucination** = the model stating something that isn't supported by the source, often confidently.) Score runs 0 to 1; higher is better. Notably, faithfulness does *not* need a ground-truth answer.

**2. Answer relevancy — "Did the answer address the question?"**
Measures how directly the answer responds to the question, penalizing incomplete or off-topic replies. (In current RAGAS versions this metric is also called **Response Relevancy**; the older name `answer_relevancy` still works.) It does not need a ground truth either.

**3. Context precision — "Was the retrieved material on-target and well-ranked?"**
Of the chunks the retriever returned, how many are actually relevant, and are the relevant ones ranked near the top? High precision means little junk was retrieved. This is a *retriever* metric.

**4. Context recall — "Did the retriever fetch everything it needed?"**
Of the information required to fully answer the question, how much did the retriever actually pull in? This is the one core metric that **requires a ground-truth reference answer**: RAGAS breaks the reference answer into statements and checks how many are supported by the retrieved context.

> **Gotcha — precision and recall pull against each other.** You can make context precision look great by retrieving fewer, tighter chunks — but that often *hurts* context recall because you drop information you needed. Never optimize one in isolation; watch all four together.

A useful diagnostic pattern: if **faithfulness is high but answer relevancy is low**, your model is being honest but the retrieved context isn't actually helping it answer the real question — a retrieval problem wearing a generation-problem costume.

### How the metrics are actually computed (and why it matters for cost)

Under the hood, RAGAS metrics call an LLM to do the judging — for example, to decide whether a given statement is supported by the context. That has two consequences you must plan for:

- **It costs money and time.** Every metric on every example is one or more LLM calls. Evaluating thousands of rows can get expensive.
- **Your judge model and embedding model choices move the scores.** Different judge models will disagree at the margins, so keep them fixed when comparing runs.

> **Gotcha — NaN scores.** If an answer or reference decomposes into *zero* statements (e.g., an empty or one-word output), a metric can come back as **NaN** ("not a number") rather than 0. Filter or handle these rows explicitly, or your averages will silently break.

### Hands-on: a minimal RAGAS evaluation

The exact import paths and helper names in RAGAS change between versions, so treat this as the *shape* of the code and confirm against the current RAGAS docs before running.

```python
# pip install ragas datasets
# Set your judge model's key, e.g. export OPENAI_API_KEY=...

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# One row = one question your RAG system answered.
data = {
    "question": ["What is the refund window for online orders?"],
    # The chunks your retriever returned (a list of strings per question):
    "contexts": [["All online orders can be refunded within 30 days of purchase."]],
    # What your RAG system generated:
    "answer": ["You can get a refund within 30 days of buying online."],
    # Ground truth is only needed for context_recall:
    "ground_truth": ["Online orders are refundable within 30 days."],
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)  # prints a score (0-1) per metric
```

> **Gotcha — column names are strict.** RAGAS expects specific field names (like `contexts` as a *list of strings*, not a single string). Passing a plain string where a list is expected, or misnaming a column, is the most common beginner error and produces confusing failures.

### Exercise 2 (hands-on)

1. Take the task from Exercise 1 and turn it into (or imagine) a tiny RAG setup: for each of 3 questions, gather the retrieved context chunks, the generated answer, and write a one-sentence ground-truth answer.
2. Build the dataset dict exactly like the example above (mind the list-of-strings for `contexts`).
3. Run `evaluate` with all four metrics.
4. For each question, write one sentence interpreting the scores: is any failure a *retrieval* problem (low context precision/recall) or a *generation* problem (low faithfulness/answer relevancy)?

### Interview questions — Section 2

**Q1 (conceptual). What are the two independent failure points in a RAG system, and which RAGAS metrics map to each?**
Retrieval (fetching the right documents) and generation (writing a good answer from them). Context precision and context recall measure the *retriever*; faithfulness and answer relevancy measure the *generator*.

**Q2 (conceptual). Explain faithfulness vs. answer relevancy in one sentence each.**
Faithfulness measures whether the answer's claims are actually supported by the retrieved context (a hallucination check). Answer relevancy measures whether the answer actually addresses the user's question.

**Q3 (practical). Which single RAGAS metric requires a human-written ground-truth answer, and why?**
Context recall. To know whether the retriever fetched *everything needed*, RAGAS decomposes the reference (ground-truth) answer into statements and checks how many are covered by the retrieved context — so it needs that reference to compare against.

**Q4 (gotcha). Your context precision looks excellent but recall is poor. What likely happened, and why is chasing precision alone dangerous?**
You're probably retrieving very few, tightly-targeted chunks — great precision, but you're dropping context you needed, which tanks recall. Optimizing precision in isolation degrades the system's ability to answer completely; the four metrics must be read together.

**Q5 (gotcha). A RAGAS run returns NaN for some rows' faithfulness. What's a common cause and how do you handle it?**
The answer (or reference) decomposed into zero statements — often an empty, trivial, or malformed output — so the metric is undefined rather than 0. Handle it by filtering or explicitly treating those rows, so they don't silently corrupt your averaged scores.

---

## Section 3 — DeepEval: Unit-Test-Style Evals

### The problem DeepEval solves

Suppose you already write normal software tests. You'd like your LLM checks to live *right next to* your code tests, run with the same command, and fail the build the same way when quality drops. RAGAS gives you RAG metrics but isn't built around that testing workflow. DeepEval is: it's an open-source Python framework often described as "Pytest for LLMs."

Two terms to lock in first:

- **Pytest:** the most popular Python testing tool. You write functions named `test_...`, put `assert` statements inside, and run them from the command line; any failed assert fails the test.
- **Unit test:** a small, automated check of one specific behavior.

DeepEval lets you write `test_...` functions where, instead of `assert x == y`, you assert that an LLM output *passes a metric* (like faithfulness or hallucination) above a threshold you set.

### The core building blocks

**`LLMTestCase`** — an object holding the pieces of one example. The common fields:

- `input` — the user's query.
- `actual_output` — what your LLM system actually produced.
- `expected_output` — the ideal answer (optional; needed by some metrics).
- `retrieval_context` — for RAG, the chunks retrieved (used by RAG metrics).
- `context` — background facts used by the hallucination metric as ground truth.

**Metrics** — DeepEval ships dozens. Each returns a **score from 0 to 1** plus a natural-language *explanation* of the score, and each has a `threshold` (default often 0.5) that decides pass/fail. Common ones:

- `AnswerRelevancyMetric` — does the output address the input?
- `FaithfulnessMetric` — is the output supported by the `retrieval_context`? (RAG hallucination check.)
- `HallucinationMetric` — does the output contradict the provided `context`?
- `SummarizationMetric` — for summaries; combines an **alignment score** (does the summary avoid contradicting/hallucinating from the source?) and a **coverage score** (does it capture the necessary information?).
- `GEval` — a flexible, custom metric described below.

**`assert_test(test_case, [metrics])`** — runs the metrics against the test case and raises a failure if any metric scores below its threshold.

> **Gotcha — most metrics call an LLM judge, so you need a key.** By default DeepEval uses an OpenAI model as the judge, so you must set an `OPENAI_API_KEY` (or configure a different judge). This means metric scores cost API calls and can vary slightly between runs. You *can* point DeepEval at a different or local judge model.

### G-Eval: writing your own criterion in plain English

**G-Eval** is a metric inside DeepEval that lets you define *any* custom quality bar in plain English and have an LLM judge score against it, using step-by-step ("chain-of-thought") reasoning. (**Chain-of-thought** = prompting the judge to reason through steps before scoring, which improves reliability.) You give it either a `criteria` string (and it invents the evaluation steps) or explicit `evaluation_steps` (more controllable), plus which fields it should look at.

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
# Note: newer versions expose these as `SingleTurnParams`. Check your version.

correctness = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct given the expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.5,
)
```

> **Gotcha — every param you reference must exist on the test case.** If your metric's `evaluation_params` includes `EXPECTED_OUTPUT`, then every `LLMTestCase` you feed it must actually set `expected_output`, or it will error. Match them exactly.

> **Gotcha — the class name for params moved.** Older DeepEval used `LLMTestCaseParams`; current versions use `SingleTurnParams`. If an import fails, this rename is a likely culprit — check the version you installed.

### Hands-on: a summarization + hallucination test

```python
# pip install deepeval
# export OPENAI_API_KEY=...   (the default judge)
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric, HallucinationMetric

SOURCE = (
    "Rice is a staple food in Bengal. It is grown in flooded fields and "
    "harvested in autumn. Most households eat it at least twice a day."
)

def test_summary():
    tc = LLMTestCase(
        input=SOURCE,
        # In a real test this comes from YOUR system, not hand-written:
        actual_output="Rice is a staple in Bengal, grown in flooded fields and eaten daily.",
        # HallucinationMetric compares the output against these facts:
        context=[SOURCE],
    )
    assert_test(tc, [
        SummarizationMetric(threshold=0.5),
        HallucinationMetric(threshold=0.5),
    ])
```

Run it with DeepEval's own runner, which gives richer LLM-specific reporting:

```bash
deepeval test run test_summary.py
# add -n 4 to run tests in parallel across 4 workers
```

You can also run the same file with plain `pytest`, because DeepEval is built on Pytest — the same exit codes let any CI system treat a failed eval exactly like a failed unit test. (**CI**, continuous integration, is the automated system that runs your tests on every code change.)

> **Gotcha — `HallucinationMetric` direction.** This metric expects the ground-truth facts in the `context` field and flags output that *contradicts* them. Putting the source text in the wrong field (e.g., `retrieval_context`, which the RAG `FaithfulnessMetric` uses instead) is a common mix-up. Read each metric's required fields before wiring it up.

### Exercise 3 (hands-on)

1. Install DeepEval and set a judge key.
2. Write a `test_...` file with **two** `LLMTestCase`s for a summarization task: one good summary and one deliberately bad one that invents a fact not in the source.
3. Add both `SummarizationMetric` and `HallucinationMetric`.
4. Run `deepeval test run`. Confirm the good case passes and the invented-fact case fails, and read the natural-language explanation DeepEval prints for the failure.
5. Bonus: add a `GEval` metric with a custom criterion like "the summary must be under 25 words" and see how the judge scores it.

### Interview questions — Section 3

**Q1 (conceptual). What does it mean to call DeepEval "Pytest for LLMs"?**
It brings unit-testing structure to LLM output: you write `test_...` functions, but instead of asserting exact equality you assert that outputs pass metrics (faithfulness, hallucination, etc.) above a threshold. It runs via its own `deepeval test run` command or plain `pytest`, and produces CI-friendly pass/fail exit codes.

**Q2 (conceptual). What is G-Eval and when would you reach for it instead of a built-in metric?**
G-Eval is a metric that uses an LLM judge with chain-of-thought reasoning to score outputs against *any* custom criterion you describe in plain English. You use it when the quality you care about is subjective or product-specific and no built-in metric captures it (e.g., "matches our brand voice," "explains the refund steps in order").

**Q3 (practical). The `SummarizationMetric` gave a low score. Its two components are alignment and coverage — how do you use them to diagnose the problem?**
A low *alignment* score means the summary contradicts or hallucinates beyond the source (a faithfulness problem). A low *coverage* score means the summary omitted necessary information (a completeness problem). Checking which component dropped tells you whether to fix accuracy or thoroughness.

**Q4 (gotcha). Your `GEval` metric references `EXPECTED_OUTPUT` but tests keep erroring. What's the likely cause?**
Every field named in a metric's `evaluation_params` must be present on each `LLMTestCase`. If the metric expects `EXPECTED_OUTPUT` but a test case doesn't set `expected_output`, it errors. (Also check whether your version renamed `LLMTestCaseParams` to `SingleTurnParams`.)

**Q5 (gotcha). Why do two runs of the exact same DeepEval suite sometimes give slightly different scores, and what should you keep constant to compare fairly?**
Most metrics use an LLM-as-a-judge, which is probabilistic, so scores can drift run to run. To compare fairly, hold the judge model fixed (and ideally the dataset), and look at trends across many examples rather than single-example scores.

---

## Section 4 — Promptfoo and the Broader Landscape

### The problem: comparing prompts and models systematically

You've written a prompt. Should you change the wording? Switch models? You could try each variation by hand — but that's the exact eyeballing trap from earlier. What you want is a **matrix**: run every prompt variation against every test input across every model, and see, side by side, which combination performs best. That's what Promptfoo is built for, and it's often the fastest way for a beginner to get a real eval running.

### Promptfoo: config-driven prompt testing

**Promptfoo** is an open-source command-line tool for testing and comparing prompts, models, and RAG systems. (A **command-line tool** runs from your terminal by typing commands rather than through a graphical interface. Promptfoo was acquired by OpenAI in early 2026 but remains open source under the permissive MIT license.)

Two ideas define it:

- **Declarative config.** You describe your whole eval in a single file, `promptfooconfig.yaml`. (**YAML** is a plain-text format for configuration — indentation and `key: value` pairs, no programming required.) You version-control that file next to your code.
- **Assertions.** For each test you list checks the output must pass. These range from cheap deterministic ones (`contains`, `not-contains`, `is-json`, `latency`) to `llm-rubric`, which uses an LLM-as-a-judge to grade against a description you write.

A **provider** in Promptfoo is a model API you're testing against, e.g. `openai:gpt-4o` or `anthropic:messages:claude-sonnet-4-5`. Listing several providers is how you get the side-by-side model comparison.

Here's a minimal config that tests one prompt across two models with a mix of assertion types:

```yaml
# promptfooconfig.yaml
prompts:
  - "Write a {{tone}} reply to this customer message: {{message}}"

providers:
  - openai:gpt-4o
  - anthropic:messages:claude-sonnet-4-5

tests:
  - vars:
      tone: casual
      message: "Hey, is my order shipping today?"
    assert:
      - type: not-contains
        value: "Dear"          # casual tone shouldn't open formally
      - type: llm-rubric
        value: "The reply is friendly and directly answers whether the order ships today."
  - vars:
      tone: formal
      message: "I would like an update on my order."
    assert:
      - type: llm-rubric
        value: "The reply is polite, professional, and gives an order status update."
```

Run it, then open the results viewer:

```bash
npm install -g promptfoo      # requires a recent Node.js (v20.20+ or v22+)
promptfoo eval                # runs the config in the current folder
promptfoo view                # opens a side-by-side results table in your browser
```

`{{tone}}` and `{{message}}` are **variables** — placeholders that Promptfoo fills in from each test's `vars`, so one prompt template gets reused across many inputs.

> **Gotcha — caching can hide changes.** Promptfoo caches model responses to disk by default (roughly a two-week window) so re-runs are free and fast. But if you change something *outside* the prompt/inputs (say, a model-side update) you may keep seeing stale cached answers. Use `--no-cache` when you need genuinely fresh responses.

> **Gotcha — it runs locally with no shared history.** Results live on your machine or in your CI run; there's no built-in central dashboard of past runs. That's fine solo, but a second teammate asking "where's last week's result?" is the moment teams start reaching for a hosted platform.

### The rest of the landscape (so the names don't surprise you)

You'll hear these three constantly. Here's the honest one-line version of each:

- **Promptfoo** — open-source CLI, YAML configs, strong at side-by-side prompt/model comparison, CI integration, and **red teaming** (deliberately attacking your app to find vulnerabilities like prompt injection or leaked personal data). Best first tool for a solo developer; local-first.
- **Inspect AI** — an open-source Python framework for LLM evals from the UK AI Safety Institute (a government body). It's organized around three concepts: a **task** (the dataset + what to measure), **solvers** (the steps that produce an answer), and **scorers** (how outputs are graded). It's Python-first and strong for rigorous capability and safety testing, rather than quick YAML prompt comparisons.
- **Braintrust** — a commercial (paid, hosted) *platform*, not just a library. You define experiments, datasets, and scorers in a TypeScript or Python SDK, and it gives you dashboards, saved history, side-by-side run diffs, and team collaboration. Teams reach for it once local, one-off evals stop being enough and they need shared, persistent tracking connected to production.

> **Gotcha — "framework" vs. "platform."** Promptfoo and Inspect AI are frameworks that run on your machine and store results locally; Braintrust is a hosted platform that stores and tracks results centrally (with a paid tier). Confusing the two leads to wrong expectations — don't expect a persistent team dashboard from a local CLI.

A common mature setup pairs *two* tools: a lightweight framework for gating changes in CI (Promptfoo, DeepEval, or RAGAS) plus a platform for history, human annotation, and stakeholder dashboards (like Braintrust). You do not need both on day one.

### How to choose (rule of thumb)

- **Testing/comparing prompts and models fast, solo, from the terminal** → Promptfoo.
- **RAG-specific retrieval + generation metrics** → RAGAS (Section 2).
- **Python, pytest-style, want it in your existing test suite / CI** → DeepEval (Section 3).
- **Rigorous, code-first capability or safety evals in Python** → Inspect AI.
- **Team needs shared dashboards, saved history, production tracking** → a platform like Braintrust.

### Exercise 4 (hands-on)

1. Install Promptfoo and run `promptfoo init` to scaffold a starter config.
2. Adapt the example config above to your own task from Exercise 1. Include **at least two providers** (two models) and **at least three test cases**.
3. Use a mix of assertion types: at least one deterministic (`contains`, `not-contains`, or `is-json`) and at least one `llm-rubric`.
4. Run `promptfoo eval` then `promptfoo view`, and read the side-by-side table. Note which model/prompt combination wins and *why* (which assertions each failed).
5. Reflection (write 3–4 sentences): for *your* task, which of the five tools in the "How to choose" list would you actually adopt, and what would make you add a second one?

### Interview questions — Section 4

**Q1 (conceptual). What makes Promptfoo's approach "declarative," and what's the advantage?**
You describe the entire eval — prompts, providers, test cases, assertions — in a single `promptfooconfig.yaml` file rather than writing procedural code. The advantage is that the eval lives in version control next to your code, is easy to diff and review, and runs identically locally and in CI.

**Q2 (practical). You need to decide between GPT-4o and Claude for a support-reply feature. How would you use Promptfoo to make that call with data instead of intuition?**
List both as providers, write a set of representative test inputs with assertions capturing what "good" means (tone, correctness, latency, an `llm-rubric` for quality), run `promptfoo eval`, and read the side-by-side matrix to see which model passes more assertions across the inputs — turning the choice into measured comparison.

**Q3 (conceptual). Distinguish a framework like Promptfoo/Inspect AI from a platform like Braintrust.**
Frameworks are libraries/CLIs that run on your machine or CI and store results locally, with no built-in shared history. A platform like Braintrust is hosted (and paid), storing results centrally with dashboards, run-to-run diffs, saved datasets, and team collaboration — used when local one-off evals no longer suffice.

**Q4 (gotcha). You changed something and re-ran `promptfoo eval`, but the outputs look identical to before. What's the likely cause and the fix?**
Promptfoo caches model responses to disk by default, so it may be serving stale cached answers if the change didn't alter the prompt or inputs it keys on. Re-run with `--no-cache` to force fresh model calls.

**Q5 (practical). A solo developer with no budget wants to start evaluating LLM output today. Which tool would you recommend first and why?**
Promptfoo: it's free, open source, runs locally, and gets you from zero to a working side-by-side eval faster than the alternatives, with no platform account or infrastructure required. You can layer on DeepEval/RAGAS or a hosted platform later as needs grow.

---

## Gotchas summary table

| # | Where | Gotcha | Why it bites | Fix / mindset |
|---|-------|--------|-------------|----------------|
| 1 | Evals generally | Jumping straight to LLM-as-a-judge | A judge is itself probabilistic and unvalidated | Nail cheap deterministic assertions first; validate the judge against humans |
| 2 | Evals generally | Automating everything, never reading outputs | You miss the failure patterns that error analysis reveals | Remove all friction from looking at data; read lots of it |
| 3 | Evals generally | Passing 100% of evals | Tests are too easy | Add hard and adversarial cases |
| 4 | RAGAS | Optimizing context precision alone | Recall drops as you retrieve fewer chunks | Read all four metrics together |
| 5 | RAGAS | NaN metric scores | Empty/trivial output → zero statements → undefined | Filter or explicitly handle those rows |
| 6 | RAGAS | Wrong column shape (`contexts` must be list of strings) | Confusing errors, silent misbehavior | Match RAGAS's exact expected field names/types |
| 7 | RAGAS / all judges | Changing judge or embedding model between runs | Scores shift, comparisons become invalid | Keep judge/embedding models fixed when comparing |
| 8 | DeepEval | Metric references a field the test case lacks | Errors at runtime | Every `evaluation_params` field must exist on the `LLMTestCase` |
| 9 | DeepEval | `LLMTestCaseParams` import fails | Renamed to `SingleTurnParams` in newer versions | Check installed version's API |
| 10 | DeepEval | Facts in the wrong field (`context` vs `retrieval_context`) | Metric measures the wrong thing | Read each metric's required fields |
| 11 | Promptfoo | Stale cached responses | Default disk cache (~2 weeks) hides changes | Use `--no-cache` for fresh runs |
| 12 | Promptfoo / Inspect | Expecting a shared team dashboard | They're local-first frameworks | Use a hosted platform for persistent shared history |

---

## Quick reference card

**The core idea**
- Eval = dataset + a way to run your system + a scoring method. Replaces "it looks fine" with a trustworthy number.
- Three levels: (1) assertions — cheap, every change; (2) LLM-as-a-judge — costs calls, on a cadence; (3) human — trustworthy, occasional.
- Build a flywheel: look at your data, do error analysis, target the most common failures, re-check every change.

**RAGAS — for RAG systems (splits retrieval vs. generation)**
- Retriever: **context precision** (relevant chunks, well-ranked) + **context recall** (fetched everything needed — *needs ground truth*).
- Generator: **faithfulness** (claims supported by context = hallucination check) + **answer relevancy** (addresses the question).
- Reference-free except context recall. Metrics use an LLM judge → costs calls, model choice matters.

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

**DeepEval — "Pytest for LLMs"**
- `LLMTestCase(input, actual_output, expected_output?, retrieval_context?, context?)`.
- Metrics score 0–1 with a `threshold`: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `HallucinationMetric`, `SummarizationMetric` (alignment + coverage), `GEval` (custom criterion in plain English).
- `assert_test(test_case, [metrics])`; run with `deepeval test run file.py` (`-n 4` for parallel) or plain `pytest`.
- Default judge = OpenAI → set `OPENAI_API_KEY`.

**Promptfoo — fast prompt/model comparison from the terminal**
- Everything in `promptfooconfig.yaml`: `prompts`, `providers`, `tests` (with `vars` + `assert`).
- Assertions: `contains`, `not-contains`, `is-json`, `latency`, `llm-rubric` (judge).
- `promptfoo eval` then `promptfoo view`. Caches by default → `--no-cache` for fresh.

**Landscape one-liners**
- **Promptfoo** — OSS CLI, YAML, side-by-side + red teaming. Best first tool. Local.
- **Inspect AI** — OSS Python (UK AI Safety Institute); task / solvers / scorers; rigorous, code-first.
- **Braintrust** — paid hosted platform; SDK experiments, dashboards, saved history, team + production tracking.

**Choosing:** prompts/models fast → Promptfoo · RAG metrics → RAGAS · pytest-style in CI → DeepEval · rigorous Python capability/safety → Inspect AI · shared dashboards/history → a platform (Braintrust). Mature teams pair a CI framework with a platform.

**Reference material:** Hamel Husain, *Your AI Product Needs Evals*; RAGAS docs; DeepEval docs; the Promptfoo, Inspect AI, and Braintrust project homepages.

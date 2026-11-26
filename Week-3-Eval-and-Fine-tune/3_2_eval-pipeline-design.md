# Designing an LLM Eval Pipeline: Golden Sets, Pairwise Comparison, Regression Testing, and Building a Suite

A hands-on beginner's guide. No prior knowledge assumed. Every term is defined the first time it appears. Roughly 70% explanation, 30% practice — each section ends with an exercise, and every section closes with five interview questions and answers.

A few definitions we'll rely on throughout:

- **LLM (Large Language Model):** a program (like GPT, Claude, or Llama) that takes text in and produces text out. It is *probabilistic* — the same input can produce different outputs on different runs.
- **Prompt:** the text instructions you send to an LLM.
- **Eval (evaluation):** a repeatable, automated test that measures whether an LLM's output is good, according to criteria *you* define. Because output varies run-to-run, an eval checks *properties* of the output (right fact, right tone, grounded in source) rather than exact string equality.

This guide is about the *pipeline* around evals: how to assemble the dataset, run it repeatedly, compare versions, and stop quality from silently degrading over time.

---

## Section 1 — Designing an Eval Pipeline

### The problem: evals are only useful if the pipeline around them is disciplined

A single eval score tells you almost nothing on its own. What makes evals valuable is running them *the same way, on the same data, every time something changes*, so you can compare "before" and "after" honestly. That requires three things most beginners skip: a curated dataset, a repeatable runner, and version control so you know exactly which dataset produced which score. Without these, you get numbers you can't trust or reproduce.

Practitioner Eugene Yan distilled the whole workflow into three steps that are worth memorizing:

1. **Label a small dataset.** Sample real inputs and outputs from your system and label whether each output is acceptable.
2. **Align your evaluators.** If you're using an automated grader, check that it agrees with your human labels before trusting it.
3. **Run an eval harness on every config change.** A **harness** is just the code that takes your dataset, runs the graders over it (in parallel, within rate limits), and aggregates the results into scores.

We'll cover step 2's alignment in more depth in later material; this section focuses on steps 1 and 3 — the dataset and the harness — plus how to version them.

### The golden set: your source of truth

A **golden set** (also called a golden dataset or eval set) is a curated collection of inputs paired with either their ideal outputs or the criteria for a good output. It's the fixed yardstick you measure every change against. Two properties make it "golden":

- It's **curated**, not random — you deliberately include your most important cases *and* known tricky/edge cases.
- It's **stable** — you don't quietly change it between runs, because if the ruler changes, the measurements aren't comparable.

Eugene Yan's advice for starting one: don't overthink it. Begin with a plain spreadsheet with columns for `input`, `output`, any `metadata` that helps you judge, and a `label` column. Keep labels **binary** — pass/fail or win/lose — rather than a 1–5 scale, because "what makes something a 3 vs a 4?" is a question nobody can answer consistently. Ten to twenty high-priority examples is a perfectly good day-one dataset; you grow it over time as you discover new failure modes.

> **Gotcha — don't start by buying a generic tool or metric.** A recurring mistake is reaching for off-the-shelf metrics before you've looked at your own data. The failures that matter are specific to *your* product, and you only find them by reading real outputs (called **error analysis**: reading outputs, categorizing the failures, counting them, and targeting the most common ones). Start with a spreadsheet and your eyes.

### Ground truth vs. criteria

For some tasks you can write the single correct answer — this is called the **ground truth** (or reference/expected output). For others (e.g., "write a friendly reply"), there's no one right answer, so instead you store *criteria* ("must be friendly, must answer the question, under 80 words"). A golden set can mix both styles. Knowing which type each row is matters, because it decides how you'll score it later — exact/fuzzy match against ground truth vs. a rubric-based judgment against criteria.

### Version control for evals

**Version control** is a system (Git is the standard) that tracks changes to files over time, so you can see what changed, when, and roll back if needed. You already version your code; you should version your evals the same way, and for the same reason: reproducibility.

Concretely, keep these in your repository, as plain-text files (CSV, JSON, or JSONL — **JSONL** is "JSON Lines," one JSON object per line, convenient for datasets):

- **The golden set itself.** When you add or change examples, that's a tracked commit. Now any past score can be tied to the exact dataset that produced it.
- **The eval config / harness code.** The prompts under test, the grader definitions, the thresholds.
- **The results**, or at least a summary. Storing run outputs lets you diff two runs later.

> **Gotcha — a moving dataset invalidates your history.** If you silently edit golden examples while also changing your prompt, and the score goes up, you can't tell whether your prompt improved or you just made the test easier. Treat dataset changes as their own deliberate, reviewed commits — separate from prompt/model changes — so each score maps to a known dataset version.

> **Gotcha — don't commit secrets.** Eval configs often reference API keys. Keep keys in environment variables or a secrets manager, never hard-coded in a file you commit. This is the single most common security slip when people first put evals in a repo.

### The harness: run it the same way every time

The harness is the automation that ties it together. A good one:

- Accepts the golden set as input (a file of input/expected pairs).
- Runs your system on each input to produce outputs.
- Runs the graders in parallel — but respects **rate limits** (the cap on how many API calls per minute a provider allows), or your run will error out mid-way.
- Aggregates into a small number of headline scores you actually look at.

Crucially, the same harness should be runnable with one command *both* on your laptop and in automation, so local results and automated results are directly comparable.

### Exercise 1 (hands-on)

1. Pick an LLM task you care about. Create a spreadsheet (or CSV) with columns: `id`, `input`, `output`, `metadata`, `label`.
2. Collect **15 real input/output pairs** — a mix of common cases and at least 4 deliberately tricky/edge cases.
3. Label each `label` column strictly **pass** or **fail** (no numeric scores). For any you're unsure about, write a one-line note in `metadata` explaining the ambiguity.
4. For each row, mark in a new column whether the correct scoring style is `ground-truth` (there's one right answer) or `criteria` (judged against a rubric).
5. Save the file, initialize a Git repository (`git init`), and commit it (`git add`, `git commit -m "initial golden set v1"`). You now have a versioned golden set.

### Interview questions — Section 1

**Q1 (conceptual). What is a golden set, and what two properties make it trustworthy?**
A golden set is a curated collection of inputs paired with ideal outputs or acceptance criteria, used as the fixed yardstick for every change. It's trustworthy when it's *curated* (deliberately covers important and edge cases, not random) and *stable* (unchanged between runs, so scores are comparable).

**Q2 (conceptual). Walk through the three-step eval workflow.**
(1) Label a small dataset — sample real inputs/outputs and mark each pass/fail. (2) Align your evaluator — verify an automated grader agrees with human labels before trusting it. (3) Run an eval harness on every config change — automated code that runs graders over the dataset and aggregates scores, so each change is measured.

**Q3 (practical). Why use binary pass/fail labels instead of a 1–5 scale when starting out?**
Because numeric scales are uncalibrated and subjective — different people (and the same person on different days) disagree on what separates a 3 from a 4. Binary labels are far more consistent and reproducible, which is what you need for a reliable yardstick early on.

**Q4 (gotcha). Your prompt change shows a higher score, but you also added a few examples to the golden set in the same commit. Why can't you trust the improvement?**
Because two variables changed at once: the prompt and the dataset. A higher score could mean the prompt improved *or* that the new examples happened to be easier. Dataset changes must be separate, reviewed commits so each score maps to a known dataset version.

**Q5 (gotcha). A colleague wants to start by installing a popular eval library and using its default metrics. What's the risk?**
Generic, off-the-shelf metrics rarely capture the failure modes specific to your product, and reaching for them first lets you skip actually looking at your data. The higher-value first step is error analysis on real outputs; build product-specific checks from what you find.

---

## Section 2 — Pairwise Comparison and Win Rate

### The problem: absolute scores are wobbly, but "which is better?" is easy

Suppose you have two versions of a system — an old prompt and a new one — and you want to know which is better. One option is **single-answer grading**: ask a judge to assign each answer an absolute score, then compare the averages. The trouble is that absolute scores are unstable; a judge's sense of "7 out of 10" drifts, and small wording changes to the judge prompt shift every score.

Humans (and LLM judges) are far more reliable at a different question: *"Given these two answers, which is better?"* This is **pairwise comparison** — you show a judge one question and two candidate answers, and it picks A, picks B, or declares a tie. Relative judgments are more stable than absolute ones because the judge only has to compare, not calibrate to an abstract scale.

This methodology was popularized by two well-known evaluation efforts, MT-Bench and Chatbot Arena (from the group behind the Vicuna model). Their core finding is encouraging: a strong LLM judge can reach over 80% agreement with human preferences — roughly the same rate at which two *humans* agree with each other. That's why LLM-as-a-judge pairwise evaluation caught on: it approximates expensive human preference at a fraction of the cost. (**LLM-as-a-judge** simply means using one language model to grade the outputs of another.)

### Win rate: turning many comparisons into one number

Once you can compare two answers, you can score a whole system with **win rate**: the probability that this system's answer is preferred over an opponent's on a given input. Run your comparison across the golden set, count wins, and win rate = wins / (total comparisons). If you're comparing more than two systems, **average win rate** is a system's win rate averaged across all the opponents it faced.

You'll see win rate reported two ways: *including ties* (a tie counts as half a win, or as its own bucket) or *excluding ties* (only decisive comparisons count). State which you're using — the numbers differ, and mixing them makes comparisons meaningless.

> **Gotcha — pairwise doesn't scale to many systems.** Comparing every pair among N systems means roughly N² comparisons — fine for 2 or 3 systems, painful for 20. For a big roster, people use rating systems like **Elo** (borrowed from chess: each system has a rating that goes up when it beats higher-rated opponents), which infers a ranking from a sample of pairwise matchups instead of every pair.

### The three biases you must design around

The same MT-Bench work that validated LLM judges also catalogued their systematic biases. These are the ones an interviewer will expect you to know:

- **Position bias.** The judge tends to favor whichever answer is shown *first* (or in a fixed slot), regardless of quality. This is the big one for pairwise.
- **Verbosity bias** (also called length bias). The judge tends to favor longer, more elaborate answers even when a shorter one is clearer or more correct.
- **Self-enhancement bias** (also called self-preference). A judge tends to favor answers produced by *itself* or its own model family. In the original study, GPT-4 favored its own answers by about 10 percentage points and Claude by about 25 — though the effect isn't universal.

### How to mitigate each

- **Position bias → swap and re-run.** Present the pair as (A, B) *and* as (B, A). If the judge flips its verdict when you swap the order, that vote is position-driven; count such inconsistent votes as ties or discard them. **Consistency** — the percentage of pairs where the judge gives the same verdict after swapping — is a good health metric for your judge.
- **Verbosity bias → control for length.** Add explicit instructions to ignore length, and watch whether your "winning" version is just consistently longer. If it is, be suspicious.
- **Self-preference → don't let a model judge itself.** Use a judge from a *different* model family than the systems under test (e.g., have one family judge another). Even better, use an **ensemble** of several different judges and aggregate their votes (majority vote, or a weighted rating), which dilutes any single model's bias.

> **Gotcha — a judge scoring "95% win rate" is meaningless until you've measured its agreement with humans.** An LLM judge is itself a fallible, probabilistic system. Before trusting its verdicts, validate them against a batch of human-labeled comparisons (the alignment step from Section 1). Agreement in the low-80s% or higher is the bar the MT-Bench work set.

### Exercise 2 (hands-on)

You'll build a tiny pairwise harness by hand (pseudocode/real code both fine).

1. Take 8 inputs from your Section 1 golden set. Produce **two** candidate answers for each (e.g., from two different prompts or two models). Call them System A and System B.
2. For each input, ask an LLM judge: *"Here is a question and two answers. Reply only with A, B, or TIE for which answer better satisfies [your criteria]."* Record the verdict.
3. **Swap the order** and ask again. If the verdict flips, mark that input `inconsistent`.
4. Compute: System A's win rate excluding ties and inconsistent votes; and the judge's **consistency** (fraction of the 8 that gave the same verdict both ways).
5. Write two sentences: what does the consistency number tell you about how much to trust this judge, and which system won?

### Interview questions — Section 2

**Q1 (conceptual). Why is pairwise comparison often more reliable than single-answer grading?**
Because relative judgments ("which is better?") are more stable than absolute ones. Absolute scores drift with the judge's internal calibration and are sensitive to small prompt wording changes, whereas comparing two concrete answers only requires the judge to pick, not to anchor to an abstract scale.

**Q2 (conceptual). Define win rate and average win rate.**
Win rate is the probability that a system's answer is preferred over an opponent's on a given input (wins ÷ comparisons). Average win rate is a system's win rate averaged across all the opponents it was compared against.

**Q3 (practical). You need to rank 15 different prompt variants by quality. Why is pure pairwise a poor fit, and what's the alternative?**
Comparing all pairs is ~N² comparisons, which explodes for 15 variants and gets expensive/slow. Use a rating system like Elo that infers a full ranking from a *sample* of pairwise matchups rather than exhaustively comparing every pair.

**Q4 (gotcha). Your new prompt "wins" 70% of pairwise comparisons, but you always put its answer first. What bias are you likely seeing, and how do you check?**
Position bias — judges tend to favor the first-shown answer. Check by swapping the order and re-running each comparison; if verdicts flip on swap, the wins were position-driven. Measure consistency (same verdict after swapping) and discard or tie the inconsistent votes.

**Q5 (gotcha). You use GPT-4 to judge a contest between a GPT-4-based system and a Claude-based system. Why is that setup risky?**
Self-enhancement (self-preference) bias: a judge tends to favor its own family's outputs, so GPT-4 may inflate the GPT-4 system's win rate. Use a judge from a neutral third family, or an ensemble of heterogeneous judges, and validate against human labels.

---

## Section 3 — Regression Testing and CI Integration

### The problem: things that used to work silently break

**Regression** means a change that makes something that used to work stop working. In LLM apps, regressions are sneaky: you edit a prompt to fix one thing, and unknowingly break the tone on a different category of input; or a model provider updates their model and your carefully-tuned outputs drift. Because nobody re-checks all the old cases by hand, the breakage ships.

**Regression testing** is the fix: re-run your golden set on every change and fail loudly if quality drops below a bar you set in advance. This is the same discipline traditional software uses — you just adapt it to non-deterministic output.

### Snapshot testing, and why the naive version fails for LLMs

**Snapshot testing** is a technique from ordinary software: you capture a known-good output (the "snapshot"), save it, and on future runs compare the new output against the saved one — any difference fails the test. It's great for catching unintended changes.

The naive version breaks immediately for LLMs, because their output is non-deterministic: the exact wording changes every run, so an exact-match snapshot would fail constantly even when nothing is actually wrong. You adapt it in two ways:

- **Make the model deterministic where you can.** LLMs have a **temperature** setting — a knob from 0 upward that controls randomness. At **temperature 0** the model is (close to) deterministic, returning the same output for the same input. For tests that check factual or structural properties, set temperature to 0 to reduce flakiness.
- **Snapshot the *meaning*, not the exact string.** Instead of exact match, compare against your saved reference using a fuzzy check — commonly **cosine similarity** on embeddings. (**Embeddings** turn text into numeric vectors where similar meanings are close; **cosine similarity** measures how close two vectors point, from 0 to 1.) You then pass if similarity to the reference is above a threshold, say 0.8. This tolerates rewording while catching real drift.

> **Gotcha — "flaky" tests destroy trust.** A **flaky test** passes sometimes and fails other times with no code change. If your regression suite is flaky, people start ignoring failures ("oh, it's just flaky again"), which defeats the purpose. Reduce flakiness by using temperature 0 for deterministic checks, semantic/fuzzy checks for stylistic ones, and — for genuinely non-deterministic behavior — running each test multiple times and requiring it to pass a fraction of them (e.g., 8 of 10) rather than every time.

### CI: making the suite mandatory

**CI (Continuous Integration)** is an automated system that runs your tests every time code changes — typically on every **pull request** (a proposed change submitted for review before it's merged). The point of putting evals in CI is to make them *unavoidable*: a change that drops quality below your bar fails the check and blocks the merge, so regressions can't quietly ship.

A tool commonly used here is **Promptfoo**, an open-source command-line tool for testing prompts. (It became part of OpenAI in early 2026 but remains open source under the permissive MIT license.) It has two complementary CI patterns worth understanding:

**Pattern 1 — a before/after PR comment.** Promptfoo publishes an official GitHub Action (`promptfoo/promptfoo-action`) that, when a prompt file changes in a pull request, runs the eval and posts a summary comment with pass/fail counts and a link to a side-by-side viewer. This is great for *review* — a human sees exactly what changed — but a comment alone doesn't block a merge.

**Pattern 2 — a quality gate that fails the build.** You run the eval, write results to a JSON file, and then read the pass/fail statistics and exit with an error if too many failed. A minimal gate reads the results and fails when there's any failure:

```yaml
# .github/workflows/prompt-eval.yml
name: LLM Prompt Eval
on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'promptfooconfig.yaml'
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-node@v6
        with:
          node-version: '22'            # promptfoo needs Node 20.20+ or 22+
      - name: Run prompt eval
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}   # key from repo secrets, never committed
        run: npx promptfoo@latest eval -c promptfooconfig.yaml -o results.json
      - name: Quality gate
        run: |
          # exit non-zero (fail the build) if any test failed
          FAILURES=$(jq '.results.stats.failures' results.json)
          echo "failures: $FAILURES"
          test "$FAILURES" -eq 0
```

The two patterns are complementary: the PR comment tells reviewers *what* changed; the quality gate is what actually *blocks* a bad change. Mature setups run both.

Inside the Promptfoo config, thresholds are how you encode "how good is good enough." Each test can carry a `threshold`, and the test passes only if the combined (weighted) score of its assertions meets it. You can also set a suite-wide minimum pass rate via an environment variable so the whole run fails if the overall pass percentage drops.

> **Gotcha — caching can hide a real regression.** Promptfoo caches model responses to disk by default so re-runs are fast and cheap. But if the *only* thing that changed is on the model provider's side (not your prompt), a cached response can mask the drift you're trying to catch. Use `--no-cache` when you specifically want fresh responses, e.g., a scheduled nightly run that checks for provider-side drift.

> **Gotcha — non-deterministic tests need repeats, not luck.** For agentic or open-ended behavior you don't expect to pass 100% of the time, use Promptfoo's `--repeat` to run each test several times and require a pass fraction, rather than gating on a single run that might flake red.

### Exercise 3 (hands-on)

1. Install Promptfoo (`npm install -g promptfoo`, or use `npx promptfoo@latest`). Create a `promptfooconfig.yaml` with one prompt, one provider, and 5 test cases drawn from your golden set. Include a mix of a deterministic assertion (`contains` / `is-json`) and a semantic one (`similar`, with a cosine-similarity threshold like `0.8`).
2. Run `promptfoo eval` and confirm all 5 pass.
3. **Deliberately introduce a regression:** edit the prompt so one category of output breaks (e.g., remove the instruction that keeps replies short). Re-run and confirm the relevant test now fails.
4. Add a quality-gate step (locally you can simulate it): `npx promptfoo eval -o results.json` then `jq '.results.stats' results.json`. Confirm the failures count is non-zero.
5. Set the broken test's provider to temperature 0 and re-run twice; note whether the pass/fail result is now stable across runs.

### Interview questions — Section 3

**Q1 (conceptual). What is a regression, and why are LLM apps especially prone to shipping them unnoticed?**
A regression is a change that breaks something that previously worked. LLM apps are prone to silent regressions because outputs are non-deterministic and interdependent — a prompt tweak can break an unrelated input category, or a provider-side model update can shift outputs — and nobody manually re-checks all past cases, so the breakage ships undetected.

**Q2 (conceptual). Why doesn't classic exact-match snapshot testing work for LLMs, and what are the two standard adaptations?**
Because LLM output wording changes every run, an exact-match snapshot fails even when the output is fine. Adaptations: (1) set temperature to 0 to make deterministic checks reproducible, and (2) snapshot the *meaning* using a fuzzy comparison like cosine similarity above a threshold, which tolerates rewording while catching real drift.

**Q3 (practical). How would you set up CI so a prompt change that degrades quality can't be merged?**
Run the eval suite on every pull request, write results to a file, and add a quality-gate step that reads the pass/fail stats and exits with an error (fails the build) when failures exceed your threshold — blocking the merge. Optionally add a PR-comment action showing a before/after diff so reviewers see what changed.

**Q4 (gotcha). Your CI eval passes on re-run without any code change but failed the first time. What's happening and how do you fix it?**
It's a flaky test caused by LLM non-determinism. Fix it by setting temperature to 0 for deterministic checks, using semantic/fuzzy assertions for stylistic ones, and for genuinely variable behavior, repeating each test N times and requiring a pass fraction (e.g., 8/10) instead of gating on a single run.

**Q5 (gotcha). Your regression suite keeps passing even though users report the model got worse after a provider update. What could be masking the problem?**
Response caching — Promptfoo caches model responses by default, so if only the provider's model changed (not your prompt), the suite may be scoring stale cached answers. Run with `--no-cache` (e.g., on a schedule) to force fresh responses and catch provider-side drift.

---

## Section 4 — Building a 30–50 Item Eval Suite for a RAG System

### First, what is RAG, and what makes evaluating it different?

**RAG (Retrieval-Augmented Generation)** is a common pattern for question-answering over your own documents. It has two stages: **retrieval** (search your document collection for chunks relevant to the question) and **generation** (feed those chunks plus the question to an LLM to write a grounded answer). The reason RAG needs its own eval design is that it has *two independent failure points*: the retriever can fetch the wrong or incomplete documents, and the generator can write a bad answer even from good documents (for example, a **hallucination** — a confident claim not supported by the source). Your eval suite has to be able to tell these two failures apart.

This section brings together everything above — golden sets, pairwise/scoring, regression discipline — into a concrete, buildable suite for a RAG project.

### Step 1 — Decide what "good" means (before writing any examples)

Write down, in one or two sentences each, the qualities you care about. For most RAG systems the shortlist is:

- **Correct / faithful:** the answer's claims are supported by the retrieved source (no hallucinations).
- **Relevant / complete:** the answer actually addresses the question and doesn't leave out needed information.
- **Retrieved the right context:** the documents pulled in were on-topic and sufficient.
- **Format / safety:** any structural rules (cites its source, stays under a length, refuses when it should).

These become your grading dimensions. Keeping the list short is deliberate — too many metrics become unmanageable and dilute focus.

### Step 2 — Assemble 30–50 items with the right *mix*

Aim for 30–50 rows to start (large enough to be meaningful, small enough to build and label by hand). Composition matters more than raw size. A good mix:

- **Core "happy path" questions (~50%)** — the common, straightforward questions your system must nail.
- **Edge cases (~30%)** — multi-part questions, questions whose answer spans several documents, ambiguous phrasings.
- **Adversarial / negative cases (~20%)** — questions your documents *can't* answer (the correct behavior is to say "I don't know," not to hallucinate), questions with false premises, and prompt-injection-style inputs.

> **Gotcha — if you're passing 100%, your suite is too easy.** A suite of only happy-path questions will look great and teach you nothing. The adversarial and edge cases are where you learn whether your system fails safely. Deliberately include cases you expect to be hard.

For each row, capture the fields a RAG eval needs: the `question`, the `retrieved_context` (the chunks your retriever returned), the `answer` (what your system generated), and — for the correctness/completeness checks — a `ground_truth` reference answer you write by hand. Store it as JSONL in your repo and commit it (version control, from Section 1).

> **Gotcha — separate retrieval failures from generation failures in your labels.** When you label a failure, note *which stage* failed. If the retrieved context didn't contain the answer, that's a retrieval problem; if the context was fine but the answer was wrong, that's a generation problem. Blending them into one "bad" label hides where to focus your fixes.

### Step 3 — Choose a grader per dimension

Match the grading method to the dimension:

- **Deterministic assertions** for format/safety (is it valid JSON? does it cite a source? is it under N words? on an unanswerable question, does it contain a refusal phrase?). These are cheap and reliable — prefer them wherever possible.
- **LLM-as-a-judge** for the fuzzy dimensions (faithfulness, relevance). Provide the judge the question, the retrieved context, the answer, and (for completeness) the ground truth, and ask a focused yes/no or 1-vs-0 question per dimension rather than a vague 1–5 score.
- **Fuzzy/semantic match** (cosine similarity to the ground truth above a threshold) when there's a clear reference answer and you mainly care that the meaning matches.

### Step 4 — Wire it into the harness and gate it

Reuse the pipeline discipline from earlier sections: one command runs the whole suite, results are written to a file, and a quality gate fails the run if the pass rate drops below your bar. Commit the config alongside the dataset so any score is reproducible.

> **Gotcha — keep the judge and embedding models fixed across runs.** The graders (LLM judge, embedding model for similarity) are themselves part of your measuring instrument. If you swap the judge model between runs, scores shift for reasons unrelated to your system, and comparisons become invalid. Pin them, and treat changing them as its own deliberate, documented change.

### Exercise 4 (hands-on) — the capstone

1. Write your "definition of good" for a RAG task: 3–4 short quality statements (Step 1).
2. Build a **30-item** JSONL eval set for that RAG task with the recommended mix: ~15 happy-path, ~9 edge cases, ~6 adversarial/unanswerable. For each item include `question`, `retrieved_context`, `answer`, `ground_truth`, and a `stage` note for any expected-hard case.
3. Assign each item a grader type: `deterministic`, `judge`, or `semantic`.
4. Run it through a harness (Promptfoo, a small script, or a RAG-specific tool — whatever you're comfortable with) and produce a headline score per dimension.
5. Do error analysis on the failures: for each, label it `retrieval` or `generation`, count how many of each, and write two sentences on which stage to fix first and why.
6. Commit the dataset, config, and results to version control as `eval-suite v1`.

### Interview questions — Section 4

**Q1 (conceptual). Why does a RAG system need an eval design that distinguishes two failure modes, and what are they?**
Because RAG has two independent stages that can each fail: retrieval (fetching wrong or incomplete documents) and generation (writing a bad or hallucinated answer even from good documents). If your labels don't separate them, you can't tell whether to fix your retriever or your prompt/model.

**Q2 (practical). You're building a 40-item RAG eval set. How would you compose it, and why not make it all realistic happy-path questions?**
Roughly half happy-path core questions, ~30% edge cases (multi-document, ambiguous, multi-part), ~20% adversarial/unanswerable. An all-happy-path suite would pass near 100% and teach you nothing about failure modes; the edge and adversarial cases reveal whether the system fails safely (e.g., says "I don't know" instead of hallucinating).

**Q3 (practical). For a RAG suite, which dimensions would you grade deterministically vs. with an LLM judge?**
Deterministic for format/safety — valid JSON, cites a source, length limits, contains a refusal on unanswerable questions. LLM-as-a-judge for the fuzzy dimensions — faithfulness (claims supported by context) and relevance/completeness — where no exact rule applies. Prefer deterministic checks wherever possible because they're cheaper and more reliable.

**Q4 (gotcha). Between two eval runs your faithfulness scores dropped, but you only changed the judge model. What went wrong?**
The judge is part of the measuring instrument; swapping it changes scores for reasons unrelated to your system, so the "drop" is an artifact, not a real regression. Pin the judge (and embedding) models across runs and treat any change to them as a deliberate, documented event.

**Q5 (conceptual). Why should the correct behavior on an unanswerable question be part of your eval set, and how would you grade it?**
Because a major RAG failure mode is hallucinating an answer when the documents don't contain one; you must verify the system refuses or says "I don't know." Grade it deterministically by asserting the output contains an appropriate refusal/uncertainty phrase and does *not* assert unsupported facts.

---

## Gotchas summary table

| # | Where | Gotcha | Why it bites | Fix / mindset |
|---|-------|--------|-------------|----------------|
| 1 | Pipeline design | Starting with generic tools/metrics | You skip looking at your own data; miss product-specific failures | Do error analysis first; build product-specific checks |
| 2 | Golden set | Editing the dataset while also changing the prompt | Can't tell if the prompt improved or the test got easier | Make dataset changes separate, reviewed commits |
| 3 | Version control | Hard-coding API keys in committed configs | Leaked secrets | Use env vars / secrets manager; never commit keys |
| 4 | Labeling | Using a 1–5 scale early on | Uncalibrated, inconsistent between raters | Use binary pass/fail (or win/lose) |
| 5 | Pairwise | Always showing one system first | Position bias inflates its win rate | Swap order, re-run, measure consistency; discard/tie flips |
| 6 | Pairwise | Judge favors longer answers | Verbosity bias rewards padding over quality | Instruct to ignore length; check if winner is just longer |
| 7 | Pairwise | A model judging its own family | Self-preference bias inflates its scores | Use a neutral-family judge or an ensemble |
| 8 | Pairwise scaling | Comparing every pair among many systems | ~N² comparisons; expensive and slow | Use Elo/rating from sampled matchups |
| 9 | Judges (all) | Trusting judge scores without validation | Judge is fallible & biased | Measure agreement vs. human labels first (aim >~80%) |
| 10 | Regression | Exact-match snapshot tests | Fail constantly due to non-determinism | Temperature 0 + semantic/fuzzy match with threshold |
| 11 | Regression / CI | Flaky tests | People start ignoring failures | Temp 0 for deterministic, repeat-N with pass fraction for variable |
| 12 | CI | Response caching masks provider-side drift | Suite passes though model got worse | `--no-cache` for scheduled fresh runs |
| 13 | RAG suite | Blending retrieval and generation failures in labels | Can't tell what to fix | Tag each failure with the stage that failed |
| 14 | RAG suite | All-happy-path dataset | Passes ~100%, teaches nothing | Include edge + adversarial/unanswerable cases |
| 15 | Any grader | Swapping judge/embedding models between runs | Scores shift for unrelated reasons | Pin the grader models; changing them is a deliberate event |

---

## Quick reference card

**Eval pipeline in three steps (Eugene Yan)**
1. Label a small dataset (binary pass/fail; start with 10–20 rows in a spreadsheet: `input`, `output`, `metadata`, `label`).
2. Align your evaluator against human labels before trusting it.
3. Run a harness on every change (runs graders in parallel within rate limits, aggregates scores). One command, runnable locally and in CI.

**Golden set** = curated + stable inputs with ideal outputs or criteria. Version-control it (Git, as CSV/JSON/JSONL). Dataset changes = separate commits. Never commit API keys.

**Pairwise comparison** = judge picks A / B / TIE for two answers to one question. More reliable than absolute scores.
- **Win rate** = P(this system preferred over opponent) = wins ÷ comparisons. **Average win rate** = across all opponents. State whether ties are included.
- Many systems → use **Elo**-style ratings, not all N² pairs.
- Biases: **position** (favors first — swap & re-check consistency), **verbosity** (favors longer — control for length), **self-preference** (favors own family — use neutral or ensemble judge).
- Validate the judge vs. humans; strong judges hit ~80%+ agreement (human-human level).

**Regression testing**
- **Snapshot for LLMs**: temperature 0 for determinism + cosine-similarity threshold (e.g. 0.8) instead of exact match.
- **Flaky tests** kill trust → temp 0, semantic checks, or repeat-N with a pass fraction (e.g. 8/10).
- **CI**: run suite on every pull request; PR-comment action shows the diff, quality gate (`jq` on `results.stats.failures`, non-zero exit) blocks the merge. Both together.
- Promptfoo: `promptfooconfig.yaml` (prompts, providers, tests + `assert` + `threshold`); `--no-cache` to defeat stale responses; Node 20.20+/22+.

**Building a RAG eval suite (30–50 items)**
1. Define "good" (faithful, relevant/complete, right context, format/safety) — keep it short.
2. Mix ~50% happy-path, ~30% edge, ~20% adversarial/unanswerable. Fields: `question`, `retrieved_context`, `answer`, `ground_truth`, `stage`.
3. Grader per dimension: deterministic (format/safety) > judge (faithfulness/relevance) > semantic (match to reference).
4. Wire into the harness, gate on pass rate, pin judge/embedding models, commit everything.
- Always include unanswerable questions; correct behavior is to refuse, not hallucinate. Tag each failure `retrieval` vs `generation`.

**Reference material:** Eugene Yan's eval writing (*Product Evals in Three Simple Steps*, eval-process posts); the MT-Bench / Vicuna "Judging LLM-as-a-Judge" work on pairwise comparison, win rate, and judge biases; Promptfoo's CI/CD and GitHub Action documentation.

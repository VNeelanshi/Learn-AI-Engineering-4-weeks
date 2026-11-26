# LLM-as-a-Judge: Rubrics, Pitfalls, Calibration, and Building a Judge You Can Trust

A hands-on beginner's guide. No prior knowledge assumed. Every term is defined the first time it appears. Roughly 70% explanation, 30% practice — each section ends with an exercise, and every section closes with five interview questions and answers.

A few definitions to anchor us:

- **LLM (Large Language Model):** a program (like GPT, Claude, or Llama) that takes text in and produces text out. It is *probabilistic* — the same input can produce different outputs on different runs.
- **Prompt:** the text instructions you send to an LLM.
- **Eval (evaluation):** a repeatable, automated test that measures whether an output is good.
- **LLM-as-a-judge:** using one LLM to grade the outputs of another (or of a human). It's the central subject of this guide.

The appeal is obvious: human judgment is the gold standard for open-ended text, but it's slow and expensive to collect. An LLM judge is fast and cheap, and it can also *explain* its verdicts. The catch — which this guide is largely about — is that a judge is itself a fallible, probabilistic system, so you can't just trust its scores. You have to design it carefully, know its biases, and prove it agrees with humans before relying on it.

---

## Section 1 — Judge Methodology: Rubrics and Calibration

### The problem: "score this from 1 to 10" doesn't work

The most natural way to build a judge is to ask an LLM, "Rate this answer's quality from 1 to 10." This fails in practice, for a reason worth internalizing early: an uncalibrated numeric scale is meaningless. What separates a 6 from a 7? Nobody knows — not the model, not you, not two different graders. The scores drift between runs and between raters, so you end up with precise-looking numbers that don't actually measure anything consistent.

The fixes are two ideas that structure every good judge: a **rubric** (so the judge knows exactly what it's scoring), and **calibration against humans** (so you know whether to believe it).

### Score rubrics: tell the judge exactly what "good" means

A **rubric** is an explicit description of the criteria and what each score level means. Instead of "rate 1–10," you write something like: "Score 1 if the answer contains any claim not supported by the source; score 0 otherwise," or a short scale where each point is defined ("2 = fully answers the question; 1 = partially answers; 0 = doesn't answer"). The tighter and more concrete the rubric, the more consistent the scores.

Two principles from practitioners who build judges for a living:

- **Prefer few, well-defined dimensions over many vague ones.** A judge scoring ten fuzzy qualities at once produces noise. Pick the one or two things you actually care about (e.g., faithfulness, relevance) and define each precisely.
- **Prefer binary or few-level scales over wide numeric ranges.** "Pass/fail" or "0/1" per dimension is far more reproducible than 1–10, because there's less room for interpretation.

### G-Eval: the standard structured-judge recipe

A widely-cited method for building judges is **G-Eval**, introduced in a 2023 paper on evaluating **NLG** (Natural Language Generation — any task where a model produces text, like summarization or dialogue). G-Eval matters because it took LLM judges from "roughly correlated with humans" to "closely correlated," and its recipe is now the default template. It has three parts:

1. **Task introduction + criteria.** You give the judge a plain-language description of the task and the criterion to assess (e.g., "coherence" or "faithfulness").
2. **Chain-of-thought (CoT) evaluation steps.** **Chain-of-thought** means prompting the model to reason through intermediate steps before answering. G-Eval has the model *generate its own detailed evaluation steps* from the criteria first, then apply them. Giving the judge explicit steps to follow improves its agreement with humans.
3. **Form-filling + probability-weighted score.** The judge "fills in" a score for the given output. Then, rather than taking the single number the model blurts out, G-Eval looks at the model's *probabilities* across the possible scores and computes a **probability-weighted average**. (If the model is 70% sure the score is 4 and 30% sure it's 3, the final score is 0.7×4 + 0.3×3 = 3.7.) This turns coarse integer scores into fine-grained continuous ones.

With GPT-4 as the backbone, G-Eval reached a **Spearman correlation of about 0.51** with human judgments on summarization — a large jump over older automatic metrics. (**Spearman correlation** measures how well two rankings agree, from −1 to +1; more on correlation in Section 3.)

> **Gotcha — you often can't get token probabilities.** The probability-weighted step needs access to the model's raw output probabilities, which many APIs don't expose. Without them you fall back to plain integer scores, which produce lots of ties. That's usable, but be aware you're running a simplified G-Eval, not the full method.

> **Gotcha — a judge tends to favor text written by an LLM.** The very paper that introduced G-Eval flagged that LLM judges show a bias toward LLM-generated text over human-written text of equal quality. Keep this in mind — it's a preview of the pitfalls in Section 2.

### Calibration with humans: the step you cannot skip

**Calibration** here means checking, and adjusting, how well your judge's verdicts match trusted human judgments. A judge's score is only meaningful once you've measured its **agreement** with humans on the same examples. The workflow:

1. Take a sample of outputs. Have humans label them (the ground truth for "what a good verdict looks like").
2. Run your judge on the same outputs.
3. Compare judge verdicts to human labels. Measure agreement (Section 3 covers the exact statistics).
4. If agreement is poor, revise the rubric/prompt and repeat. This is iterative — you go several rounds.

Only after the judge reliably agrees with humans do you let it grade unlabeled data at scale.

> **Gotcha — building the judge is partly a trick to make you read your data.** Practitioners point out that the real value of the calibration loop isn't just the automated judge at the end; it's that defining a rubric and comparing to human labels *forces* you to look carefully at your outputs and pin down what "good" actually means. If you skip straight to automation without ever reading outputs, you've missed most of the value.

### Exercise 1 (hands-on)

1. Pick a task and one quality you care about (e.g., "faithfulness: are all claims supported by the source?").
2. Write a **rubric** for it as a binary criterion: one sentence for what scores 1 (pass) and one for what scores 0 (fail). Add 1–2 concrete examples of each.
3. Turn it into a G-Eval-style judge prompt with three parts: task/criteria, an instruction to reason step-by-step before scoring, and a final "output only 0 or 1."
4. Run it on 6 outputs (mix of clearly-good and clearly-bad). Read the reasoning it produced for each.
5. Note any case where the reasoning was right but the final score was wrong, or vice versa — that mismatch tells you where your rubric or prompt needs tightening.

### Interview questions — Section 1

**Q1 (conceptual). Why is asking an LLM to "rate quality from 1 to 10" a poor way to build a judge?**
Because an uncalibrated numeric scale is undefined — there's no consistent meaning separating adjacent scores, so ratings drift between runs and raters. You get precise-looking numbers that don't measure anything reproducible. Tight rubrics and binary/few-level scales fix this.

**Q2 (conceptual). Explain the three parts of the G-Eval method.**
(1) Give the judge the task description and evaluation criteria. (2) Use chain-of-thought — have it generate and follow detailed evaluation steps before scoring. (3) Use a form-filling paradigm and compute a probability-weighted average over the possible scores, turning coarse integers into fine-grained continuous scores.

**Q3 (practical). When would you use a binary rubric versus a multi-level scale for a judge?**
Prefer binary (pass/fail, 0/1) whenever the quality can be framed as a yes/no property (e.g., "any unsupported claim?") because it's the most reproducible. Use a small multi-level scale only when there are genuinely distinct, definable degrees of quality that a binary can't capture — and define each level explicitly.

**Q4 (gotcha). You built a judge and it scores your data 92% good. Your manager wants to ship. What's missing?**
Calibration against humans. The score is meaningless until you've measured the judge's agreement with human labels on the same examples and iterated the rubric until agreement is acceptable. Without that, "92%" reflects the judge's untested opinions, not reality.

**Q5 (gotcha). Your G-Eval implementation only ever outputs whole-number scores with lots of ties. What's likely going on?**
The probability-weighted step needs the model's raw token probabilities, which your API probably isn't exposing. You're running a simplified G-Eval that falls back to integer scores (hence the ties). It still works, but you've lost the fine-grained continuous scoring the full method provides.

---

## Section 2 — Pitfalls: The Systematic Biases of LLM Judges

### The problem: judges fail in predictable, repeatable ways

An LLM judge doesn't fail randomly — it fails *systematically*, leaning the same direction every time. That's actually good news: predictable biases can be measured and corrected. The bad news is that if you don't know they exist, they silently corrupt every conclusion you draw. Three biases show up constantly and are the ones interviewers expect you to name.

### Position bias

**Position bias** is the judge's tendency to favor an answer based on *where it appears* rather than its quality — most commonly favoring whichever answer is shown **first** in a side-by-side comparison. (A **side-by-side** or **pairwise** comparison shows the judge two answers to the same question and asks which is better.) Studies of LLM judges found this can be substantial: swap the two answers' order and the judge sometimes flips its verdict, which means the verdict was about position, not merit.

**How to detect and mitigate:**
- **Swap and re-run.** Present the pair as (A, B) and also as (B, A). Compute the **position flip rate** — the fraction of pairs where swapping changes the verdict. A high flip rate means heavy position bias.
- **Count inconsistent verdicts as ties** (or discard them), keeping only verdicts that survive a swap.
- Track **consistency** (the percentage that give the same verdict both ways) as a health metric for the judge.

### Length / verbosity bias

**Verbosity bias** (also called length bias) is the tendency to favor **longer, more elaborate answers** even when a shorter answer is clearer or more correct. LLM judges reward the *appearance* of thoroughness.

**How to detect and mitigate:**
- Check whether your "winning" outputs are simply consistently longer. If your preferred version is always the wordier one, be suspicious.
- Add explicit instructions telling the judge to ignore length and not reward verbosity.
- Where possible, control for length (e.g., compare answers of similar length, or normalize).

### Self-preference (self-enhancement) bias

**Self-preference bias** is the tendency of a judge to favor answers produced by **itself or its own model family**. In the study that first quantified this, one model favored its own outputs by roughly 10 percentage points and another by about 25 — though the effect isn't universal across all models. The G-Eval work independently noted the same pull toward LLM-generated text.

**How to detect and mitigate:**
- **Never let a model be both contestant and judge.** If you're comparing outputs from model X, don't use model X as the judge.
- **Use a judge from a different family** than the systems under test.
- **Use an ensemble** of several different judges and aggregate their verdicts (more in Section 3); this dilutes any single model's self-preference.

> **Gotcha — these biases compound in a single evaluation.** A pairwise setup can suffer all three at once: the judge favors the first-shown, longer answer from its own family. If you don't control for them, you might "prove" your new system is better when you've only demonstrated that it was placed first, wrote more, or shared a family with the judge.

> **Gotcha — other biases exist too.** Beyond these three, judges can show sensitivity to formatting, tone, the presence of confident-sounding language, and cultural/linguistic context. The three above are the canonical ones to name, but treat any judge as guilty of *some* bias until you've checked.

### Exercise 2 (hands-on)

1. Take 8 questions and, for each, produce two answers: a **concise correct** answer and a **longer, padded** answer that's no more accurate.
2. Ask your judge which is better, always showing the padded one **first**. Record verdicts.
3. Now swap: show the concise one first. Record verdicts. Compute the **position flip rate**.
4. Separately, tally how often the *longer* answer won overall — a rough read on verbosity bias.
5. Write three sentences: estimate how much of your judge's "preference" was position vs. length, and state one concrete change to your judge prompt to reduce each.

### Interview questions — Section 2

**Q1 (conceptual). Name and define the three canonical LLM-judge biases.**
Position bias — favoring an answer by its location (often the first shown). Verbosity/length bias — favoring longer answers regardless of quality. Self-preference/self-enhancement bias — favoring answers from the judge's own model or family.

**Q2 (practical). How do you detect and correct for position bias in a pairwise eval?**
Present each pair in both orders (A,B) and (B,A) and compute the position flip rate — the fraction of pairs whose verdict changes on swap. Treat flipped verdicts as ties or discard them, keep only swap-stable verdicts, and track consistency as a judge-health metric.

**Q3 (practical). You must evaluate outputs from GPT-4. Which model should judge them, and why not GPT-4?**
Use a judge from a different family (e.g., Claude or Gemini) or an ensemble, because self-preference bias means GPT-4 tends to favor its own outputs, inflating their scores. A model should never be both contestant and judge.

**Q4 (gotcha). Your new prompt wins 68% of pairwise comparisons and its answers are noticeably longer. What two biases could explain the "win" and how would you rule them out?**
Verbosity bias (longer answers favored) and, if the new answer is shown first, position bias. Rule them out by swapping order and computing the flip rate, and by checking whether the winner is simply always longer — then re-test with length controlled and order randomized.

**Q5 (gotcha). Why is it dangerous to conclude "System B is better" from a single-order, single-judge pairwise run?**
Because position, verbosity, and self-preference biases can each produce a spurious win, and they can compound. Without swapping order, controlling for length, and using a neutral or ensemble judge, the result may reflect artifacts rather than true quality.

---

## Section 3 — Calibrating Judges: Agreement, Confidence Intervals, and Ensembles

### The problem: "the judge mostly agrees with me" is not a measurement

To trust a judge you need a *number* for how well it agrees with humans — and the obvious number, **percent agreement** (the fraction of items where judge and human gave the same label), is misleading. Here's why: if 95% of your items are "pass," a judge that blindly says "pass" every time will hit 95% agreement while measuring nothing. High percent agreement can hide near-total failure. This trap is common enough to have a name, the **kappa paradox** (high agreement, low real reliability). You need **chance-corrected** metrics that subtract out the agreement you'd expect by luck.

### Chance-corrected agreement: Cohen's kappa and Krippendorff's alpha

**Cohen's kappa (κ)** measures agreement between **two** raters, corrected for chance. It answers: "how much better than random guessing do these two agree?" The scale:

- **1.0** = perfect agreement.
- **0** = no better than chance.
- **negative** = systematic disagreement (worse than chance).

A common rough interpretation: below 0.20 is slight, 0.21–0.40 fair, 0.41–0.60 moderate, 0.61–0.80 substantial, and above 0.80 near-perfect. For a judge you'd like to see *substantial* agreement (κ ≳ 0.6) with humans, though the right bar depends on how much your human raters even agree with each other.

**Krippendorff's alpha (α)** is the more general tool. It also corrects for chance but, unlike kappa, it handles **more than two raters**, **any scale** (yes/no, ordinal ranks, continuous numbers), and **missing ratings** (not every rater rated every item). A conventional acceptability threshold is around **0.667**, with 0.8+ considered good. Because it generalizes kappa, alpha is the safer default when your setup is anything beyond two raters and a binary label.

> **Gotcha — don't report percent agreement alone.** It's the number everyone reaches for and the one that most often misleads, especially when your labels are imbalanced (mostly "pass"). Always pair it with a chance-corrected metric (kappa or alpha), and note the label balance.

> **Gotcha — pick the metric that matches your data.** Cohen's kappa is for two raters and categorical labels. If you have three or more judges, ordinal/continuous scores, or missing data, use Krippendorff's alpha (or an intraclass correlation for continuous scores). Using the wrong statistic produces numbers that look fine but don't mean what you think.

### Correlation, for continuous scores

When your judge produces a *continuous* score (like G-Eval's), you often care about whether it *ranks* items the way humans do, not exact-value match. Three correlation measures (all from −1 to +1):

- **Pearson** — linear relationship between the two sets of numbers.
- **Spearman** — agreement of the two *rankings* (robust to non-linear but monotonic relationships).
- **Kendall's tau** — another rank-agreement measure, robust to ties.

G-Eval, for instance, reported Spearman correlation with humans because rank agreement is what matters for choosing between systems.

### Confidence intervals: don't over-read a single number

A judge's agreement or a system's win rate is estimated from a *finite sample*, so it has uncertainty. A **confidence interval (CI)** is a range that expresses that uncertainty — e.g., "win rate 62%, 95% CI [55%, 69%]" means, loosely, the true value is plausibly anywhere in that range. The most common way to get one for eval metrics is **bootstrapping**: repeatedly resample your evaluation items with replacement, recompute the metric each time, and read off the spread of results.

Why this matters: if System A scores 62% and System B scores 60% but both CIs overlap heavily, you have **not** shown A is better — the gap is within noise. Reporting eval results without error bars is one of the most common ways teams fool themselves.

> **Gotcha — small samples give huge intervals.** With only 20 items, your confidence interval is wide, so only large differences are meaningful. Don't declare a 2-point win on a 20-item set; either collect more data or acknowledge the difference is within noise.

### Intra-rater reliability: is the judge even consistent with itself?

Before comparing a judge to humans, check whether it agrees with *itself*. **Intra-rater reliability** (also **test-retest reliability** or **self-consistency**) is how consistent one judge is when it re-grades the same items. Because LLMs are probabilistic, a judge run twice can give different verdicts. Measure it by running the judge N times on the same items and checking how often the verdicts match (for continuous scores, an **intraclass correlation coefficient, ICC**, is used).

> **Gotcha — a judge that disagrees with itself can't agree with anyone.** If your judge flips its own verdicts across runs, no amount of human calibration will make it reliable. Improve self-consistency first — often by setting **temperature** (the randomness knob) to 0, tightening the rubric, and using chain-of-thought — then measure agreement with humans.

### Multi-judge ensembles: strength in numbers

An **ensemble** (or **panel**) of judges uses several *different* models to grade each item and combines their verdicts. The simplest combination is **majority vote**; you can also average continuous scores or weight judges by their measured reliability. Ensembles help because different models have different biases, so aggregating cancels out some individual quirks (notably self-preference, since no single model dominates). The cost is more API calls, so ensembles are typically reserved for higher-stakes evaluations.

> **Gotcha — an ensemble of similar models shares their biases.** Three models from the same family will largely agree *and* share the same blind spots, so the "ensemble" gives false confidence. Diversity is the point — pick judges from different families.

### Exercise 3 (hands-on)

1. Take 20 items with human pass/fail labels (or make them). Run one judge on all 20.
2. Compute **percent agreement** and **Cohen's kappa** between judge and human. Note the gap between them, especially if your labels are imbalanced.
3. Run the judge on the same 20 items a **second time**. Compute self-consistency (fraction of items with the same verdict both runs). If it's low, set temperature to 0 and repeat.
4. Bootstrap a confidence interval for the agreement: resample the 20 items with replacement ~1000 times, recompute agreement each time, and take the 2.5th and 97.5th percentiles as a 95% CI.
5. Write two sentences on what the width of that interval tells you about whether 20 items is enough.

### Interview questions — Section 3

**Q1 (conceptual). Why is percent agreement a poor standalone measure of judge quality?**
Because it doesn't correct for chance: with imbalanced labels (say 95% "pass"), a judge that always says "pass" hits 95% agreement while measuring nothing — the kappa paradox. You need chance-corrected metrics like Cohen's kappa or Krippendorff's alpha to know real agreement.

**Q2 (conceptual). When would you use Cohen's kappa vs. Krippendorff's alpha?**
Cohen's kappa for exactly two raters with categorical labels. Krippendorff's alpha when you have more than two raters, ordinal/continuous scales, or missing ratings — it generalizes kappa across all those cases, making it the safer default beyond the simple two-rater binary setup.

**Q3 (practical). System A scores 62% and System B 60% on a 200-item eval. How do you decide whether A is really better?**
Compute confidence intervals (e.g., by bootstrapping). If the intervals overlap substantially, the 2-point gap is within noise and you can't claim A is better; if they're well separated, the difference is meaningful. Never conclude from point estimates alone.

**Q4 (gotcha). Your judge scores the same 10 items differently on two runs. What's wrong and what do you fix first?**
It has poor intra-rater reliability (self-consistency) — expected given LLM randomness. Fix self-consistency before anything else: set temperature to 0, tighten the rubric, use chain-of-thought. A judge that can't agree with itself can't reliably agree with humans.

**Q5 (gotcha). You build a three-judge ensemble but agreement barely improves and biases persist. What might be the flaw?**
The three judges are probably from the same model family, so they share biases and blind spots and mostly agree with each other — giving false confidence. Ensembles work only with *diverse* judges from different families, so their independent errors can cancel out.

---

## Section 4 — Building and Validating a Judge on Human-Labeled Samples

### Why validation on a labeled set is the whole game

Everything so far converges here: a judge is worthless until you've proven it agrees with humans, and the way you prove it is by running it against a set of human-labeled samples and measuring agreement. Twenty labeled samples is a realistic starting point — enough to catch a badly-miscalibrated judge, while small enough to label by hand in an afternoon. (Twenty is a floor, not a target: your confidence intervals will be wide, so treat conclusions as directional.)

### The end-to-end procedure

**Step 1 — Create the human-labeled set.**
Collect ~20 real outputs. For each, a human assigns the ground-truth label using the *same rubric* the judge will use. Keep it binary (pass/fail) for reliability. If you can, have two people label independently and check their agreement (Cohen's kappa) — this tells you the *ceiling*: your judge can't be expected to agree with humans more than humans agree with each other.

> **Gotcha — vague rubric, useless labels.** If two humans can't agree using your rubric, the rubric is too vague, and no judge built on it can be reliable. Fix the rubric until human-human agreement is solid *before* building the judge.

**Step 2 — Build the judge.**
Write a G-Eval-style prompt: task + criteria, chain-of-thought steps, then "output only 0 or 1." Set temperature to 0 for consistency. Use a judge model from a *different* family than the system that produced the outputs, to avoid self-preference.

**Step 3 — Run the judge on the labeled set.**
Grade all 20 items. Run twice to check self-consistency; if the judge flips its own verdicts, tighten the prompt before proceeding.

**Step 4 — Measure agreement.**
Compute percent agreement *and* Cohen's kappa between judge and human labels. Bootstrap a confidence interval so you know how much to trust the estimate on just 20 items.

**Step 5 — Do error analysis on disagreements.**
This is the highest-value step. Read every item where the judge and human disagreed. Categorize why: Did the judge miss a subtle hallucination? Reward length? Misread the rubric? Each disagreement points to a specific fix.

**Step 6 — Iterate.**
Revise the rubric or prompt to address the disagreement patterns, then re-run and re-measure. Repeat until agreement (kappa) is acceptable relative to the human-human ceiling. Only then use the judge on unlabeled data at scale.

> **Gotcha — don't tune the judge into the noise.** With 20 items it's tempting to keep tweaking until the judge matches all 20. Beware **overfitting** — tuning so hard to your tiny sample that the judge learns *those specific items* rather than the general rule, and then fails on new data. Hold out a few items, or validate on a fresh labeled batch, to confirm the improvement generalizes.

> **Gotcha — the labeled set can go stale.** As your product and inputs evolve, an old labeled set may stop representing what users actually send. Periodically refresh it with recent real examples, or your "validated" judge is validated against a world that no longer exists.

### Exercise 4 (hands-on) — the capstone

1. Assemble **20 real outputs** for your task and label each pass/fail by hand using one clear rubric. If possible, get a second person to label them too, and compute human-human Cohen's kappa (your ceiling).
2. Build a G-Eval-style judge (task + CoT steps + binary output, temperature 0, neutral judge family).
3. Run it twice; compute self-consistency. Tighten the prompt if verdicts flip.
4. Compute judge-vs-human percent agreement and Cohen's kappa, plus a bootstrapped 95% CI.
5. Read all disagreements, categorize the causes, make one targeted rubric/prompt change, and re-run.
6. Write a short verdict (4–5 sentences): is this judge trustworthy enough to run on unlabeled data? Cite your kappa, your CI width, and the human-human ceiling to justify the answer.

### Interview questions — Section 4

**Q1 (conceptual). Why measure human-human agreement before validating a judge against humans?**
Because human-human agreement sets the ceiling: if two humans only agree at κ = 0.7 using your rubric, you can't expect a judge to exceed that. It also diagnoses the rubric — low human-human agreement means the rubric is too vague to build any reliable judge on.

**Q2 (practical). Outline the steps to validate a judge on a 20-item human-labeled set.**
Create the labeled set with a shared rubric (ideally two labelers + their kappa); build a G-Eval-style judge with a neutral-family model at temperature 0; run it (twice, to check self-consistency); measure percent agreement and Cohen's kappa with a bootstrapped CI; do error analysis on every disagreement; iterate the rubric/prompt and re-measure until agreement is acceptable.

**Q3 (practical). Which step in judge validation is most valuable and why?**
Error analysis on disagreements. Every judge-vs-human mismatch reveals a concrete failure (missed hallucination, length bias, misread rubric) and thus a specific, actionable fix — turning a single agreement number into a to-do list for improving the judge.

**Q4 (gotcha). You tweaked your judge until it matched all 20 labeled items perfectly. Why might you not celebrate?**
Likely overfitting — you may have tuned the judge to those 20 specific items rather than the general rule, so it could fail on new data. Confirm by validating on held-out or fresh labeled examples; only generalizing improvements count.

**Q5 (gotcha). Your judge was validated six months ago at κ = 0.75 and you've trusted it since. What could have quietly gone wrong?**
The labeled set may be stale — if your product's inputs have drifted, the judge is validated against inputs users no longer send, so its real-world reliability is unknown. Refresh the labeled set with recent examples and re-validate periodically.

---

## Gotchas summary table

| # | Where | Gotcha | Why it bites | Fix / mindset |
|---|-------|--------|-------------|----------------|
| 1 | Rubrics | "Rate 1–10" scales | Uncalibrated; scores drift & mean nothing | Use tight binary/few-level rubrics with defined levels |
| 2 | G-Eval | Probability-weighting needs token probs | Many APIs don't expose them → integer scores, ties | Know you're running simplified G-Eval; accept ties |
| 3 | Judges | Bias toward LLM-generated text | Inflates scores for machine text | Anticipate self-preference; use neutral judges |
| 4 | Calibration | Skipping human calibration | Judge scores are untested opinions | Always measure agreement vs. human labels first |
| 5 | Pitfalls | Position bias (favors first answer) | Verdict reflects order, not quality | Swap order, compute flip rate, tie/discard flips |
| 6 | Pitfalls | Verbosity/length bias | Longer ≠ better, but judge rewards it | Instruct to ignore length; check if winner is just longer |
| 7 | Pitfalls | Self-preference bias | Model favors its own family | Never self-judge; use a different family or ensemble |
| 8 | Pitfalls | Biases compound | Multiple artifacts stack into a fake "win" | Control order, length, and judge family together |
| 9 | Agreement | Reporting percent agreement alone | Kappa paradox hides failure on imbalanced labels | Always add a chance-corrected metric (kappa/alpha) |
| 10 | Agreement | Wrong statistic for the data | Numbers look fine but mean the wrong thing | Kappa: 2 raters/categorical; alpha: >2 raters/any scale/missing |
| 11 | Results | No confidence intervals | Over-read tiny differences as real | Bootstrap CIs; overlapping CIs = no proven difference |
| 12 | Results | Small sample (20 items) | CIs are wide; only big gaps meaningful | Don't claim small wins; collect more or hedge |
| 13 | Consistency | Judge disagrees with itself | Can't agree with humans if inconsistent | Fix self-consistency first (temp 0, tighter rubric, CoT) |
| 14 | Ensembles | Same-family judge panel | Shared biases → false confidence | Use diverse judges from different families |
| 15 | Validation | Overfitting the judge to 20 items | Learns those items, not the rule | Hold out / validate on fresh labeled data |
| 16 | Validation | Stale labeled set | Validated against inputs users no longer send | Refresh with recent real examples; re-validate |

---

## Quick reference card

**Building a judge (rubric + G-Eval)**
- Kill "rate 1–10." Use a **rubric**: precise criteria, binary or few defined levels.
- **G-Eval** recipe: (1) task + criteria → (2) chain-of-thought eval steps → (3) form-filling + probability-weighted score (continuous). Needs token probabilities for the last part; else integer scores.
- Judges favor LLM-written text → expect self-preference.

**Calibration loop (never skip)**
Human-label a sample → run judge on same items → measure agreement → error-analyze disagreements → revise rubric/prompt → repeat. Then scale to unlabeled data.

**The three canonical biases**
- **Position** — favors first answer → swap order, compute flip rate, tie/discard flips; track consistency.
- **Verbosity/length** — favors longer → instruct to ignore length; check if winner is just wordier.
- **Self-preference** — favors own family → neutral-family judge or ensemble; never self-judge.

**Agreement statistics**
- **Percent agreement** alone lies (kappa paradox on imbalanced labels).
- **Cohen's κ** — 2 raters, categorical; chance-corrected; 1 perfect, 0 chance, <0 systematic disagreement; ≳0.6 substantial.
- **Krippendorff's α** — ≥2 raters, any scale, missing data; ~0.667 acceptable, 0.8+ good. Safer default.
- **Correlation** (continuous scores): Pearson (linear), Spearman (rank), Kendall's tau (rank, tie-robust).

**Uncertainty & consistency**
- **Confidence interval** via **bootstrap** (resample items ~1000×, take 2.5/97.5 percentiles). Overlapping CIs = no proven difference. Small n → wide CI.
- **Intra-rater / self-consistency**: run judge N× on same items; fix low consistency (temp 0) before trusting it.
- **Ensemble**: multiple *diverse* judges, majority vote / averaged score; dilutes single-model bias; costs more calls.

**Validating on ~20 human-labeled samples**
Shared rubric (ideally 2 labelers + their κ = your ceiling) → G-Eval judge, temp 0, neutral family → run twice for self-consistency → percent agreement + κ + bootstrapped CI → error-analyze disagreements → iterate. Avoid overfitting to the 20; refresh the set as inputs drift.

**Reference material:** the G-Eval paper (*NLG Evaluation using GPT-4 with Better Human Alignment*); LLM-as-a-judge survey literature on judge taxonomy and biases; inter-rater reliability methods (Cohen's kappa, Krippendorff's alpha, ICC, bootstrap confidence intervals).

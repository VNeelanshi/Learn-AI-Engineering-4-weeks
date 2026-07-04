# The Embedding Model Landscape: Providers, Open-Source, Benchmarks, and Fine-Tuning

The model you use to turn text into vectors quietly determines the ceiling on every search and question-answering system you build. This guide maps the whole landscape: the commercial API providers, the open-source families you can self-host, the benchmark everyone cites (and how to read it without being misled), and when and how to train your own. By the end you'll be able to pick an embedding model deliberately instead of defaulting to whatever a tutorial happened to use.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order.

> **A note on specifics:** model names, benchmark scores, and prices in this field change *monthly*. The figures here are accurate as of early/mid 2026 and are included to build intuition, but always verify the current model lineup and pricing on each provider's own documentation before committing. The way to read the landscape outlasts any single number.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers — a **vector** — that represents the *meaning* of text. Text with similar meaning produces vectors that sit close together; different meaning, far apart.
- An **embedding model** is the trained program that turns text into a vector. Each number in the vector is a **dimension**; a model outputs a fixed number of them (e.g. 384, 1024, 1536, 3072).
- You compare two vectors with a **similarity metric**, usually **cosine similarity** (the angle between them — meaning, ignoring length).
- **Retrieval** means finding the stored texts most similar to a query. **RAG** (Retrieval-Augmented Generation) is the common pattern of retrieving relevant text and feeding it to a text-generating AI so its answers are grounded in your documents.
- A **token** is roughly a word-piece; models and APIs measure input length and price in tokens.

That's enough to begin. The thread running through this whole guide: **a vector is only comparable to other vectors from the same model.** Switching models means re-computing every vector you've stored. Keep that in mind — it makes the choice consequential.

---

## Hour 1 — The Commercial Provider Landscape

### The problem we're solving

The fastest way to get good embeddings is to call a hosted **API** (a service you send text to over the internet and get vectors back, paying per token). You don't manage any hardware. Three providers dominate the quality conversation, and they've each made different bets. Knowing what distinguishes them lets you choose on more than brand familiarity.

### A shared idea you must not miss: input types

Before the specific providers, one concept that trips up nearly every beginner. Several modern models want you to tell them *what role* a piece of text plays: are you embedding a **document** (something to be stored and searched) or a **query** (a search request)? The model embeds the two slightly differently to make them match better at search time. This is variously called an **input type** or **prompt**.

> **Gotcha — forgetting the input type silently hurts retrieval.** If a model expects you to mark documents vs. queries and you don't (or you mark them the same), search quality drops — with no error. This single mistake is behind a large share of "my retrieval is mediocre" complaints. Always check whether your model wants an input type and set it correctly.

### OpenAI: the safe default

OpenAI offers two current-generation text embedding models: **text-embedding-3-small** (1536 dimensions) and **text-embedding-3-large** (3072 dimensions). The small model is the most widely deployed embedding model in production — cheap, well-documented, and supported by essentially every vector database out of the box. The large model scores modestly higher on quality benchmarks at several times the price.

A few defining details:
- **Matryoshka embeddings.** Both models are trained with a technique (Matryoshka Representation Learning) that packs the most important information into the *first* dimensions of the vector. This lets you request a *shorter* vector via a `dimensions` parameter (e.g. 256, 512, 1024) and lose surprisingly little quality. **Matryoshka** here just means "nested": a smaller vector is a usable prefix of the larger one. Shorter vectors cost less to store and search.
- **Normalized output.** OpenAI's vectors come out with length 1 (**normalized**), which means cosine similarity and dot product give identical rankings — so you can use the cheaper dot product.
- **Input limit.** Up to about 8,191 tokens per request; longer text is **silently truncated**, so you must split long documents into **chunks** (smaller passages) before embedding.
- **No input-type parameter** (unlike the next two providers) — you embed text directly.
- **Batch pricing.** A batch mode runs embeddings at roughly half price with slower turnaround — ideal for embedding a whole corpus up front.

```python
from openai import OpenAI
client = OpenAI()  # reads the OPENAI_API_KEY environment variable

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["Refunds take 5 business days.", "Office hours are 9 to 5."],
    dimensions=512,   # optional: shorten the vector (256–1536 for the small model)
)
vector = resp.data[0].embedding
print(len(vector))   # 512
```

### Cohere: multimodal and enterprise-oriented

Cohere's current model, **embed-v4.0**, is **multimodal** — it can embed not just text but images and even interleaved text-and-image content (like a screenshot of a PDF page with a chart) into the *same* vector space. **Multimodal** means it handles more than one type of input. That's powerful for document-heavy systems: you can index slides, tables, and figures directly without first converting them to text.

Defining details:
- **Input types required.** Cohere wants `input_type="search_document"` for stored content and `input_type="search_query"` for searches (also options for classification and clustering). This is the input-type concept from above, and Cohere is strict about it.
- **Matryoshka dimensions** of 256, 512, 1024, or 1536.
- **Quantization options** (float, int8, binary, and more). **Quantization** means storing each number with fewer bits (e.g. 8-bit integers or single bits instead of 32-bit floats), shrinking storage dramatically at a small quality cost.
- **Long context** (up to ~128k tokens) and 100+ languages.

```python
import cohere
co = cohere.ClientV2()  # reads the COHERE_API_KEY environment variable

resp = co.embed(
    model="embed-v4.0",
    texts=["Refunds take 5 business days."],
    input_type="search_document",     # use "search_query" for searches
    output_dimension=1024,
    embedding_types=["float"],
)
# (Exact response attribute can vary by SDK version; check the docs.)
```

### Voyage AI: quality-focused and domain-specialized

**Voyage AI** (now part of MongoDB) focuses on top-tier retrieval quality and offers **domain-specialized** models alongside general ones — separate models tuned for **code**, **law**, and **finance**, where specialized vocabulary makes a general model weaker. Its general-purpose lineup moves quickly (the flagship general models are the current "voyage-4" generation, with strong prior generations like voyage-3.5 still available).

Defining details:
- **Domain models.** For example, a code-specialized model for programming text, and finance/legal models for those domains. On specialized corpora these typically beat general models by a noticeable margin.
- **Input types** (`input_type="query"` or `"document"`), like Cohere.
- **Matryoshka dimensions and quantization**, similar in spirit to the others.
- **Shared embedding space across a generation.** Models within a generation are designed to be compatible, so you can embed documents with a high-accuracy model and embed queries with a cheaper, faster one *without re-indexing*. (This is the exception that proves the rule — compatibility is a deliberate design feature *within* a family, not something you can assume across providers or generations.)
- **Normalized output**, so cosine equals dot product.

```python
import voyageai
vo = voyageai.Client()  # reads the VOYAGE_API_KEY environment variable

result = vo.embed(
    texts=["Refunds take 5 business days."],
    model="voyage-3.5",          # general; or a domain model like "voyage-code-3"
    input_type="document",        # use "query" for searches
    output_dimension=1024,
)
vector = result.embeddings[0]
```

### How to choose among them (for now)

A reasonable rule of thumb as of early/mid 2026: **OpenAI text-embedding-3-small** is the safe, cheap, universally-supported default for English-heavy work; **Cohere embed-v4** stands out when you need multimodal (documents with images) or strong multilingual support; **Voyage** stands out when retrieval quality is paramount or your data is in a specialized domain (code, law, finance). But these positions shift constantly, and the deciding factor should always be a small test on *your* data (more on that in Hour 3).

### Hands-on exercise (Hour 1)

1. Get an API key for at least one provider (all offer free starting credits). Embed the same 5 sentences with it.
2. If you have access to two providers, embed the same sentences with both and confirm the vectors have *different* lengths and values — concretely showing the two outputs are not interchangeable.
3. With a provider that uses input types (Cohere or Voyage), embed a short "document" with the document input type and a related "query" with the query input type, then compute their cosine similarity. Re-embed both with the *same* (document) input type and compare similarities. Note any difference — this makes the input-type concept tangible.
4. With OpenAI (or any Matryoshka model), embed one sentence at full size and again with a smaller `dimensions` value. Confirm the shorter vector is a near-prefix and reason about the storage savings at 1 million documents.

### Interview questions — Hour 1

**1. (Conceptual) What is an "input type" (or prompt) in an embedding API, and why does it matter?**
It tells the model what role the text plays — typically document (to be stored/searched) versus query (a search request) — so the model embeds each in a way that makes matching queries to documents more accurate. It matters because models that expect it will return lower-quality retrieval if you omit it or mark everything the same, with no error to warn you.

**2. (Conceptual) What are Matryoshka embeddings and what practical benefit do they give?**
They're embeddings trained so the most important information sits in the leading dimensions, letting you truncate the vector to a smaller size with minimal quality loss. The benefit is flexible cost control: you can store and search shorter vectors (less memory, faster) without retraining or switching models.

**3. (Practical) For an English-only RAG prototype on a tight budget, which commercial model would you reach for, and why?**
OpenAI text-embedding-3-small is a strong default: it's inexpensive, widely supported by vector databases and frameworks, and good enough for most English retrieval. You'd only move to a larger or specialized model if a test on your own data showed a meaningful quality gain that justified the extra cost.

**4. (Practical) When would Voyage's domain-specific models or Cohere's multimodal model be worth choosing over a general OpenAI model?**
Voyage's domain models (code, law, finance) are worth it when your corpus is heavy in that specialized vocabulary, where they typically outperform general models. Cohere embed-v4 is worth it when you need to embed images or mixed text-and-image documents in the same space, or need strong multilingual coverage — capabilities a text-only general model doesn't provide.

**5. (Gotcha) Why can't you store vectors from OpenAI's model and later query them with a vector from Cohere's model?**
Each model maps text into its own distinct vector space, so vectors from different models (and often different generations of the same provider) aren't comparable — similarity scores between them are meaningless. Querying an OpenAI-built index with a Cohere query vector would return essentially random results; you'd have to re-embed the whole corpus with one consistent model.

---

## Hour 2 — Open-Source Embeddings

### The problem we're solving

APIs are convenient, but sometimes you want to run the model yourself: to keep sensitive data on your own machines, to avoid per-token costs at high volume, to work offline, or for full control. **Open-source** (also called **open-weight**) embedding models — whose trained parameters are published for anyone to download and run — make this possible. You typically get them from **Hugging Face**, a hub that hosts models, and run them locally (often via the `sentence-transformers` Python library), which requires a **GPU** (a graphics processor that accelerates the math) for good speed at scale.

A model's **parameters** are the learned numbers that make it work; the count (e.g. 22M, 568M, 8B — M = million, B = billion) is a rough proxy for its size, memory needs, and speed. Bigger models are usually more accurate but slower and hungrier for memory.

### The major families

A **family** is a set of related models from the same developer, usually offered in several sizes. The ones you'll encounter constantly:

- **BGE** (from the Beijing Academy of Artificial Intelligence). Comes in small/base/large English sizes plus **bge-m3**, a multilingual model (100+ languages, ~568M parameters) notable for **multi-functionality**: a single model that can produce dense vectors, sparse (keyword-style) vectors, and multi-vector representations, and handle inputs up to ~8,192 tokens. Permissively licensed (MIT). A strong 2026 choice for multilingual and hybrid search.
- **E5** (from Microsoft). Sizes from **e5-small** (~118M parameters) up through large, plus **multilingual-e5-large** (1024 dimensions, 100 languages). E5 has a signature quirk: it expects you to prefix text with **`query:`** or **`passage:`** — its version of the input-type concept from Hour 1.
- **GTE** (General Text Embeddings, from Alibaba). small/base/large and a **gte-multilingual-base** (~305M parameters); a solid, efficient general-purpose family.
- **Qwen3-Embedding** (from Alibaba's Qwen team). Offered at **0.6B, 4B, and 8B** parameters, this is among the strongest open families in 2026 — the 8B model has topped the multilingual benchmark leaderboard. It's **instruction-aware** (you can prepend a task instruction to steer it, typically gaining a few percent), supports 100+ languages and long text, offers flexible output dimensions, and is permissively licensed (Apache 2.0).

You'll also hear about lightweight options like **all-MiniLM-L6-v2** (a small, fast, 384-dimension model that's hugely popular for prototyping but uses an older architecture and underperforms modern models on quality), **EmbeddingGemma** (a ~300M on-device model from Google), and larger research-grade models like **NV-Embed** (NVIDIA, ~8B).

### The two big trade-offs

**Size vs. quality vs. cost.** Larger models tend to score higher but cost more in GPU memory, latency, and infrastructure. Crucially, **bigger is not automatically better for your task** — small models sometimes beat much larger ones on specific workloads. And dimension count drives storage: a 1024-dimension 32-bit vector is about 4 KB, so 10 million documents is roughly 40 GB of vector storage — doubling dimensions doubles that.

**Generality vs. specialization.** General models are fine for broad content; for specialized domains (legal, biomedical, code), a domain-tuned model usually wins.

> **Gotcha — many open-source models require specific prefixes or instructions.** E5 needs `query:`/`passage:` prefixes; BGE uses a retrieval instruction for the query side; instruction-aware models like Qwen3 want a task instruction (e.g. `Instruct: <task>\nQuery: <text>`). **Omitting these causes a silent, sometimes large, quality drop.** Always read the model card (the model's documentation page) for the exact expected input format before judging a model's quality.

```python
from sentence_transformers import SentenceTransformer

# E5 REQUIRES "query:" / "passage:" prefixes — omitting them hurts quality
model = SentenceTransformer("intfloat/multilingual-e5-large")
doc = model.encode("passage: Refunds are issued within 5 business days.",
                   normalize_embeddings=True)
qry = model.encode("query: how long until I get my refund",
                   normalize_embeddings=True)

# Other families load the same way (mind each one's expected input format):
# SentenceTransformer("BAAI/bge-m3")
# SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
# SentenceTransformer("Alibaba-NLP/gte-multilingual-base")
```

> **Gotcha — check the license before commercial use.** "Open" doesn't always mean "free for any use." Most of the families above use permissive licenses (MIT, Apache 2.0), but always confirm the license on the model card matches how you intend to deploy.

### Hands-on exercise (Hour 2)

1. `pip install sentence-transformers`. Download two open-source models of different sizes (e.g. `all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`).
2. Embed the same 8–10 sentences with each. Note the dimension count and roughly how long each took to load and encode.
3. With an E5 model, embed a document **with** the `passage:` prefix and **without** it, then compare its similarity to a `query:`-prefixed query. Observe how much the prefix matters — this makes the prefix gotcha concrete.
4. Pick a tiny retrieval task (5 questions, 10 candidate passages you wrote). Run it with two different models and eyeball which retrieves better. Write 3–4 sentences on whether the larger/slower model was actually worth it for your data.

### Interview questions — Hour 2

**1. (Conceptual) Why might a team choose a self-hosted open-source embedding model over a commercial API?**
For data privacy (keeping sensitive text on their own infrastructure), to avoid per-token API costs at high volume, for offline or air-gapped operation, and for full control over the model and its versioning. The trade-off is that they must provide and operate the hardware (typically GPUs) and manage deployment themselves.

**2. (Conceptual) Name a few major open-source embedding families and one distinguishing trait of each.**
BGE (BAAI) — multilingual bge-m3 with multi-functionality (dense, sparse, and multi-vector in one model). E5 (Microsoft) — requires `query:`/`passage:` prefixes. GTE (Alibaba) — efficient general-purpose family. Qwen3-Embedding (Alibaba) — instruction-aware, available from 0.6B to 8B, a top performer with strong multilingual support.

**3. (Practical) Is the largest model on a leaderboard always the right pick? Why or why not?**
No. Larger models cost more in memory, latency, and infrastructure, and they don't always win on a given task — smaller models sometimes outperform much larger ones on specific workloads, and dimension count drives storage cost. The right pick balances quality against size, speed, memory, license, and how it performs on your actual data.

**4. (Gotcha) A colleague benchmarks an E5 model, gets poor results, and concludes it's a weak model. What's the likely mistake?**
They probably didn't use E5's required `query:`/`passage:` prefixes. Omitting the expected input format causes a silent quality drop, so the model looks worse than it is. The fix is to read the model card and supply the exact prefixes/instructions the model was trained to expect, then re-benchmark.

**5. (Practical) How does embedding dimension affect a production system beyond model quality?**
Dimension count directly drives vector storage and search cost: higher dimensions mean larger vectors (e.g. ~4 KB for a 1024-dim 32-bit vector), so at millions of documents the storage and memory footprint grows substantially, and search can slow. Techniques like Matryoshka truncation or quantization let you reduce that footprint, trading a little quality for significant savings.

---

## Hour 3 — The MTEB Benchmark

### The problem we're solving

With dozens of models claiming to be "best," you need a common yardstick. **MTEB** — the **Massive Text Embedding Benchmark** — is that yardstick: a standardized suite of tasks and datasets that scores embedding models so they can be compared fairly. Before MTEB, every paper used its own tasks, making comparison nearly impossible. Now essentially every major model reports MTEB scores, and there's a public **leaderboard** (a ranked table) on Hugging Face with thousands of submissions. Its multilingual expansion is called **MMTEB** (Massive *Multilingual* Text Embedding Benchmark), covering 1000+ languages and even image and audio tasks.

### What MTEB actually measures

MTEB evaluates models across several **task types**, because a model good at one isn't automatically good at another. The main ones:

- **Retrieval** — given a query, find relevant documents (the task most relevant to search and RAG).
- **Classification** — assign a label to a text.
- **Clustering** — group similar texts together.
- **Semantic Textual Similarity (STS)** — score how similar two texts are.
- **Reranking** — reorder a candidate list by relevance.
- **Pair classification**, **summarization**, and **bitext mining** (matching sentences across languages).

Each model gets a score per task type plus an **overall average**. **Retrieval is usually the number to watch for RAG/search**, not the headline average.

### How to read the leaderboard without being fooled

This is the real skill of the hour. The leaderboard is useful but easy to misread:

- **The overall average hides task-specific strength.** A model with a great average can be mediocre at retrieval specifically. Filter to the task type (and language, and domain) that matches your use case, and read *that* column.
- **Top scores are often statistically tied.** The gap between the #1 and #5 model is frequently within noise — not a meaningful difference. Don't agonize over tiny score differences.
- **Watch for "trained on the test."** Some models score high partly because they were trained on data resembling the benchmark's. The current leaderboard annotates whether a model saw a task's training data or is **zero-shot** (seeing it fresh) — prefer fair comparisons and be skeptical of suspiciously high outliers, as benchmark **overfitting** (tuning to the test rather than to general ability) is a real problem.
- **The average ignores size, speed, and cost.** A model two points higher but ten times larger may be the wrong choice. The leaderboard offers size-bracket and performance-by-runtime views for exactly this reason — use them.
- **Domain matters.** For specialized corpora (legal, medical, code), look at domain-specific tasks; general leaders often lose to domain-tuned models there.

> **Gotcha — the leaderboard is a shortlist tool, not a decision.** No benchmark captures your documents' style, your queries' phrasing, or your domain's vocabulary. The decisive test is always how a model performs on *your* data. Use MTEB to pick 2–3 candidates, then evaluate them yourself.

> **Gotcha — rankings are volatile and the site has changed.** New submissions constantly reshuffle the order, and the leaderboard itself was rebuilt (the older version is now archived separately). A ranking you screenshot today may differ next month. Re-check before relying on it, and note the date.

### Validating on your own data

You can run MTEB-style evaluation yourself with the `mteb` Python package, and — more importantly — measure quality on *your* corpus using standard retrieval metrics: **recall@k** (of the truly relevant documents, how many appear in your top *k* results), **MRR** (Mean Reciprocal Rank — how high the first relevant result lands), and **NDCG** (a ranking-quality score that rewards putting relevant results near the top).

```python
import mteb

# Evaluate a model on a specific task rather than trusting the overall average
model = mteb.get_model("BAAI/bge-small-en-v1.5")
tasks = mteb.get_tasks(tasks=["Banking77Classification"])  # pick tasks like YOUR use case
results = mteb.evaluate(model, tasks=tasks)
print(results)
```

The far more valuable evaluation, though, is on your own question/answer pairs: take ~50–500 queries where you know the correct document, embed everything with each candidate model, and measure recall@k. An afternoon of this tells you more than any leaderboard.

### Hands-on exercise (Hour 3)

1. Open the Hugging Face MTEB leaderboard. Filter to the **Retrieval** task and to English. Note the top 5 models and how close their scores are.
2. Now filter by a **size bracket** (e.g. small models). Notice how the "best" model changes when size is constrained — write down the best small model and the best overall, and the score gap between them.
3. Look up one model's **model card** and check whether it reports being zero-shot on the tasks, plus its license, dimensions, and context length.
4. **Build a mini gold set:** write 15–20 questions and mark the correct passage for each from a small collection you control. Embed everything with two candidate models and compute **recall@5** for each. Decide which model wins *on your data*, and note whether it matches the leaderboard order.

### Interview questions — Hour 3

**1. (Conceptual) What is MTEB and what problem did it solve?**
MTEB (Massive Text Embedding Benchmark) is a standardized suite of tasks and datasets for scoring and comparing embedding models. It solved the problem that, before it, every paper used different tasks and protocols, making fair comparison between models nearly impossible. Now there's a common yardstick and a public leaderboard.

**2. (Conceptual) Why shouldn't you pick a model based on its overall MTEB average?**
Because the average blends many task types, and a model strong on average can be weak at the specific task you care about (often retrieval, for RAG). It also ignores size, speed, cost, and statistical significance. You should filter to your task type, language, and domain, and weigh practical constraints rather than chasing the top average.

**3. (Practical) You're building legal-document search. How would you use MTEB to choose a model?**
I'd filter the leaderboard to retrieval and, where possible, legal or domain-specific tasks and the relevant language, rather than the overall average — expecting domain-tuned models to outperform general ones on legal text. I'd shortlist 2–3 candidates, check their model cards (license, context length, dimensions, zero-shot status), then run my own recall@k evaluation on real legal queries and documents to make the final call.

**4. (Gotcha) Two models are 0.3 points apart at the top of the retrieval leaderboard. How much should that gap drive your decision?**
Very little on its own — gaps that small among top models are frequently within noise and not statistically meaningful. I'd treat them as roughly tied and decide based on other factors: performance on my own data, size and latency, cost, license, context length, and required input format.

**5. (Gotcha) Why is "validate on your own data" the recurring advice with MTEB?**
Because no benchmark captures the specific style, vocabulary, and query phrasing of your corpus, and leaderboard rankings can be skewed by overfitting to public datasets. A model that tops MTEB may not top *your* task. Running a small recall@k/MRR/NDCG evaluation on your own labeled queries gives decisive, relevant evidence that a general leaderboard can't.

---

## Hour 4 — Fine-Tuning Embeddings

### The problem we're solving

Sometimes no off-the-shelf model is good enough on your specific data — your domain has unusual vocabulary (internal product names, legal or medical jargon, code), or your queries are phrased unlike anything the model trained on. **Fine-tuning** means taking an existing pre-trained model and training it a bit more on *your* data so it adapts to your domain. Done well, it can meaningfully boost retrieval quality where a general model struggles.

### When to fine-tune (and when not to)

Fine-tune when: a general model underperforms on your domain after honest evaluation; you have (or can create) **training pairs** — examples of what *should* match, such as (question, correct passage) pairs, which you may already have in support logs or click data; and the quality gain justifies the effort.

**Don't** fine-tune as a first move. Cheaper fixes usually help more: better chunking, cleaner document text, metadata filtering, adding a reranker, or simply trying a stronger base model. Reach for fine-tuning when those have been tried.

> **Gotcha — you can't fine-tune most commercial API models.** Providers like OpenAI generally don't let you fine-tune their embedding models. To fine-tune, use an open-source model (e.g. a BGE, E5, GTE, or MPNet model) that you control. (Some specialized vendors offer custom-tuned embeddings as a paid service — but the do-it-yourself path runs on open-source models.)

### How fine-tuning works with sentence-transformers

The standard tool is the **`sentence-transformers`** library, whose modern training API centers on the **`SentenceTransformerTrainer`** (introduced in version 3.0 — older `model.fit()` code still works but the trainer is the current, more capable approach). Training has a few components:

- **Dataset** — your training examples, in a format that matches your chosen loss (often pairs of an **anchor** and a **positive**: e.g. a query and the passage that answers it).
- **Loss function** — the formula that defines "good" and steers the training. The choice depends on your data shape. The most common for retrieval is **MultipleNegativesRankingLoss**, which works from (anchor, positive) pairs and treats the *other* examples in the batch as negatives — it just needs examples of things that *should* match. (Other losses exist for pairs-with-scores, triplets, etc.)
- **Training arguments** (optional) — settings like number of passes over the data (**epochs**), batch size, and learning rate, via `SentenceTransformerTrainingArguments`.
- **Evaluator** (optional) — measures quality during/after training, e.g. an information-retrieval evaluator that reports recall and NDCG.
- **Trainer** — ties it together and runs the training.

```python
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

# 1) Start from an open-source base model (you can't fine-tune most API models)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 2) Training data: (anchor, positive) pairs — things that SHOULD match
train_dataset = Dataset.from_dict({
    "anchor":   ["how do I get a refund?", "what time do you open?"],
    "positive": ["Refunds are issued within 5 business days.",
                 "Our stores open at 9am daily."],
})

# 3) Loss suited to (anchor, positive) pairs; uses in-batch negatives
loss = MultipleNegativesRankingLoss(model)

# 4) Training settings
args = SentenceTransformerTrainingArguments(
    output_dir="./finetuned-embedding-model",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
)

# 5) Train, then save
trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_dataset, loss=loss,
)
trainer.train()
model.save_pretrained("./finetuned-embedding-model")
```

Encouragingly, fine-tuning an embedding model is often *cheap and fast* — a small dataset can fine-tune in around a minute on a modest rented GPU for pennies. The hard part is good data and honest evaluation, not compute.

> **Gotcha — fine-tuning changes the vector space, so you must re-embed everything.** A fine-tuned model produces vectors incompatible with the base model's. After fine-tuning you have to re-embed your entire corpus with the new model; you cannot mix fine-tuned and original vectors in the same index.

> **Gotcha — you need a held-out evaluation set, or you're flying blind.** Without measuring retrieval quality on data the model didn't train on, you can't tell whether fine-tuning helped, did nothing, or **overfit** (memorized the training examples while getting worse on real queries). Always reserve evaluation queries and compare the fine-tuned model against the base model on them.

> **Gotcha — match your data format to the loss.** Each loss expects a specific data shape (anchor/positive pairs, pairs-with-similarity-scores, triplets, etc.). Feeding the wrong shape errors or trains poorly. Pick the loss that fits the data you actually have.

### Hands-on exercise (Hour 4)

1. Create a small training set of 30–100 (anchor, positive) pairs from a domain you know — e.g. questions paired with the passage that answers each.
2. Hold out 10–20 pairs as an **evaluation set** (do not train on these).
3. Record the **base** model's recall@k on your evaluation queries *before* fine-tuning.
4. Fine-tune `all-MiniLM-L6-v2` (or another open base) on the training pairs using `MultipleNegativesRankingLoss` and the `SentenceTransformerTrainer`, as above.
5. Re-measure recall@k on the same held-out queries with the fine-tuned model. Compare to the baseline and write 3–4 sentences: did it help, by how much, and do you see any sign of overfitting (great on training-like queries, no better or worse on the held-out set)?

### Interview questions — Hour 4

**1. (Conceptual) What does it mean to fine-tune an embedding model, and when is it appropriate?**
Fine-tuning takes a pre-trained model and trains it further on your own data so it adapts to your domain's vocabulary and query style. It's appropriate when a general model underperforms on your data after evaluation, you have training pairs of what should match, and the expected quality gain justifies the effort — typically after cheaper fixes (better chunking, a reranker, a stronger base model) have been tried.

**2. (Practical) You want to fine-tune embeddings for your domain. Can you fine-tune OpenAI's model? If not, what do you do?**
Generally no — commercial providers like OpenAI don't expose their embedding models for fine-tuning. Instead, fine-tune an open-source model you control (a BGE, E5, GTE, or MPNet model) using a library like sentence-transformers, or use a vendor that offers custom-tuned embeddings as a service.

**3. (Conceptual) In sentence-transformers training, what role does the loss function play, and why is MultipleNegativesRankingLoss popular for retrieval?**
The loss function defines what "good" means and steers the optimization; the right choice depends on your data shape. MultipleNegativesRankingLoss is popular because it only needs (anchor, positive) pairs — examples that should match — and uses the other items in the batch as negatives, which fits common retrieval data (like query–answer pairs) without needing explicit negative examples.

**4. (Gotcha) After fine-tuning, your old search index suddenly returns poor results. What happened?**
Fine-tuning changed the model's vector space, so its vectors are no longer compatible with the ones in your existing index (which were made by the base model). You must re-embed the entire corpus with the fine-tuned model and rebuild the index; mixing fine-tuned and original vectors produces meaningless comparisons.

**5. (Gotcha) How do you know whether fine-tuning actually improved your embeddings?**
By evaluating on a held-out set of queries the model didn't train on, using retrieval metrics like recall@k, MRR, or NDCG, and comparing against the base model's scores on the same set. Without that held-out evaluation you can't distinguish real improvement from overfitting, where the model memorizes training examples but doesn't generalize to real queries.

---

## Gotchas Summary Table

| # | Gotcha | Where | Why it bites | How to avoid it |
|---|--------|-------|--------------|-----------------|
| 1 | Forgetting input types / prefixes / instructions | Providers & open-source | Silent retrieval-quality drop, no error | Check the model's expected input format; set `input_type` or prefixes (`query:`/`passage:`) / instructions |
| 2 | Mixing vectors from different models | Everywhere | Different vector spaces → meaningless comparisons | Use one consistent model; re-embed everything when switching |
| 3 | Silent truncation of long inputs | Providers | Text beyond the token limit is dropped | Chunk documents before embedding; respect the token limit |
| 4 | Assuming bigger = better | Open-source / MTEB | Larger models cost more and don't always win on your task | Weigh size, speed, cost, license; validate on your data |
| 5 | Ignoring license terms | Open-source | "Open" ≠ free for any use | Confirm the model card's license fits your deployment |
| 6 | Trusting the MTEB overall average | MTEB | Average hides task-specific (e.g. retrieval) strength | Filter to your task type, language, domain; read that column |
| 7 | Over-reading tiny leaderboard gaps | MTEB | Top models are often statistically tied | Treat small gaps as ties; decide on other factors |
| 8 | Benchmark overfitting / "trained on test" | MTEB | Inflated scores from training on benchmark-like data | Prefer zero-shot annotations; be skeptical of outliers |
| 9 | Skipping validation on your own data | MTEB / all | No benchmark matches your corpus and queries | Build a small gold set; measure recall@k / MRR / NDCG yourself |
| 10 | Trying to fine-tune an API model | Fine-tuning | Most providers don't allow it | Fine-tune an open-source model you control |
| 11 | Not re-embedding after fine-tuning | Fine-tuning | Fine-tuned vectors are incompatible with base-model vectors | Re-embed the whole corpus with the fine-tuned model |
| 12 | No held-out evaluation set | Fine-tuning | Can't tell improvement from overfitting | Reserve eval queries; compare fine-tuned vs. base |
| 13 | Wrong data format for the loss | Fine-tuning | Errors or poor training | Match the data shape to the chosen loss |

---

## Quick Reference Card

**The one rule:** a vector is only comparable to other vectors from the **same model**. Switching models ⇒ re-embed everything.

**Commercial providers (verify current models/pricing)**
- **OpenAI** — `text-embedding-3-small` (1536d, cheap default), `text-embedding-3-large` (3072d). Matryoshka `dimensions` param, normalized output, ~8k token limit, batch ≈ 50% off, **no** input type.
- **Cohere** — `embed-v4.0`: multimodal (text + images), needs `input_type` (`search_document`/`search_query`), dims 256–1536, quantization, ~128k context, 100+ languages.
- **Voyage AI** — quality-focused; general (`voyage-4`/`voyage-3.5`) + **domain** models (`voyage-code-3`, `voyage-law-2`, `voyage-finance-2`); needs `input_type` (`query`/`document`); compatible *within* a generation.

**Open-source families (run via `sentence-transformers`)**
- **BGE** (BAAI) — `bge-m3` multilingual, dense+sparse+multi-vector, MIT.
- **E5** (Microsoft) — needs `query:`/`passage:` prefixes; multilingual-e5-large (1024d).
- **GTE** (Alibaba) — efficient general-purpose.
- **Qwen3-Embedding** (Alibaba) — 0.6B/4B/8B, instruction-aware, top performer, Apache 2.0.
- Lightweight: `all-MiniLM-L6-v2` (384d, fast, older/weaker). On-device: EmbeddingGemma.

**MTEB (the benchmark)**
- Tasks: retrieval (watch this for RAG), classification, clustering, STS, reranking, etc. MMTEB = multilingual.
- Read it right: filter to **your task/language/domain**; ignore tiny gaps (often tied); check zero-shot status & license; use size/runtime views; **always validate on your own data** (recall@k, MRR, NDCG).
- Leaderboard: huggingface.co/spaces/mteb/leaderboard (volatile; recently rebuilt).

**Fine-tuning (sentence-transformers, v3+)**
```python
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                    SentenceTransformerTrainingArguments)
from sentence_transformers.losses import MultipleNegativesRankingLoss
model = SentenceTransformer("open-source-base-model")
loss  = MultipleNegativesRankingLoss(model)       # needs (anchor, positive) pairs
args  = SentenceTransformerTrainingArguments(output_dir="out", num_train_epochs=1)
SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
```
- Fine-tune **open-source** models (not API models). Cheap/fast; hard part is data + evaluation.
- After fine-tuning: **re-embed everything**; keep a **held-out eval set**; match data format to the loss.
- Try cheaper fixes first (chunking, reranker, stronger base model).

**Reference docs:** each provider's own embedding docs (OpenAI, Cohere, Voyage AI); Hugging Face model cards for open-source families; the MTEB leaderboard on Hugging Face; the Sentence Transformers training documentation.

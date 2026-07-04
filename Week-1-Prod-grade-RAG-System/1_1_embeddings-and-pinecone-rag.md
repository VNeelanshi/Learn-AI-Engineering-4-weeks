# Embeddings and Pinecone: From Vectors to Your First RAG System

A complete, beginner-friendly walkthrough. By the end you'll understand what embeddings are, how a vector database stores and searches them, how to make search both *meaning-aware* and *keyword-aware*, and you'll have built a small working question-answering system over your own documents.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order — each hour builds on the one before.

> **A note on running the code:** vector database APIs change often. The Pinecone code here uses the current instance-based SDK (the `pinecone` package). If something doesn't match what you see, the official Quickstart is the source of truth — always trust the live docs over any tutorial when versions disagree.

---

## Hour 1 — Embeddings Refresher

### The problem we're solving

Computers don't understand text. To a computer, the words "car" and "automobile" are just different sequences of characters — as unrelated as "car" and "banana." But humans know "car" and "automobile" mean almost the same thing.

We need a way to turn text into something a computer *can* compare, where similar meanings end up "close together" in some measurable sense. That representation is called an **embedding**.

### What is an embedding?

An **embedding** is a list of numbers (a **vector**) that represents the meaning of a piece of text (or an image, or audio). A **vector** is just an ordered list of numbers, like `[0.12, -0.45, 0.88, ...]`.

The key property: text with similar meaning produces vectors that are numerically close to each other, and text with different meaning produces vectors that are far apart. An **embedding model** is the trained program that converts text into these vectors. You feed it a sentence, it hands you back a vector.

Each number in the vector is called a **dimension**. A model might produce vectors of 384, 768, or 1536 dimensions. You don't choose this number — it's fixed by whichever model you use. You can't picture 384 dimensions, and you don't need to. The intuition from 2D and 3D (points that are near each other are "similar") carries over directly to hundreds of dimensions.

> **Gotcha — dimensions are model-specific and fixed.** A vector from one model cannot be meaningfully compared to a vector from a different model, *even if they happen to have the same number of dimensions*. Different models place meaning in different regions of vector space. If you ever switch embedding models, you must re-embed your entire dataset with the new model. Mixing them produces nonsense results that often *look* fine (no error is raised) but are silently wrong.

### Measuring similarity: the core idea

Once two pieces of text are vectors, "how similar are they?" becomes "how close are these two vectors?" There are two common ways to measure this, and the difference matters.

#### Dot product

The **dot product** multiplies the two vectors element by element and sums the results. For vectors `a = [a1, a2, ...]` and `b = [b1, b2, ...]`:

```
dot(a, b) = a1*b1 + a2*b2 + a3*b3 + ...
```

A larger dot product means "more similar." The catch: the dot product grows with the **magnitude** (length) of the vectors, not just their direction. So a longer vector can score a high dot product just for being long, even if its *direction* (which is what actually encodes meaning) isn't a great match. Magnitude is the geometric length of the vector — how far the arrow reaches from the origin, regardless of which way it points.

#### Cosine similarity

**Cosine similarity** measures only the *angle* between two vectors, ignoring their length entirely. It ranges from `-1` (pointing in opposite directions, i.e. opposite meaning) through `0` (unrelated) to `1` (pointing the same way, i.e. same meaning).

Cosine similarity is the dot product *after* dividing out the magnitudes:

```
cosine(a, b) = dot(a, b) / (length(a) * length(b))
```

Because it ignores length, cosine is usually the safer default for text similarity — you care whether two sentences *mean* the same thing, not how long they are.

#### Normalization: the bridge between the two

**Normalization** means scaling a vector so its length becomes exactly 1, while keeping its direction unchanged. A vector of length 1 is called a **unit vector**.

Here's the useful fact that ties everything together: **if both vectors are normalized to length 1, then the dot product and cosine similarity give the exact same number.** The division by magnitudes in the cosine formula becomes division by 1.

This matters practically because dot product is cheaper to compute than cosine. Many systems normalize all vectors once up front, then use plain dot product everywhere and get cosine behavior for free.

> **Gotcha — know whether your model already normalizes.** Some embedding models return normalized (unit-length) vectors by default; others don't. If you assume yours is normalized when it isn't, your "cosine = dot product" shortcut breaks and your similarity scores drift. Always check, and normalize explicitly if you're unsure.

> **Gotcha — match your similarity metric to how the model was trained.** Most modern text embedding models are trained to be compared with cosine similarity. Use that unless the model's documentation tells you otherwise. Picking the wrong metric quietly degrades result quality without any error message.

### Hands-on exercise (Hour 1)

Install the Sentence Transformers library and generate some embeddings yourself.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# A small, fast, popular model that outputs 384-dimensional vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",   # similar meaning to #1
    "Quarterly revenue grew by 12%." # unrelated meaning
]

# normalize_embeddings=True makes dot product equal cosine similarity
embeddings = model.encode(sentences, normalize_embeddings=True)

print("Vector length (dimensions):", len(embeddings[0]))

def cosine(a, b):
    return np.dot(a, b)  # already normalized, so dot == cosine

print("similar pair:  ", round(cosine(embeddings[0], embeddings[1]), 3))
print("unrelated pair:", round(cosine(embeddings[0], embeddings[2]), 3))
```

**Your tasks:**
1. Confirm the "similar pair" scores much higher than the "unrelated pair."
2. Re-run with `normalize_embeddings=False` and compute cosine *manually* (divide by the product of the two vector lengths using `np.linalg.norm`). Confirm you get the same numbers as the normalized version.
3. Add two sentences of your own — one a paraphrase of an existing sentence, one totally unrelated — and check the scores match your intuition.

### Interview questions — Hour 1

**1. (Conceptual) Explain what an embedding is in one or two sentences.**
An embedding is a vector (a list of numbers) that represents the meaning of an input such as text, produced by a trained embedding model. Inputs with similar meaning produce vectors that are numerically close, which lets a computer compare meaning by comparing numbers.

**2. (Conceptual) What's the difference between cosine similarity and dot product, and why does it matter?**
Dot product sums the element-wise products of two vectors and is influenced by both their direction and their magnitude (length). Cosine similarity measures only the angle between vectors, ignoring magnitude. It matters because for text you usually care about direction (meaning), not length; a long vector can inflate a dot product without being a better semantic match.

**3. (Practical) When would normalizing your vectors be useful?**
When you want cosine behavior but the cheaper dot-product computation, normalize all vectors to length 1 up front. After normalization, dot product equals cosine similarity, so you get correct semantic ranking at lower compute cost. It's also useful for consistency when a downstream system expects unit vectors.

**4. (Gotcha) Two vectors both have 768 dimensions but came from two different embedding models. Can you compare them with cosine similarity?**
No — not meaningfully. Matching dimension counts is necessary but not sufficient. Different models arrange meaning differently in their vector space, so a cosine score between vectors from different models is essentially noise. You'd need to re-embed both inputs with the *same* model.

**5. (Gotcha) Your similarity scores look "off" even though your code runs without errors. What are the first things you'd check?**
Whether the model's expected metric matches the one you're using (most expect cosine); whether the vectors are normalized when your math assumes they are; and whether every vector was produced by the *same* model. None of these raise errors, so silent quality degradation is the usual symptom.

---

## Hour 2 — Pinecone Basics

### The problem we're solving

In Hour 1 you compared a handful of vectors with a Python loop. That works for three sentences. It does not work for a million documents: comparing a query against a million vectors one by one, every time someone searches, is far too slow.

A **vector database** is a system built to store huge numbers of vectors and find the most similar ones to a query *quickly* — in milliseconds, not minutes. **Pinecone** is a managed (hosted-for-you) vector database. "Managed" means you don't run any servers yourself; Pinecone handles the infrastructure and you talk to it over the internet through its API.

> **Important context — the SDK was rewritten.** Older Pinecone tutorials use `pinecone.init(...)`. That style no longer works. The current SDK is *instance-based*: you create a `Pinecone(...)` client object and call methods on it. Also, the package is now named `pinecone` — if you have an old `pinecone-client` package installed, remove it to avoid confusing import errors.

### The core concepts

**Index.** An **index** is the top-level container that holds your vectors. Every index has a fixed **dimension** (set when you create it) and a fixed **metric** (the similarity measure, e.g. `cosine`). All vectors you store in it must have that exact dimension. You typically create one index per use case.

**Namespace.** A **namespace** is a partition *inside* an index — a way to keep separate groups of vectors apart within the same index. A query runs against one namespace and never sees vectors in other namespaces. This is handy for separating customers, projects, or environments. Namespaces are created automatically the first time you write to them, and they don't cost extra.

**Upsert.** **Upsert** = "update or insert." When you send a vector with a given ID, Pinecone inserts it if that ID is new, or overwrites it if that ID already exists. This is how you load data in.

**Query.** A **query** sends one vector (your search) and asks the index for the `top_k` most similar stored vectors — where `top_k` is just "how many results you want back," e.g. the top 5.

**Metadata.** **Metadata** is extra information you attach to each vector as key-value pairs — for example the original text, a source filename, or a date. You can then **filter** queries by metadata (e.g. "only search vectors where `source = 'handbook'`").

### Putting it together in code

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# 1) Create an index. dimension MUST match your embedding model (384 here).
pc.create_index(
    name="quickstart",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

# 2) Get a handle to the index
index = pc.Index("quickstart")

# 3) Upsert a few vectors (id, values, metadata)
index.upsert(
    vectors=[
        ("doc1", [0.10, 0.22, 0.31, "..."], {"text": "Refunds take 5 business days.", "source": "handbook"}),
        ("doc2", [0.05, 0.19, 0.27, "..."], {"text": "Office hours are 9am to 5pm.",  "source": "handbook"}),
    ],
    namespace="support-docs",
)

# 4) Query with a search vector, filtered by metadata
results = index.query(
    vector=[0.11, 0.20, 0.30, "..."],
    top_k=3,
    include_metadata=True,           # return the metadata, not just IDs
    namespace="support-docs",
    filter={"source": {"$eq": "handbook"}},  # $eq means "equals"
)

for match in results["matches"]:
    print(round(match["score"], 3), match["metadata"]["text"])
```

A note on those filter operators: `$eq` (equals), `$ne` (not equal), `$gt` / `$gte` / `$lt` / `$lte` (greater/less than), `$in` (value is in a list). You combine them in the `filter` dictionary.

> **Gotcha #1 — dimension mismatch is the #1 beginner error.** If your index is 384-dimensional and you upsert a 1536-dimensional vector (or vice versa), it fails. Always confirm your model's output length matches the index's `dimension`. You can check an existing index with `pc.describe_index("quickstart").dimension`.

> **Gotcha #2 — namespace drift causes "no results despite successful upserts."** If you *write* to namespace `support-docs` but *query* the default namespace (by forgetting the `namespace=` argument), you'll get zero matches and no error. This is the single most common cause of "I added data but search returns nothing." Always use the same namespace for writing and reading.

> **Gotcha #3 — metadata has a size limit (about 40 KB per vector).** Don't dump the full text of long documents into metadata. Store a short snippet or a reference ID, and keep the bulky original text in your own database or file store. Oversized metadata causes upserts to fail at scale.

> **Gotcha #4 — index creation is asynchronous.** Asking Pinecone to create an index queues the job; the index isn't instantly ready. If you upsert immediately you may get an error. In real scripts, wait until `pc.describe_index(name)` reports the index is ready before writing to it.

### Hands-on exercise (Hour 2)

Combine Hour 1 and Hour 2 into a tiny searchable store.

1. Sign up for a free Pinecone account and get an API key.
2. Use `all-MiniLM-L6-v2` (384 dims) from Hour 1 to embed 5–10 short factual sentences of your choice.
3. Create a 384-dimensional, `cosine` index and upsert your sentences. Put the original sentence in each vector's metadata under a `text` key, and add a `category` metadata field with at least two different values.
4. Embed a question related to one of your sentences, query with `top_k=3`, and print the results with their scores.
5. Re-run the query with a metadata `filter` that restricts to one `category`. Confirm the results change.
6. **Deliberately break it:** query *without* the `namespace` argument you used for upserting, and observe that you get zero results. Now you've seen namespace drift firsthand.

### Interview questions — Hour 2

**1. (Conceptual) Why use a dedicated vector database instead of just looping over vectors in your own code?**
Brute-force comparison against every stored vector is too slow once you have many vectors and frequent queries. Vector databases use specialized indexing structures to return the most similar vectors in milliseconds at scale, and they handle storage, persistence, filtering, and scaling for you.

**2. (Conceptual) What's the difference between an index and a namespace?**
An index is the top-level container with a fixed dimension and metric. A namespace is a partition within a single index that isolates a subset of vectors; queries run against one namespace and don't see others. Namespaces are a lightweight way to separate data (e.g. per customer) without creating separate indexes.

**3. (Practical) When would you use metadata filtering?**
When you want semantic search *constrained* to a subset — for example, searching only documents from a particular source, language, date range, or user. The filter narrows the candidate set so results are both semantically relevant and contextually appropriate.

**4. (Gotcha) A teammate says "I upserted my data successfully but queries return nothing." What's your first hypothesis?**
Namespace mismatch: they likely wrote to one namespace and are querying another (often the default namespace because they omitted the `namespace` argument). The fix is to query the same namespace they wrote to. A dimension or metric problem would usually raise an error, whereas namespace drift fails silently.

**5. (Gotcha) Why shouldn't you store the full text of large documents in vector metadata?**
There's a per-vector metadata size limit (around 40 KB). Large documents blow past it and cause failures. Best practice is to store a short snippet or an ID in metadata and keep the full text in a separate store you look up after retrieval.

---

## Hour 3 — Pinecone Advanced: Hybrid Search, Sparse Vectors, and Serverless

### The problem we're solving

Embedding-based search (everything so far) is **semantic search** — it matches *meaning*. That's powerful, but it has a weak spot: exact strings with little semantic content. Think product codes (`SKU-4471`), error codes (`ERR_503`), function names (`upsert_records`), or rare proper nouns. A semantic model may not place these where you'd expect, so a meaning-based search can miss an exact term the user clearly typed on purpose.

Old-fashioned **keyword search** (also called **lexical search**) is great at exactly this: it matches the literal words. The idea behind **hybrid search** is to combine both — semantic search for meaning *plus* keyword search for exact terms — so you get the strengths of each.

### Dense vs. sparse vectors

The embeddings from Hour 1 are **dense vectors**: every one of their hundreds of dimensions holds a meaningful number. "Dense" = mostly non-zero.

A **sparse vector** is the opposite: a very high-dimensional vector that is almost entirely zeros, with non-zero values only at the positions corresponding to words that actually appear. Conceptually, each position represents a word in the vocabulary, and the value represents that word's importance. Sparse vectors are how keyword/lexical signals get represented in the same "vector" framework. Modern sparse vectors (produced by trained models in the spirit of the classic keyword-scoring method BM25, which you'll meet in depth later) act like a *learned* version of keyword matching — they weight tokens by relevance rather than just counting them.

So: **dense vector = meaning, sparse vector = keywords.** Hybrid search uses both for each record.

### How hybrid search works in Pinecone

There are two architectures, and the recommendation has shifted over time, so here's the current picture:

- **Single index (recommended for most use cases).** One index stores both a dense and a sparse vector for each record. You query both together in a single request. Simpler: one index to manage, the dense↔sparse link is implicit, and you get results in one round trip.
- **Separate indexes (more flexible).** One dense-only index and one sparse-only index, queried separately and merged in your own code. More moving parts and you maintain the linkage yourself, but it gives finer control.

A weighting knob commonly called **alpha** lets you tune how much the dense (semantic) signal counts versus the sparse (keyword) signal.

> **Gotcha — hybrid requires the `dotproduct` metric. This is non-negotiable.** An index using the `cosine` metric does **not** support querying sparse-dense vectors. If you want hybrid search on a single index, you must create it with `metric="dotproduct"`. You can sometimes upsert sparse values into a cosine index without error, but the *query* will fail — a confusing failure mode if you didn't know the rule.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Hybrid REQUIRES dotproduct
pc.create_index(
    name="hybrid-index",
    dimension=384,
    metric="dotproduct",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
index = pc.Index("hybrid-index")

# Each record carries BOTH a dense vector and a sparse vector.
# A sparse vector is given as indices (which positions are non-zero)
# and values (the weight at each of those positions).
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": [0.10, 0.22, "..."],                       # dense (meaning)
        "sparse_values": {"indices": [12, 88, 304], "values": [0.7, 0.4, 0.9]},  # keywords
        "metadata": {"text": "Reset code ERR_503 by restarting the service."},
    },
])

# Query with both a dense and a sparse vector
results = index.query(
    top_k=5,
    vector=[0.11, 0.20, "..."],                              # dense query
    sparse_vector={"indices": [88, 304], "values": [0.5, 0.8]},  # keyword query
    include_metadata=True,
)
```

In practice you don't hand-build sparse vectors; you use a sparse-embedding model or Pinecone's built-in capabilities to generate them. Pinecone also now offers fully managed sparse indexes and built-in (integrated) embedding so it can generate vectors for you — useful once you're comfortable with the basics.

### The serverless tier

**Serverless** means the database automatically scales its compute up and down based on your workload, and you pay per use rather than reserving (and paying for) a fixed amount of capacity in advance. Architecturally, serverless **separates compute from storage**, so each can scale independently. For a beginner this is the easiest and most cost-effective starting point: no capacity planning, no idle servers to pay for.

> **Gotcha — serverless cold starts.** When an index has been idle, the first query after the lull can be noticeably slower (a couple of seconds) while it "warms up." Later queries are fast. If you see occasional slow first-queries but consistently fast ones afterward, this is expected behavior, not a bug.

### Hands-on exercise (Hour 3)

You don't need to perfect sparse embeddings to learn the trade-off. The goal here is to *feel* where semantic search alone falls short.

1. Take the index from Hour 2 (semantic only). Add 3–4 sentences that each contain a distinctive exact token — a fake product code, an error code, or an unusual ID (e.g. "Activate plan with code `ZX-9920`.").
2. Query semantically for the *meaning* (e.g. "how do I turn on my subscription") and note whether the code-bearing sentence ranks well.
3. Now query using the exact code string itself (e.g. "`ZX-9920`") and observe how semantic-only search handles a bare identifier — often poorly.
4. Write 3–4 sentences explaining, in your own words, *why* a hybrid (dense + sparse) approach would help your code-query case. Note that to actually run hybrid you'd recreate the index with `metric="dotproduct"` and attach sparse vectors — sketch what you'd change in your code.

### Interview questions — Hour 3

**1. (Conceptual) What is hybrid search and what weakness of semantic search does it address?**
Hybrid search combines semantic (meaning-based) retrieval with lexical (keyword-based) retrieval. It addresses semantic search's weakness on exact strings — product codes, error codes, identifiers, function names — where matching meaning isn't enough and the literal token matters.

**2. (Conceptual) Explain the difference between a dense vector and a sparse vector.**
A dense vector has meaningful values across all of its (relatively few hundred) dimensions and encodes semantic meaning. A sparse vector is very high-dimensional but almost entirely zeros, with non-zero weights only at positions for words that appear — it encodes lexical/keyword signal, like a learned form of keyword scoring.

**3. (Practical) When is hybrid search worth the extra complexity over plain semantic search?**
When your content is full of exact terms users will search for verbatim — technical documentation with API names and error codes, catalogs with SKUs, legal text with citations. If your queries are purely conceptual natural language, plain semantic search may be enough and simpler.

**4. (Gotcha) You set up sparse-dense hybrid search but your queries error out. What's the most likely cause?**
The index metric. Hybrid sparse-dense querying requires the `dotproduct` metric; a `cosine` index won't support it. Upserts may even appear to succeed, but the query fails. Recreate the index with `metric="dotproduct"`.

**5. (Gotcha/Practical) Why might the very first query to a serverless index be slow, and what should you conclude from it?**
Serverless indexes scale compute down when idle, so the first query after inactivity incurs a cold-start delay of a second or two while it warms. Subsequent queries are fast. You should conclude it's normal serverless behavior, not a performance regression — distinguishable because median latency stays low while only the occasional first-query spikes.

---

## Hour 4 — Build a Basic RAG System

### The problem we're solving

Large language models (LLMs) — the AI systems that generate human-like text, like the model answering you in a chat app — have two relevant limitations. First, they only "know" what was in their training data, so they're unaware of your private documents or anything recent. Second, when asked about something they don't know, they may **hallucinate** — produce a confident but false answer.

**RAG** stands for **Retrieval-Augmented Generation**. The idea: before the LLM answers, *retrieve* the relevant pieces of your documents (using vector search) and hand them to the LLM as context, so it generates its answer *from your actual material* rather than from memory alone. "Augmented" = we augment the model's input with retrieved facts.

This is exactly why everything so far matters: RAG = embeddings (Hour 1) + a vector database to search them (Hours 2–3) + an LLM to write the final answer.

### The RAG pipeline, step by step

A basic RAG system has two phases.

**Indexing (done once, ahead of time):**
1. **Chunk** your documents — split them into smaller passages. A **chunk** is a bite-sized piece of text (say, a few sentences to a paragraph). We chunk because whole documents are too big to embed well or to fit into an LLM's input, and because retrieving a focused passage beats retrieving an entire document.
2. **Embed** each chunk into a vector (Hour 1).
3. **Upsert** the vectors into Pinecone, storing the chunk's text in metadata (Hour 2).

**Querying (done every time a user asks something):**
4. **Embed the user's question** with the *same* model used for the chunks.
5. **Retrieve** the top-k most similar chunks from Pinecone.
6. **Generate**: put those chunks into a prompt as context, ask the LLM to answer using only that context, and return its answer.

> **Gotcha — the question and the chunks must use the same embedding model.** If you embed chunks with one model and questions with another, their vectors live in different spaces and retrieval breaks (silently — no error). One model, both phases.

### A complete minimal example

```python
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ---------- Setup ----------
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dims, used EVERYWHERE
pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="rag-demo",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
index = pc.Index("rag-demo")

# ---------- Phase 1: Indexing ----------
documents = [
    "Our return policy allows refunds within 30 days of purchase.",
    "Standard shipping takes 5 to 7 business days within the country.",
    "Premium members get free expedited shipping on all orders.",
    "Customer support is available Monday through Friday, 9am to 5pm.",
]

# Simple fixed chunking: here each document is already short enough to be one chunk.
chunks = documents

# Embed all chunks (same model)
chunk_vectors = model.encode(chunks, normalize_embeddings=True)

# Upsert: store the chunk text in metadata so we can read it back after retrieval
index.upsert(
    vectors=[
        (f"chunk-{i}", chunk_vectors[i].tolist(), {"text": chunks[i]})
        for i in range(len(chunks))
    ],
    namespace="kb",
)

# ---------- Phase 2: Querying ----------
def retrieve(question, k=3):
    q_vec = model.encode(question, normalize_embeddings=True).tolist()  # SAME model
    res = index.query(vector=q_vec, top_k=k, include_metadata=True, namespace="kb")
    return [m["metadata"]["text"] for m in res["matches"]]

def build_prompt(question, context_chunks):
    context = "\n".join(f"- {c}" for c in context_chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

question = "How long do I have to return something?"
context_chunks = retrieve(question)
prompt = build_prompt(question, context_chunks)

print(prompt)
# Send `prompt` to your LLM of choice (e.g. an OpenAI or Anthropic API call)
# and print the model's response. The retrieved context grounds the answer.
```

The final step sends `prompt` to an LLM. Any chat model API works — you pass the prompt and read back the generated text. The crucial part is already done: the model now sees your actual return-policy chunk and can answer from it instead of guessing.

> **Gotcha — instruct the model to stick to the context.** Notice the prompt explicitly says to answer *only* from the context and to admit when it doesn't know. Without that instruction, the model may "fill in" from its training data and reintroduce the hallucination problem RAG is meant to solve.

> **Gotcha — retrieval quality caps everything.** If step 5 retrieves the wrong chunks, no amount of clever prompting saves the answer — the model simply doesn't have the right facts in front of it. When a RAG answer is bad, check *what was retrieved* first, before blaming the LLM. Most RAG quality problems are retrieval problems.

> **Gotcha — chunk size is a real trade-off.** Chunks too large dilute relevance and may overflow the LLM's input; chunks too small lose the surrounding context needed to make sense of a passage. The simple one-document-per-chunk approach here works for tiny demos; real corpora need a deliberate chunking strategy (a topic worth studying on its own).

### Hands-on exercise (Hour 4)

Build an end-to-end RAG system over a small document set of your own.

1. Gather 8–15 short factual passages on a topic you know (notes, an FAQ, product descriptions — anything). Keep each to a few sentences.
2. Run the full indexing phase: embed with `all-MiniLM-L6-v2`, upsert into a fresh 384-dim `cosine` index, storing each passage's text in metadata.
3. Write the `retrieve` and `build_prompt` functions as above.
4. Connect any LLM you have access to and complete the loop: ask a question, retrieve, build the prompt, get an answer.
5. **Test the failure modes deliberately:**
   - Ask something your documents *do* cover — confirm a grounded answer.
   - Ask something they *don't* cover — confirm the model says it doesn't know (this proves your "use only the context" instruction is working).
   - Print the retrieved chunks for each question and judge whether retrieval picked the right passages. When an answer is wrong, diagnose whether the failure was in *retrieval* (wrong chunks) or *generation* (right chunks, bad answer).

### Interview questions — Hour 4

**1. (Conceptual) What is RAG and what two problems does it solve?**
Retrieval-Augmented Generation retrieves relevant documents and supplies them to an LLM as context before it answers. It addresses the LLM's lack of knowledge about private or recent information (by injecting your own documents) and reduces hallucination (by grounding answers in retrieved facts rather than the model's memory).

**2. (Conceptual) Walk through the steps of a basic RAG pipeline.**
Indexing phase: chunk the documents, embed each chunk, upsert the vectors into a vector database with the text in metadata. Query phase: embed the user's question with the same model, retrieve the top-k most similar chunks, place them in a prompt as context, and have the LLM generate an answer grounded in that context.

**3. (Practical) Why do we chunk documents instead of embedding whole files?**
Whole documents are often too large to embed effectively and too large to fit in an LLM's input. Smaller chunks produce more focused embeddings and let retrieval return the specific passage that answers the question, improving both relevance and the quality of the context handed to the model.

**4. (Gotcha) A RAG system gives a wrong answer. Where do you look first, and why?**
Look at what was retrieved before blaming the LLM. If the retrieved chunks don't contain the answer, the model can't possibly answer correctly — it's a retrieval problem (wrong embedding model on one side, poor chunking, wrong namespace, too-low top_k). Most RAG failures are retrieval failures, so inspecting retrieved context first is the fastest path to the root cause.

**5. (Gotcha) Why must the question and the document chunks be embedded with the same model, and what happens if they aren't?**
Different models map text into different vector spaces, so vectors from different models aren't comparable. If you embed chunks with one model and questions with another, similarity scores become meaningless and retrieval returns irrelevant results — with no error raised, so the failure is silent and easy to miss.

---

## Gotchas Summary Table

| # | Gotcha | Why it bites | How to avoid it |
|---|--------|--------------|-----------------|
| 1 | Comparing vectors from different embedding models | Different models use different vector spaces; matching dimension counts doesn't make them comparable | Use one model everywhere; re-embed everything if you switch models |
| 2 | Assuming vectors are normalized when they aren't | Breaks the "dot product = cosine" shortcut; scores drift | Check your model's behavior; normalize explicitly if unsure |
| 3 | Wrong similarity metric for the model | Silently degrades result quality, no error | Default to cosine unless the model's docs say otherwise |
| 4 | Old `pinecone.init()` SDK style | The SDK was rewritten; old code throws import/attribute errors | Use the instance-based `Pinecone(...)` client; install the `pinecone` package, remove `pinecone-client` |
| 5 | Dimension mismatch between index and vectors | Upserts/queries fail | Make index `dimension` equal the model's output length; check with `describe_index` |
| 6 | Namespace drift (write one namespace, query another) | Zero results, *no error* — looks like data vanished | Use the same `namespace` for upsert and query |
| 7 | Oversized metadata (full document text) | Hits the ~40 KB per-vector limit; upserts fail at scale | Store snippets or IDs in metadata; keep full text elsewhere |
| 8 | Upserting before an index finishes creating | Index creation is async; early writes error | Wait until `describe_index` reports ready |
| 9 | Hybrid search on a `cosine` index | Sparse-dense querying requires `dotproduct`; query errors out | Create hybrid indexes with `metric="dotproduct"` |
| 10 | Serverless cold start | First query after idle is slow; looks like a bug | Expected behavior; only worry if *sustained* latency is high |
| 11 | LLM not told to use only the context | Model fills gaps from training data and hallucinates | Explicitly instruct "answer only from the context; say you don't know otherwise" |
| 12 | Blaming the LLM for bad RAG answers | Most failures are bad *retrieval*, not bad generation | Inspect retrieved chunks first when debugging |

---

## Quick Reference Card

**Core mental model**
- Embedding = vector representing meaning. Similar meaning → close vectors.
- Dimensions are fixed by the model and not comparable across models.
- Cosine = angle only (meaning). Dot product = angle + length. Normalize → the two are equal.

**Similarity (normalized vectors)**
```python
similarity = numpy.dot(vec_a, vec_b)   # equals cosine when both are unit length
```

**Embed text (Sentence Transformers)**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")     # 384 dims
vecs = model.encode(texts, normalize_embeddings=True)
```

**Pinecone essentials (current SDK)**
```python
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key="...")

pc.create_index(name="idx", dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index("idx")

index.upsert(vectors=[("id1", vec, {"text": "..."})], namespace="ns")

res = index.query(vector=qvec, top_k=5, include_metadata=True,
                  namespace="ns", filter={"field": {"$eq": "value"}})
```

**Metadata filter operators:** `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`

**Pinecone vocabulary**
- Index → top-level container (fixed dimension + metric)
- Namespace → partition inside an index (free; created on first write)
- Upsert → insert-or-overwrite by ID
- Query → return top_k most similar vectors
- Metadata → key-value extras you can filter on

**Hybrid search**
- Dense vector = meaning; sparse vector = keywords.
- Single index (recommended, simpler) or separate indexes (more flexible).
- **Must use `metric="dotproduct"`** — cosine doesn't support sparse-dense.
- `alpha` weights dense vs. sparse.

**RAG pipeline**
1. Chunk → 2. Embed → 3. Upsert  *(indexing, once)*
4. Embed question → 5. Retrieve top-k → 6. Generate from context  *(per query)*
- Same embedding model for chunks and questions.
- Tell the LLM to answer only from the retrieved context.
- Debug bad answers by checking retrieval first.

**Reference docs:** Sentence Transformers documentation; Cohere embedding guide; Pinecone Quickstart and hybrid-search docs; Pinecone + OpenAI RAG examples.

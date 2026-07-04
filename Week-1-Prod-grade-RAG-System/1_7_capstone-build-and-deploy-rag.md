# Capstone: Build and Deploy a Production RAG System

This is the end-to-end project: take a real document collection and turn it into a deployed question-answering service that retrieves relevant passages, generates answers grounded in them, cites its sources, and runs behind a live API. Each stage builds one piece — corpus, ingestion, retrieval, generation, evaluation, deployment — and by the end you'll have a working system and, just as importantly, a mental model of how the pieces fit.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order — each stage depends on the one before.

> **A note on running the code:** the services and libraries here (Pinecone, Cohere, the Anthropic and Modal SDKs) change their APIs periodically. The code reflects current versions, but if something doesn't match, check each provider's current docs — the *architecture* is what matters and it's stable.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers (a **vector**) representing the *meaning* of text; similar meaning → close vectors. An **embedding model** produces them.
- **Retrieval** = finding the stored passages most similar to a query by comparing vectors. A **vector database** stores vectors and does this fast.
- A **chunk** is a small passage of a document (we split documents because whole documents are too big to embed well or fit into a model's input).
- **RAG** (Retrieval-Augmented Generation): retrieve relevant chunks, feed them to a **large language model** (LLM — an AI that generates text) so its answer is grounded in your documents.
- **Metadata** = structured extras attached to each chunk (title, source, URL, section) used for filtering and citations.
- **Hybrid search** = combining semantic (meaning) search with keyword search. **Reranking** = re-scoring retrieved candidates with a sharper model. **Contextual retrieval** = prepending a short generated context blurb to each chunk before embedding, so chunks carry their document context.

The system you're building has two halves: an **offline ingestion pipeline** (run once: chunk → contextualize → embed → store) and an **online query path** (run per request: retrieve → rerank → generate with citations). The golden rule that ties it together: **the same embedding model must be used everywhere** — a query vector is only comparable to chunk vectors made by the identical model.

---

## Hour 1 — Corpus Setup

### The problem we're solving

A RAG system is only as good as the documents behind it, and toy corpora hide problems. To build something real, you need a **non-trivial corpus** — large and varied enough that retrieval actually matters (you can't just paste everything into the model) and structured enough to test filtering and citations. A good target is on the order of 1,000+ technical documents.

### Choosing and gathering a corpus

Good sources for a substantial, legally usable corpus:
- **arXiv** — abstracts (and full papers) of scientific preprints, with rich metadata (title, authors, date, categories). Great for testing domain vocabulary.
- **Software/product documentation** — a documentation site or repository you can clone. Naturally sectioned, full of exact terms.
- **A Wikipedia subset** — available as bulk datasets; broad and well-structured.

Whatever you pick, **respect licenses and terms of use**, and download a manageable slice first (a few hundred to a couple thousand documents) to control cost — you can scale up once the pipeline works.

### Capture metadata from the start

This is the step beginners skip and regret. For every document, record structured **metadata** you'll need later: a stable **id**, `title`, `source`/`url`, `date`, and (if available) `section` and `author`. You'll use this metadata two ways downstream: to **filter** searches ("only docs after 2023") and to **cite** sources in answers ("according to *Title*, section 3"). Metadata you didn't capture at ingestion is expensive to backfill.

### Clean the text

Retrieval quality starts with clean text. Strip boilerplate (navigation, ads, repeated headers/footers), fix encoding issues, and normalize whitespace. **Garbage in, garbage out** applies literally: noisy source text produces noisy chunks and noisy embeddings.

```python
# A simple document record you'll carry through the whole pipeline
documents = [
    {
        "id": "arxiv-2404.16130",
        "title": "From Local to Global: A Graph RAG Approach",
        "url": "https://arxiv.org/abs/2404.16130",
        "date": "2024-04-24",
        "section": "abstract",
        "text": "The use of retrieval-augmented generation ...",   # cleaned text
    },
    # ... 1,000+ more
]
```

> **Gotcha — capture metadata now, not later.** Citations and metadata filtering both depend on structured fields attached to each chunk. If you ingest raw text without ids, titles, and sources, you can't cite or filter later without re-ingesting. Design the metadata schema before you write the ingestion code.

> **Gotcha — start small to control cost.** Contextual retrieval and embedding both call APIs per chunk. Prove the pipeline on a few hundred documents before running it on tens of thousands, or you may spend real money debugging a typo at scale.

### Hands-on exercise (Hour 1)

1. Choose a corpus source and download a slice of at least a few hundred documents (respecting its license).
2. Define a document record schema with `id`, `title`, `url`/`source`, `date`, `section`, and `text`.
3. Write a loader that reads the raw data into that schema and cleans the text (strip boilerplate, normalize whitespace).
4. Estimate the total size in tokens (a rough rule: ~0.75 words per token) and note it — this drives your later cost and chunking decisions. Confirm every record has complete metadata.

### Interview questions — Hour 1

**1. (Conceptual) What makes a corpus "non-trivial," and why does it matter for building RAG?**
A non-trivial corpus is large and varied enough that retrieval genuinely matters — you can't just put everything in the model's context — and structured enough to exercise filtering and citations. It matters because small toy corpora hide the real problems (retrieval precision, chunking trade-offs, metadata plumbing) that a production system must handle, so building against a realistic corpus surfaces those issues early.

**2. (Practical) Why capture metadata during ingestion rather than adding it later?**
Because downstream features depend on it: metadata filtering restricts searches by fields like date or category, and citations reference a chunk's title, source, and location. If you ingest raw text without these fields attached, you can't filter or cite without re-ingesting the whole corpus — an expensive redo. Designing the metadata schema up front avoids that.

**3. (Conceptual) How does source text quality affect the final system?**
It sets the ceiling on everything: noisy text (boilerplate, broken encoding, junk) produces noisy chunks and noisy embeddings, which degrade retrieval, which degrades answers no matter how good the model is. Cleaning and normalizing text before chunking — "garbage in, garbage out" — is a foundational quality step, not an optional polish.

**4. (Practical) You plan to eventually index 100,000 documents. Why start with a few hundred?**
Because ingestion (contextualization and embedding) calls APIs per chunk and costs real money and time, so debugging the pipeline at full scale is wasteful and expensive. Proving the pipeline on a small slice lets you catch bugs cheaply, then scale up once it works correctly. It's cost control and faster iteration.

**5. (Conceptual) What metadata fields are most valuable to capture, and what are they used for?**
A stable id (to reference and update chunks), title and source/url (for citations and traceability), date (for time-based filtering), and section/author where available (for finer filtering and citation precision). Together they enable metadata-filtered retrieval and source-attributed answers — two features that make a RAG system trustworthy and controllable.

---

## Hour 2 — Ingestion Pipeline

### The problem we're solving

Raw documents can't be searched semantically — you must transform them into stored, embedded chunks. The **ingestion pipeline** does this offline, once (and again when documents change). Our pipeline has four steps: **chunk** each document, **contextualize** each chunk (prepend a generated context blurb), **embed** the chunks, and **upsert** them into the vector database with their metadata.

### Step 1: Chunk

Split each document into passages small enough to embed precisely but large enough to be meaningful. A **recursive** splitter (which tries to break on paragraph, then sentence, then word boundaries) is a solid default.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# For each document, produce chunks while carrying its metadata onto each chunk.
```

### Step 2: Contextualize (Contextual Retrieval)

A lone chunk often loses the context of its document — "the rate rose to 3.2%" doesn't say whose rate or which year. **Contextual retrieval** fixes this: for each chunk, an LLM writes a short blurb situating it within its document, and you **prepend** that blurb before embedding. The chunk's vector then carries document-level context, improving retrieval.

To make this affordable (one LLM call per chunk would be costly), use **prompt caching** — a feature that caches the full document once so every chunk of it reuses the cached copy cheaply.

```python
import anthropic
client = anthropic.Anthropic()

CONTEXT_PROMPT = (
    "Here is a document:\n<document>{doc}</document>\n\n"
    "Here is a chunk from it:\n<chunk>{chunk}</chunk>\n\n"
    "Give a short (1-2 sentence) context situating this chunk within the document, "
    "to improve search retrieval. Respond with only the context."
)

def contextualize(chunk_text, full_doc_text):
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",   # a small, fast, cheap model is ideal here
        max_tokens=150,
        messages=[{"role": "user",
                   "content": CONTEXT_PROMPT.format(doc=full_doc_text, chunk=chunk_text)}],
        # In production, structure the call so the document portion is cached across the
        # document's chunks (prompt caching) to keep per-chunk cost low.
    )
    context = msg.content[0].text
    return f"{context}\n\n{chunk_text}"   # prepend context, THEN embed this combined text
```

### Step 3: Embed

Embed each (contextualized) chunk into a vector with your chosen embedding model. **Use this exact model everywhere**, including at query time.

```python
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer("all-MiniLM-L6-v2")   # 384 dims; pick once, use everywhere
vectors = embed_model.encode(contextualized_chunks, normalize_embeddings=True)
```

### Step 4: Upsert

Store the vectors in the vector database with the chunk text and metadata. **Upsert** means insert-or-overwrite by id, which makes re-runs safe.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
pc.create_index(name="rag", dimension=384, metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"))
index = pc.Index("rag")

index.upsert(
    vectors=[
        (chunk_id, vector.tolist(),
         {"text": chunk_text, "title": doc["title"], "url": doc["url"], "section": doc["section"]})
        for chunk_id, vector, chunk_text in batch
    ],
    namespace="docs",
)   # upsert in batches of ~100; keep chunk text + citation metadata in the metadata
```

> **Gotcha — the ingestion embedding model and the query embedding model must be identical.** They produce vectors in the same space only if they're the same model. Mixing them silently breaks retrieval (no error). Pin the model name and dimension in one config used by both the pipeline and the query path.

> **Gotcha — make ingestion idempotent and resumable.** Ingesting a large corpus can fail partway (network blip, rate limit). Because upsert overwrites by id, using stable chunk ids lets you safely re-run without duplicating data. Track progress so you can resume rather than restart, and batch upserts (~100 vectors) rather than one at a time.

> **Gotcha — store the chunk text (or a reference) in metadata, and mind the size limit.** You need the chunk's text back at query time to feed the model and to cite it, so store it in metadata — but respect the per-vector metadata size limit (keep chunks reasonably sized; for very large text, store a reference id and keep the full text in a separate store).

### Hands-on exercise (Hour 2)

1. Build the pipeline end to end on your corpus from Hour 1: chunk → contextualize → embed → upsert.
2. For a few chunks, print the generated context blurb and confirm it accurately situates the chunk — this verifies contextualization is working.
3. Upsert in batches with stable chunk ids and metadata (text, title, url, section). Confirm the index's vector count matches your chunk count.
4. Re-run the pipeline and confirm the vector count does *not* double (proving idempotent upserts). Write 2–3 sentences on where you'd add resumability for a large corpus.

### Interview questions — Hour 2

**1. (Conceptual) What are the four stages of the ingestion pipeline, and what does each do?**
Chunk (split documents into embeddable passages), contextualize (prepend an LLM-generated blurb situating each chunk in its document, to preserve context), embed (turn each contextualized chunk into a vector with the chosen model), and upsert (store the vectors with chunk text and metadata in the vector database, insert-or-overwrite by id). Together they transform raw documents into a searchable index.

**2. (Conceptual) Why does contextual retrieval prepend context before embedding, and how is it made affordable?**
Because a lone chunk often loses its document's context (ambiguous references, missing subject), which hurts retrieval; prepending a short situating blurb makes the chunk's vector carry that context so it matches relevant queries better. It's made affordable with prompt caching, which caches the full document once so generating context for each of its chunks reuses the cached copy cheaply rather than reprocessing the document every time.

**3. (Gotcha) Why must the ingestion and query embedding models be identical?**
Because vectors are only comparable when produced by the same model — different models map text into different vector spaces. If chunks are embedded with one model and queries with another, similarity scores are meaningless and retrieval returns irrelevant results, with no error raised. Pinning one model (and dimension) in shared config for both paths prevents this silent failure.

**4. (Practical) How do you make a large ingestion job safe to re-run after a failure?**
Use stable chunk ids so that upserts (insert-or-overwrite by id) don't create duplicates on re-run, batch the upserts, and track which chunks have been processed so you can resume rather than restart. This idempotence and resumability let a partially failed large ingestion continue cheaply instead of redoing everything.

**5. (Conceptual) Why store the chunk text in the vector's metadata, and what's the constraint?**
Because at query time you need the actual text back — to feed retrieved chunks to the LLM and to cite them — and the vector alone doesn't contain readable text. The constraint is the per-vector metadata size limit, so chunks should be kept reasonably sized; for very large text, store a reference id in metadata and keep the full text in a separate store to look up after retrieval.

---

## Hour 3 — Retrieval Layer

### The problem we're solving

With chunks indexed, you need the **query path's** first job: given a user question, return the most relevant chunks. A single semantic search is a decent start, but production retrieval does better in two ways: **hybrid search** (combine semantic search with keyword search, so exact terms like codes and names aren't missed) and **reranking** (re-score the top candidates with a sharper model). The pattern is **two-stage retrieval**: cast a wide net cheaply, then sharpen.

### Stage 1: Hybrid retrieval (wide net)

Embed the query with the same model used at ingestion, and query the vector database for a generous number of candidates (say, the top 50). For true hybrid search on Pinecone, you'd use an index with the `dotproduct` metric and supply both a dense vector and a **sparse vector** (a keyword-signal vector) for each query and chunk; the query then blends meaning and exact terms. (A purely semantic first stage also works as a simpler starting point.)

### Stage 2: Rerank (sharpen)

A **reranker** re-scores each candidate against the query by examining the pair together, far more accurately than the first-stage retrieval. Run it on the ~50 candidates to keep the best ~8. **Cohere Rerank** is a hosted reranker; you send the query and candidate texts and get them back reordered by relevance.

```python
import cohere
co = cohere.ClientV2()   # reads COHERE_API_KEY

def retrieve(query, top_n=50, top_k=8):
    # Stage 1: embed the query with the SAME model as ingestion, get wide candidate set
    qvec = embed_model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=qvec, top_k=top_n, include_metadata=True, namespace="docs")
    matches = res["matches"]
    candidate_texts = [m["metadata"]["text"] for m in matches]

    # Stage 2: rerank the candidates down to the best top_k
    reranked = co.rerank(model="rerank-v3.5", query=query,
                         documents=candidate_texts, top_n=top_k)

    # Return the best chunks WITH their metadata (needed for generation + citations)
    return [matches[r.index]["metadata"] for r in reranked.results]
```

Notice the retrieval function returns each chunk's **metadata** (text, title, url, section) — not just the text — because the next stage needs the sources to cite.

> **Gotcha — retrieve wide, rerank narrow.** The first stage should return more candidates than you ultimately use (e.g. 50 → 8), because the reranker can only reorder what it's given. Retrieve too few and the right chunk may never reach the reranker. But don't rerank thousands — reranking cost and latency grow with the candidate count.

> **Gotcha — reranking can't fix missing-from-retrieval chunks.** If the correct chunk isn't in the first-stage candidates, no reranking recovers it. When answers are wrong, check first-stage recall before blaming the reranker (or the model). Most quality problems trace back to what retrieval did or didn't surface.

> **Gotcha — carry metadata through, keyed by a stable id.** The reranker returns results by their index in the list you sent; map that index back to the original match and its metadata carefully, or you'll cite the wrong source. Keep chunk ids consistent end to end.

### Hands-on exercise (Hour 3)

1. Implement the `retrieve(query)` function: embed the query (same model as ingestion), get ~50 candidates from your index, rerank to ~8 with Cohere.
2. Run it on 5 sample questions and print the top chunks with their titles/sources. Confirm the reranked order looks more relevant than the raw vector order.
3. Compare retrieval *with* and *without* the reranking stage on a question where the best answer wasn't the top vector hit. Note whether reranking promoted it.
4. Write 2–3 sentences on your chosen `top_n` and `top_k` and the latency/quality trade-off you'd tune.

### Interview questions — Hour 3

**1. (Conceptual) What is two-stage retrieval and why use it?**
It's retrieving a wide set of candidates cheaply in a first stage (semantic or hybrid search), then reordering them with a sharper, slower model (a reranker) in a second stage to select the best few. It's used because first-stage retrieval scales to the whole corpus but is coarse, while rerankers are accurate but too slow for the full corpus — combining them gives both scale and precision.

**2. (Conceptual) Why add hybrid search rather than relying on semantic search alone?**
Because semantic search can miss exact terms — product codes, error codes, names, rare identifiers — that carry little semantic signal, while keyword search matches them precisely. Hybrid search combines both, so the system handles paraphrased/meaning-based queries and exact-term queries well. Pure semantic search alone has a blind spot for literal strings.

**3. (Practical) How do you choose the candidate count for stage one versus the final count?**
Retrieve wider than you'll use (e.g. 50 candidates to ultimately keep 8), because the reranker can only reorder what it receives — too few candidates and the right chunk never reaches it. But don't retrieve/rerank excessively, since reranking cost and latency scale with candidate count. Tune the numbers against a recall target and your latency budget.

**4. (Gotcha) A user gets a wrong answer. Why check first-stage recall before blaming the model?**
Because reranking and generation can only work with the chunks retrieval surfaced; if the correct chunk wasn't in the first-stage candidates, no reranker or model can produce a correct, grounded answer. Most RAG failures are retrieval failures, so verifying whether the right chunk was retrieved is the fastest way to locate the real cause.

**5. (Gotcha) Why must the retrieval function return chunk metadata, not just text?**
Because the generation stage needs the source information — title, url, section — to cite where each claim came from, and metadata filtering and traceability depend on it too. Returning only text discards the ability to attribute answers to sources, which is essential for a trustworthy RAG system. Metadata must be carried through, keyed by a stable id so sources map correctly.

---

## Hour 4 — Generation with Citations

### The problem we're solving

Now the query path's final job: take the retrieved chunks and the question, and have the LLM produce an answer **grounded in those chunks and citing its sources**. Citations matter because they make answers **verifiable** — the user can check that a claim really came from a source — and they reduce hallucination. There are two ways to do this.

### Approach 1: Prompt-based citations (portable)

Put the retrieved chunks into the prompt, numbered and labeled with their sources, and instruct the model to answer using *only* those chunks and to cite them by number. This works with any LLM (Claude, GPT, others), but it's somewhat brittle: the model can miscite or drift from the format, and if you ask it to quote sources, those quotes count as output tokens.

```python
def build_prompt(question, chunks):
    context = "\n\n".join(
        f"[{i+1}] (Source: {c['title']}) {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        "Answer the question using ONLY the sources below. "
        "Cite sources inline by their number, like [1] or [2]. "
        "If the sources don't contain the answer, say you don't know.\n\n"
        f"Sources:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
# Send build_prompt(...) to any chat model; then parse the [n] markers back to sources.
```

### Approach 2: Native citations (more reliable, Claude)

Anthropic's API has a built-in **citations** feature that removes the brittleness. You pass each retrieved chunk as a *document* with citations enabled; Claude returns its answer as a series of text blocks, and blocks that make a factual claim carry a structured `citations` array pointing to the exact source text (`cited_text`), which document it came from, and the character range. The advantages: citations are **guaranteed to point at real provided text** (the model can't fabricate a citation), the cited text **doesn't count toward output tokens**, and citation quality is higher than prompt-based approaches.

```python
import anthropic
client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY

def generate(question, chunks):
    # Each retrieved chunk becomes a citable document
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c["text"]},
            "title": c.get("title", "source"),
            "citations": {"enabled": True},          # enable on all documents (all-or-none)
        }
        for c in chunks
    ]
    resp = client.messages.create(
        model="claude-opus-4-8",   # or claude-sonnet-4-6 / claude-haiku-4-5-20251001
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": documents + [
                {"type": "text",
                 "text": f"Answer using only the documents. If they don't answer it, say so.\n\nQuestion: {question}"}
            ],
        }],
    )
    # resp.content is a list of blocks; blocks with claims carry a `citations` array
    # (each with cited_text, document_index, document_title, and char offsets).
    return resp.content
```

Either way, the crucial instruction is the same: **answer only from the provided context, and admit when the answer isn't there.** Without it, the model may fall back on its training data and reintroduce the hallucinations RAG exists to prevent.

> **Gotcha — always instruct "use only the context; say you don't know otherwise."** If you don't constrain the model to the retrieved chunks, it will happily answer from memory, defeating the grounding that citations provide. This one instruction is the difference between a grounded answer and a confident guess.

> **Gotcha — prompt-based citations are brittle; validate them.** With the prompt approach, the model can invent citation numbers or break format, so parse and verify that each cited number maps to a real source. The native citations feature avoids this by guaranteeing citations point to provided documents — prefer it when using Claude.

> **Gotcha — map citations back to your original sources for the user.** A citation references a *document* by index (its position in the list you sent). Translate that back to the chunk's real metadata (title, url, section) so the user sees a meaningful, clickable source, not an internal index.

### Hands-on exercise (Hour 4)

1. Implement `generate(question, chunks)` using the native citations feature (if using Claude) and/or the prompt-based template (for any model).
2. Ask a question your corpus answers, and print the answer along with each citation's source text and title. Confirm the citations point at the chunks that actually support the claims.
3. Ask a question your corpus does *not* cover and confirm the model says it doesn't know (proving your "use only the context" instruction works).
4. Map at least one returned citation back to its original document's url/section. Write 2–3 sentences comparing the prompt-based and native approaches on reliability.

### Interview questions — Hour 4

**1. (Conceptual) Why are citations important in a RAG system?**
They make answers verifiable — a user can confirm a claim actually came from a cited source — and they reduce hallucination by tying the answer to real retrieved text. This builds trust and auditability, which are essential in domains like legal, medical, and finance where an unsupported or fabricated claim is a serious liability.

**2. (Conceptual) Contrast prompt-based citations with a native citations feature.**
Prompt-based citations instruct the model to cite sources by number in its text; they're portable across any LLM but brittle — the model can miscite, break format, or inflate output tokens by quoting. A native citations feature (like Anthropic's) has the API return structured citations guaranteed to point at provided source text, with the cited text not counting toward output tokens and higher citation quality — more reliable but provider-specific.

**3. (Practical) What single instruction most protects answer grounding, and why?**
"Answer using only the provided context, and say you don't know if it isn't there." Without it, the model fills gaps from its training data, reintroducing hallucination and undermining the whole point of retrieval and citations. Constraining the model to the retrieved chunks is what keeps answers grounded and citations meaningful.

**4. (Gotcha) With prompt-based citations, why must you validate the citations the model produces?**
Because the model can fabricate citation numbers or break the expected format, producing references that don't map to real sources — which misleads users and defeats verifiability. Parsing and checking that each cited number corresponds to an actual provided source (or using a native feature that guarantees valid citations) keeps the attribution trustworthy.

**5. (Practical) A citation comes back referencing "document index 2." How do you make it useful to the end user?**
Map that index back to the original chunk's metadata — its title, url, and section — that you sent as document 2, and present that human-readable, ideally clickable source instead of the internal index. The user needs to see and verify the real source, so carrying and translating metadata through the pipeline is what turns an index into a usable citation.

---

## Hour 5 — Quick Evaluation

### The problem we're solving

You now have a working pipeline — but is it any good? Without measurement, "improving" it is guesswork. This hour builds a lightweight **evaluation** so you can see how the system performs and, crucially, *which stage* is the weak link. Even 10–20 carefully chosen test cases beat opinions.

### Build a small gold set

A **gold set** (evaluation set) is a collection of test questions paired with the known-correct answer or the chunk/document that should be retrieved. Hand-write **10–20** realistic questions about your corpus, and for each, record which chunk (by id) or document actually answers it — and optionally a reference answer. This is manual work, but it's the foundation of all measurement.

### Measure retrieval first

Because retrieval caps everything, measure it first. The key metric is **recall@k**: of the questions whose correct chunk you know, what fraction have that chunk appear in the top *k* retrieved results. (If each question has one right chunk, recall@5 is simply "how often the right chunk was in the top 5.") Related metrics: **MRR** (Mean Reciprocal Rank — rewards ranking the first correct result higher) and **NDCG** (rewards putting relevant results near the top).

```python
def recall_at_k(gold_set, k=5):
    """gold_set: list of (question, correct_chunk_id)."""
    hits = 0
    for question, correct_id in gold_set:
        retrieved = retrieve(question, top_k=k)            # your Hour 3 function
        retrieved_ids = [c["id"] for c in retrieved]       # requires ids in metadata
        if correct_id in retrieved_ids:
            hits += 1
    return hits / len(gold_set)

print("recall@5:", round(recall_at_k(gold_set, k=5), 3))
```

### Then check answer quality

Separately, judge the *generated answers* on your gold questions: are they **faithful** (grounded in the retrieved context, with valid citations) and **relevant** (do they actually answer the question)? You can judge manually for 10–20 cases, or use an LLM to score them. Keep the two measurements distinct: retrieval quality and answer quality are different failure modes.

### Diagnose the weak link

The point of measuring both is to know *where* to fix things:
- **Low recall@k** → the problem is retrieval: revisit chunking, hybrid weighting, candidate count, or the embedding model.
- **Good recall but bad answers** → the problem is generation: revisit the prompt, the "use only context" instruction, or the model.

Fix the diagnosed stage, then re-measure. And change **one thing at a time**, or you won't know what helped.

> **Gotcha — a 10–20 question gold set is noisy but essential.** It won't give statistically tight numbers, and a couple of lucky or unlucky cases can swing the score. But it's vastly better than no measurement, and it catches gross problems. Grow it over time; treat early numbers as directional, not precise.

> **Gotcha — measure before you optimize.** It's tempting to add reranking, contextual retrieval, and query rewriting all at once. Without a baseline measurement you can't tell which change helped or hurt. Establish the baseline first, then add one improvement at a time and re-measure.

> **Gotcha — separate retrieval quality from answer quality.** A good answer requires both good retrieval *and* good generation, but they fail independently. Measuring them together hides which one is broken. Recall@k isolates retrieval; faithfulness/relevance isolates generation.

### Hands-on exercise (Hour 5)

1. Write a gold set of 10–20 realistic questions about your corpus, each labeled with the correct chunk id (and a reference answer if you can).
2. Compute **recall@5** for your retrieval function using the harness above. Record the number as your baseline.
3. Run the full pipeline on the same questions and judge (manually or with an LLM) whether each answer is faithful and relevant.
4. Diagnose the weak link (retrieval vs. generation), make *one* targeted change, and re-measure. Write 3–4 sentences on what you changed, the before/after numbers, and your conclusion.

### Interview questions — Hour 5

**1. (Conceptual) What is a gold set and why is it the foundation of RAG evaluation?**
A gold set is a collection of test questions paired with their known-correct answers or the chunks that should be retrieved. It's foundational because it turns "is this good?" from an opinion into a measurement: you can compute recall@k on retrieval and judge answer quality objectively, enabling before/after comparisons when you change the system. Without it, improvement is guesswork.

**2. (Conceptual) Define recall@k and explain why retrieval is measured first.**
Recall@k is the fraction of questions whose correct chunk appears in the top *k* retrieved results. Retrieval is measured first because it caps the whole system: if the right chunk isn't retrieved, the model can't produce a correct grounded answer regardless of how well it writes. Measuring recall isolates whether the necessary context is reaching the generation stage.

**3. (Practical) Your recall@5 is high but answers are still poor. Where's the problem and what do you do?**
The problem is in generation, not retrieval — the right chunks are being retrieved, but the model isn't using them well. I'd revisit the prompt (especially the "answer only from the context" instruction), check citation handling, and consider the model choice, then re-measure answer faithfulness and relevance on the gold set. High recall rules out retrieval as the cause.

**4. (Gotcha) Why change only one thing at a time when improving the pipeline?**
Because if you change several things at once (add reranking, tweak chunking, swap the model) and the score moves, you can't tell which change caused it — or whether one helped while another hurt. Changing one variable at a time and re-measuring against the baseline attributes each effect correctly, which is the only reliable way to improve the system.

**5. (Gotcha) What are the limits of a 10–20 question gold set, and is it still worth building?**
Its limits are statistical noise: with so few cases, a couple of lucky or unlucky results can swing the numbers, so they're directional rather than precise. But it's absolutely worth building — it catches gross problems, establishes a baseline for comparison, and is vastly better than no measurement. You grow it over time as the system matures.

---

## Hour 6 — Deploy and Document

### The problem we're solving

A pipeline in a notebook helps no one else. The final step is to **deploy** it as a live service others can call, and to **document** it so it can be run, understood, and maintained. We'll expose the query path as a web **API** (an endpoint that accepts a question over HTTP and returns an answer with citations), deploy it **serverlessly** (no servers to manage; it scales automatically and you pay for use), and write a README.

### Deploy the backend on Modal

**Modal** turns Python functions into serverless web services. You define an app, an image with your dependencies, your secrets (API keys), and wrap your query function as a web endpoint. A key detail: loading the embedding model on every request would be slow, so you load it **once per container** using a class with an entry hook.

```python
import modal

app = modal.App("rag-service")

# Container image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "pinecone", "cohere", "anthropic", "sentence-transformers"
)

@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("rag-secrets")],   # PINECONE/COHERE/ANTHROPIC keys
)
class RAGService:
    @modal.enter()                 # runs ONCE when the container starts (not per request)
    def load(self):
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone
        import cohere, anthropic
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")   # same model as ingestion
        self.index = Pinecone().Index("rag")
        self.co = cohere.ClientV2()
        self.client = anthropic.Anthropic()

    @modal.fastapi_endpoint(method="POST")     # exposes an HTTP POST endpoint
    def query(self, item: dict):
        question = item["question"]
        chunks = self.retrieve(question)       # Hour 3 logic (uses self.* clients)
        answer_blocks = self.generate(question, chunks)   # Hour 4 logic
        return {"answer_blocks": answer_blocks}
```

You develop with a temporary, live-reloading URL (`modal serve app.py`) and ship a permanent one with `modal deploy app.py`. Store API keys as Modal **secrets** (never hard-code them). For a user interface, you can deploy a separate frontend (for example on **Vercel**, which specializes in hosting web frontends) that calls this Modal endpoint over HTTP.

```bash
modal serve app.py     # temporary URL, live-reloads on save — for development
modal deploy app.py    # permanent URL — for production
```

### Write the README

Documentation is part of the deliverable. A good README covers: what the system does and its architecture (ingestion pipeline + query path); setup (required API keys/environment variables, how to run ingestion); how to query the deployed endpoint (with an example request); your evaluation results (recall@k and notes); and known limitations. Someone should be able to clone, configure, ingest, and query from the README alone.

> **Gotcha — never hard-code API keys; use the platform's secrets.** Embedding keys in code leaks them (especially if you push to a public repo) and makes rotation painful. Use Modal secrets (or your platform's secret manager) and read keys from the environment. This is a security requirement, not a nicety.

> **Gotcha — load models once, not per request (cold starts).** Serverless containers spin up on demand; loading a model inside the request handler adds that cost to every call. Load it once on container startup (the entry hook) so warm containers serve fast. Be aware the *first* request to a cold container is slower — that's expected serverless behavior.

> **Gotcha — the deployed system needs the same config as ingestion.** The live service must use the identical embedding model, index name, and namespace as your ingestion pipeline, or retrieval breaks. Keep these in shared configuration so the deployed query path and the offline pipeline can never drift apart.

### Hands-on exercise (Hour 6 — capstone completion)

1. Wrap your retrieval + generation logic into a Modal class with a `@modal.enter()` loader (models/clients loaded once) and a `@modal.fastapi_endpoint` query method. Store your API keys as a Modal secret.
2. Run `modal serve` and hit the temporary URL with a test question (via `curl` or the auto-generated docs). Confirm you get back an answer with citations.
3. Deploy it permanently with `modal deploy` and test the live URL.
4. Write a README covering purpose, architecture, setup, how to query (with an example request/response), your recall@k results, and limitations. Optionally, sketch or build a simple frontend that calls the endpoint.

### Interview questions — Hour 6

**1. (Conceptual) What does "serverless" deployment give you for a RAG service?**
It lets you run your code as a web service without provisioning or managing servers: the platform handles infrastructure, scales the service up and down with traffic automatically, and charges for actual usage. For a RAG service that may have bursty or low traffic, this means no idle server costs and no ops burden, while still exposing a live, scalable endpoint.

**2. (Practical) Why load the embedding model in an entry hook rather than inside the request handler?**
Because loading a model is expensive, and doing it inside the request handler repeats that cost on every call, making each request slow. Loading it once when the container starts (the entry hook) means warm containers already have the model ready and serve requests fast. Only the first request to a freshly started (cold) container pays the load cost.

**3. (Conceptual) Why must the deployed query service share configuration with the ingestion pipeline?**
Because retrieval only works if the query path uses the identical embedding model, index, and namespace that ingestion used — vectors must be comparable and point at the same data. If the deployed service drifts (different model or index), retrieval silently breaks. Keeping these values in shared configuration prevents the two halves of the system from diverging.

**4. (Gotcha) What's the risk of hard-coding API keys, and what's the correct approach?**
Hard-coded keys can leak — especially if the code is pushed to a shared or public repository — exposing paid accounts and sensitive access, and they're painful to rotate. The correct approach is to store keys in the platform's secret manager (e.g. Modal secrets) and read them from the environment at runtime, keeping them out of source code entirely.

**5. (Practical) What should a README for this system contain so someone else can run it?**
Purpose and architecture (the ingestion pipeline and the query path), setup instructions (required API keys/environment variables and how to run ingestion), how to query the deployed endpoint with an example request and response, the evaluation results (recall@k and any notes), and known limitations. The goal is that a new person can clone, configure, ingest, and query using only the README.

---

## Gotchas Summary Table

| # | Gotcha | Stage | Why it bites | How to avoid it |
|---|--------|-------|--------------|-----------------|
| 1 | Not capturing metadata at ingestion | Corpus | Can't cite or filter later without re-ingesting | Design the metadata schema up front (id, title, source, date, section) |
| 2 | Debugging at full scale | Corpus | API costs per chunk make big runs expensive | Prove the pipeline on a few hundred docs first |
| 3 | Dirty source text | Corpus | Garbage in → garbage chunks → garbage retrieval | Clean/normalize text before chunking |
| 4 | Different embedding model for query vs. ingestion | Ingestion/Retrieval | Incompatible vector spaces → silent broken retrieval | Pin one model + dimension in shared config |
| 5 | Non-idempotent ingestion | Ingestion | Re-runs duplicate data; failures force restart | Use stable ids (upsert overwrites); track progress; batch |
| 6 | Oversized/absent chunk text in metadata | Ingestion | Hits metadata size limit or can't cite | Keep chunks sized; store text or a reference id |
| 7 | Retrieving too few candidates | Retrieval | Reranker can't recover missing chunks | Retrieve wide (e.g. 50), rerank narrow (e.g. 8) |
| 8 | Blaming the model for bad answers | Retrieval | Most failures are missing-from-retrieval | Check first-stage recall first |
| 9 | Mis-mapping rerank results | Retrieval | Results are by input index → wrong source cited | Map index back to the match; keep ids consistent |
| 10 | No "use only the context" instruction | Generation | Model answers from training data; hallucinates | Always instruct grounding + "say you don't know" |
| 11 | Trusting unvalidated prompt citations | Generation | Model can fabricate/miscite | Validate citation numbers, or use native citations |
| 12 | Showing internal document index to users | Generation | Meaningless reference | Map citation index back to real source metadata |
| 13 | 10–20 question gold set treated as precise | Evaluation | Small sets are noisy | Treat as directional; grow over time |
| 14 | Optimizing before measuring | Evaluation | Can't tell what helped | Baseline first; change one thing at a time |
| 15 | Hard-coded API keys | Deploy | Leak risk; painful rotation | Use platform secrets; read from environment |
| 16 | Loading models per request | Deploy | Slow every call (cold-start cost) | Load once on container start (entry hook) |
| 17 | Deployed config drifts from ingestion | Deploy | Model/index mismatch breaks retrieval | Share model/index/namespace config across both |

---

## Quick Reference Card

**System shape:** offline **ingestion** (chunk → contextualize → embed → upsert) + online **query path** (retrieve → rerank → generate w/ citations) → deployed API.
**Golden rule:** the *same embedding model* everywhere; carry *metadata* end to end.

**1. Corpus** — 1,000+ docs; capture metadata (id, title, url, date, section) up front; clean text; start small.

**2. Ingestion**
```python
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)   # chunk
# contextualize: prepend an LLM-generated context blurb before embedding (use prompt caching)
embed_model.encode(contextualized_chunks, normalize_embeddings=True)   # embed (pin this model)
index.upsert(vectors=[(id, vec, {"text":..., "title":..., "url":...})], namespace="docs")  # upsert (idempotent, batched)
```

**3. Retrieval** — two-stage: wide then sharp
```python
qvec = embed_model.encode(query, normalize_embeddings=True).tolist()      # SAME model
matches = index.query(vector=qvec, top_k=50, include_metadata=True, namespace="docs")["matches"]
reranked = co.rerank(model="rerank-v3.5", query=query,
                     documents=[m["metadata"]["text"] for m in matches], top_n=8)
return [matches[r.index]["metadata"] for r in reranked.results]           # keep metadata!
```

**4. Generation with citations**
```python
# Native (Claude): each chunk = a document with citations enabled
docs = [{"type":"document","source":{"type":"text","media_type":"text/plain","data":c["text"]},
         "title":c["title"],"citations":{"enabled":True}} for c in chunks]
client.messages.create(model="claude-opus-4-8", max_tokens=1024,
    messages=[{"role":"user","content": docs + [{"type":"text","text":"Answer only from the documents...\nQuestion: "+q}]}])
```
- Always instruct: "use only the context; say you don't know." Map citation indices → real sources.
- Prompt-based (any LLM): numbered sources in the prompt, cite by [n], then validate.

**5. Evaluation** — build a 10–20 question gold set (question → correct chunk id)
```python
# recall@k: fraction of questions whose correct chunk is in the top k
hits / len(gold_set)
```
- Measure retrieval (recall@k) *and* answer quality (faithful + relevant) separately.
- Low recall → fix retrieval; good recall + bad answers → fix generation. One change at a time.

**6. Deploy (Modal) + document**
```python
@app.cls(image=image, secrets=[modal.Secret.from_name("rag-secrets")])
class RAGService:
    @modal.enter()
    def load(self): ...                        # load model + clients ONCE per container
    @modal.fastapi_endpoint(method="POST")
    def query(self, item): ...                 # retrieve → generate → return
```
```bash
modal serve app.py     # dev (temporary URL, live reload)
modal deploy app.py    # prod (permanent URL)
```
- Secrets from the platform (never hard-code). README: purpose, architecture, setup, example request, eval results, limitations. Frontend (e.g. Vercel) optional.

**Reference docs:** Pinecone docs (indexing, hybrid); Cohere Rerank docs; Anthropic Messages API and the Citations feature; Modal docs (web endpoints, secrets, deployment); Vercel docs (frontend hosting). Validate every stage on your own corpus.

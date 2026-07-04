# Hybrid Search and Reranking: BM25, Fusion, and Cross-Encoders

Pure meaning-based search has a blind spot: exact terms. Pure keyword search has the opposite blind spot: meaning. This guide shows how to get the best of both — combining keyword and semantic search, then sharpening the result with a model that re-scores the top candidates. By the end you'll understand the classic keyword-ranking method that still beats embeddings on exact matches, how to merge two ranked lists sensibly, and how a reranker squeezes out the final gains.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order.

> **A note on running the code:** library APIs and hosted-model names change. The code reflects current versions, but if something doesn't match, check the relevant library or provider documentation — the underlying ideas (keyword ranking, rank fusion, reranking) are stable.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers (a **vector**) representing the *meaning* of text. Comparing vectors (usually by **cosine similarity**, the angle between them) finds text with *similar meaning*. This is **semantic search** (also called **dense** or **vector** search).
- **Keyword search** (also called **lexical** or **sparse** search) matches the literal *words* in the query against the words in documents, regardless of meaning.
- **Retrieval** means finding, for a given query, the most relevant documents. A **ranked list** is the results ordered best-to-worst.
- **RAG** (Retrieval-Augmented Generation) is the pattern of retrieving relevant text and feeding it to a **large language model** (LLM — an AI that generates text) so its answers are grounded in your documents.
- **Top-k** means "the top k results" (e.g. top 5).

The big idea of this guide: semantic search understands *meaning* but can miss an exact term the user clearly typed; keyword search nails *exact terms* but misses paraphrases. **Hybrid search** runs both and merges them. **Reranking** then re-scores the merged top candidates with a slower but sharper model.

---

## Hour 1 — BM25 Fundamentals

### The problem we're solving

Suppose a user searches for the error code `TS-999` or the product `SKU-4471`. A semantic embedding model, which is built to understand *meaning*, has little meaning to grab onto in a bare alphanumeric code — so it may rank the exact, correct document poorly. Yet a human knows the user wants that *literal string*. Matching literal words is exactly what classic keyword search does well, and the dominant keyword-ranking method is **BM25**. Understanding it (and the **TF-IDF** idea underneath) explains *why* keyword search still earns its place alongside fancy embeddings.

### TF-IDF: the foundation

**TF-IDF** stands for **Term Frequency–Inverse Document Frequency**. It scores how relevant a document is to a query word by combining two intuitions:

- **Term Frequency (TF)** — how often the query word appears in the document. More mentions suggests more relevance. ("Refund" appearing five times in a document hints it's about refunds.)
- **Inverse Document Frequency (IDF)** — how *rare* the word is across the whole collection. A word that appears in almost every document (like "the" or "is") carries little signal, so it gets a low weight; a rare, distinctive word (like "refund") gets a high weight. IDF is computed from `log(N / df)`, where `N` is the total number of documents and `df` is how many of them contain the word — so rarer words (small `df`) score higher.

The TF-IDF relevance of a word to a document is roughly `TF × IDF`: frequent *and* distinctive words drive the score.

TF-IDF has two weaknesses BM25 was designed to fix: raw term frequency grows without limit (a word appearing 100 times isn't 100× more relevant), and it ignores document length (a very long document can rack up matches just by being long).

### BM25: TF-IDF, refined

**BM25** (the name means "Best Matching 25," from a series of experiments) keeps the TF and IDF intuitions but adds two refinements:

1. **Term-frequency saturation.** Instead of letting repeated terms count linearly forever, BM25 applies diminishing returns: the 2nd mention of a word matters a lot more than the 20th. A parameter called **k1** (commonly around 1.2) controls how quickly the benefit of repetition flattens out.
2. **Document-length normalization.** BM25 compares each document's length to the *average* document length and discounts long documents so they don't win just for being long. A parameter called **b** (commonly 0.75) controls how strongly length is penalized (b=0 means no length penalty; b=1 means full).

Put together, a document's BM25 score for a query is the sum, over each query term, of that term's IDF multiplied by its saturated, length-normalized frequency:

```
score(Document, Query) = Σ  IDF(term) · [ f · (k1 + 1) ] / [ f + k1 · (1 − b + b · |D|/avgdl) ]
                    term in Query

  f      = how often the term appears in the document
  |D|    = length of the document
  avgdl  = average document length across the collection
  k1, b  = tuning parameters (often ~1.2 and 0.75)
```

You don't need to memorize this — the takeaway is: **BM25 rewards rare query words that appear (but with diminishing returns), while correcting for document length.** It's the default ranking function in search engines like **Elasticsearch** (a widely used search system), needs no training and no GPU, runs fast, and is fully interpretable (you can see exactly which words drove a match).

### Why lexical search still matters

In an era of powerful embeddings, why keep keyword search? Because semantic search has real blind spots that BM25 covers:
- **Exact identifiers:** product codes, error codes, SKUs, version numbers, function names — literal strings with little semantic content. BM25 matches them precisely; embeddings often miss.
- **Rare words, acronyms, and proper nouns:** distinctive tokens a general embedding model may not represent well.
- **Out-of-domain queries:** embeddings trained on general text can underperform on specialized vocabulary, where exact word matching is more reliable.
- **No training, low cost, interpretable:** BM25 works out of the box and explains itself.

This is precisely why hybrid search (Hour 2) exists: keep BM25 for exact-term strengths, add semantic search for meaning.

```python
# A simple local BM25 with the rank-bm25 library
# pip install rank-bm25
from rank_bm25 import BM25Okapi

corpus = [
    "Refunds are issued within 5 business days.",
    "Our stores open at 9 am every day.",
    "Activate plan ZX-9920 from the billing page.",
]
tokenized_corpus = [doc.lower().split() for doc in corpus]   # BM25 works on tokens (words)
bm25 = BM25Okapi(tokenized_corpus)

query = "ZX-9920".lower().split()
scores = bm25.get_scores(query)                  # a BM25 score per document
print(bm25.get_top_n(query, corpus, n=2))        # the exact-code doc ranks first
```

> **Gotcha — BM25 matches words, not meaning.** It will *not* connect "refund" to "money back" or "automobile" to "car" — different words are simply different to BM25. That's its weakness and exactly why you pair it with semantic search. Don't expect BM25 alone to handle paraphrases or synonyms.

> **Gotcha — tokenization and preprocessing matter.** BM25 operates on tokens, so how you split text, whether you lowercase, and whether you strip punctuation all affect matches. Inconsistent preprocessing between indexing and querying (e.g. lowercasing one but not the other) quietly degrades results. Keep it consistent.

### Hands-on exercise (Hour 1)

1. `pip install rank-bm25`. Build a small corpus of 8–10 short documents, making sure a few contain distinctive exact tokens (a fake SKU, an error code, an acronym).
2. Query BM25 with one of those exact tokens and confirm the right document ranks first.
3. Now query with a *paraphrase* of a document's meaning (different words, same idea) and observe BM25 struggle — making its blind spot concrete.
4. Write 3–4 sentences listing query types where you'd trust BM25 over a semantic model, and vice versa.

### Interview questions — Hour 1

**1. (Conceptual) What do TF and IDF each capture in TF-IDF scoring?**
TF (term frequency) captures how often a query word appears in a document — more occurrences suggest more relevance. IDF (inverse document frequency) captures how rare the word is across the whole collection — common words like "the" get low weight while distinctive, rare words get high weight. Together they favor words that are both frequent in a document and rare overall.

**2. (Conceptual) What two problems with TF-IDF does BM25 fix, and how?**
BM25 fixes unbounded term frequency by applying saturation (diminishing returns on repeated terms, controlled by k1), so the 20th mention adds far less than the 2nd. And it fixes the lack of length handling by normalizing for document length against the collection average (controlled by b), so long documents don't score highly just for being long.

**3. (Practical) Give examples of queries where BM25 outperforms semantic search.**
Queries centered on exact identifiers — product codes like SKU-4471, error codes like TS-999, version numbers, function names — and on rare acronyms or proper nouns. These are literal strings with little semantic content that embeddings often rank poorly, whereas BM25 matches them precisely. Out-of-domain specialized vocabulary is another case.

**4. (Conceptual) Why does keyword search remain relevant despite strong embedding models?**
Because semantic search has blind spots BM25 covers: exact identifiers and rare tokens, out-of-domain vocabulary, and cases needing literal matching. BM25 also needs no training or GPU, is cheap and fast, and is interpretable. Rather than replacing embeddings, it complements them — which is the basis for hybrid search.

**5. (Gotcha) A colleague's BM25 search misses obvious paraphrase matches. Is BM25 broken?**
No — that's expected behavior. BM25 matches literal words, not meaning, so it won't connect synonyms or paraphrases ("money back" vs "refund"). The fix isn't to debug BM25 but to add semantic (vector) search alongside it in a hybrid setup, letting embeddings handle meaning while BM25 handles exact terms.

---

## Hour 2 — RRF and Hybrid Scoring

### The problem we're solving

Hybrid search runs two retrievers — semantic (vector) and keyword (BM25) — and each returns its own ranked list. But you can't just add their scores together: a cosine similarity might range from 0 to 1, while a BM25 score is unbounded and could be 2 or 25. Adding `0.8 + 17.3` is meaningless. You need a principled way to **fuse** two ranked lists into one. There are two main approaches.

### Approach 1: score-based fusion (weighted combination)

Bring both lists' scores onto the same scale, then blend them. The usual scaling is **normalization** — for example, **min-max normalization**, which rescales each list's scores to the range 0–1 (the highest becomes 1, the lowest 0). Then combine with a weight:

```
final_score = alpha · normalized_semantic_score + (1 − alpha) · normalized_keyword_score
```

The weight **alpha** controls the mix: `alpha = 1` is pure semantic, `alpha = 0` is pure keyword, and values between blend them. (You met this same `alpha` idea if you've used a vector database's built-in hybrid search.)

This approach uses the actual score magnitudes, which can be informative — but it's sensitive to how scores are distributed. A single outlier score can distort the normalization, and different score distributions make the right `alpha` hard to pin down.

### Approach 2: rank-based fusion (Reciprocal Rank Fusion)

**Reciprocal Rank Fusion (RRF)** sidesteps the scaling problem entirely by *ignoring the raw scores* and using only each document's **rank position** (1st, 2nd, 3rd…) in each list. The insight: a document that ranks near the top of *multiple* lists is probably genuinely relevant, regardless of the exact scores.

For each document, RRF sums a small contribution from each list it appears in, where the contribution shrinks as the rank gets worse:

```
RRF_score(d) = Σ   1 / (k + rank_i(d))
             list i

  rank_i(d) = the position (1-based) of document d in list i
  k         = a constant that dampens the weight of top ranks (commonly 60)
```

A document ranked 1st in a list contributes `1/(60+1)`; ranked 2nd, `1/(60+2)`; and so on. Sum these across all lists, sort by the total, and you have your fused ranking. Documents appearing high in *both* the semantic and keyword lists rise to the top.

The constant **k** (default **60**, from the original RRF research) controls how much the very top ranks dominate: a larger `k` flattens the differences between rank positions. Some implementations also add per-list **weights** to favor one retriever over another.

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """ranked_lists: a list of lists of document ids, each ordered best -> worst."""
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):   # rank is 1-based
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Return doc ids sorted by fused score, best first
    return sorted(scores, key=scores.get, reverse=True)

semantic_results = ["docA", "docB", "docC"]   # from vector search, best-first
keyword_results  = ["docC", "docD", "docA"]   # from BM25, best-first
print(reciprocal_rank_fusion([semantic_results, keyword_results]))
```

RRF is popular because it's **simple, robust, and needs no score normalization** — it just works across retrievers with wildly different score scales. It's the default fusion method in many systems (including Elasticsearch's hybrid search). The trade-off is that it *discards score magnitude*: a document that barely edged into 1st place is treated identically to one that dominated 1st place.

> **Gotcha — don't add raw scores from different retrievers.** Cosine and BM25 scores live on different scales; summing them lets whichever scale is larger dominate arbitrarily. Either normalize first (score-based fusion) or use ranks (RRF). Raw addition is a classic hybrid-search bug.

> **Gotcha — RRF's `k` and any per-list weights are tunable, not sacred.** The default `k=60` is a reasonable starting point, not a law. If keyword matches matter more for your data, weight that list higher; if you want top ranks to dominate more, lower `k`. Tune on your own evaluation set.

> **Gotcha — deduplicate before and after fusion.** The same document can appear in both lists. Fusion handles this naturally (it sums contributions per document id), but only if you key by a *stable, identical document id* across both retrievers. Mismatched ids make the same document look like two, splitting its credit.

### Hands-on exercise (Hour 2)

1. From a small corpus, get a semantic ranked list (vector search) and a keyword ranked list (BM25) for the same query. Print both — note how they differ.
2. Implement the `reciprocal_rank_fusion` function above and fuse the two lists. Compare the fused top-3 to each individual top-3.
3. Try `k=10`, `k=60`, and `k=200` and observe how the fused ordering shifts (smaller `k` lets top ranks dominate more).
4. Construct a query where the right document ranks 1st in *one* list but poorly in the other, and confirm fusion still surfaces it. Write 2–3 sentences on why fusion beat either retriever alone here.

### Interview questions — Hour 2

**1. (Conceptual) Why can't you simply add a semantic score and a BM25 score to combine two retrievers?**
Because they're on different, incompatible scales — cosine similarity might be 0–1 while BM25 is unbounded (e.g. up to 20+). Adding them lets the larger-scaled score dominate arbitrarily rather than reflecting true relevance. You must either normalize both onto a common scale first or use a rank-based method like RRF that ignores raw magnitudes.

**2. (Conceptual) Explain how Reciprocal Rank Fusion works and what it deliberately ignores.**
RRF combines ranked lists using only each document's rank position, not its score. For each document it sums `1/(k + rank)` over every list it appears in, then sorts by that total, so documents ranked highly across multiple lists rise to the top. It deliberately ignores raw score magnitude, which makes it robust to different score scales but blind to how strongly a document won a position.

**3. (Conceptual) What does the `k` constant do in RRF?**
It dampens how much the very top ranks dominate. Because contributions are `1/(k + rank)`, a larger `k` shrinks the gap between, say, rank 1 and rank 5, flattening the influence of top positions; a smaller `k` makes top ranks count for proportionally more. The common default is 60, but it's tunable.

**4. (Practical) When might you prefer score-based (weighted) fusion over RRF, or vice versa?**
Prefer RRF when you want simplicity and robustness across retrievers with very different score scales, or when score magnitudes are unreliable — it needs no normalization. Prefer score-based weighted fusion when the actual score magnitudes carry useful information you want to exploit and you're willing to handle normalization and outlier sensitivity, and when you want fine control via an alpha weight.

**5. (Gotcha) After implementing hybrid search, the same document seems to appear twice in results. What likely went wrong?**
The two retrievers probably used different identifiers for the same document, so fusion treated one document as two and split its credit instead of summing it. The fix is to key both retrievers' results by a single stable, identical document id so fusion (and deduplication) correctly recognizes the same document across both lists.

---

## Hour 3 — Cohere Rerank

### The problem we're solving

Hybrid search gives you a good merged top-N (say, the top 50 or 100). But the *ordering* within that top-N still comes from retrieval methods optimized for speed over a huge corpus, not for pinpoint relevance judgment. A **reranker** is a model that takes the query and each candidate document *together* and scores how relevant the document is to that specific query — far more accurately than the original retrieval, because it examines the pair directly. You run it on the top-N candidates to produce a sharper top-K. **Cohere Rerank** is a popular hosted reranker you call through an API.

This is the standard **two-stage retrieval** pattern: a fast **first stage** (hybrid search) casts a wide net to get a candidate set; a slower, sharper **second stage** (reranking) reorders that set and keeps only the best. Feeding the LLM fewer, more relevant documents also cuts cost and latency downstream.

### How rerankers work

A reranker scores relevance by considering the query and a document *jointly* — reading them together rather than comparing two separately-made vectors. (The technical name for this architecture, a **cross-encoder**, is the focus of Hour 4.) Because it sees both texts at once, it can judge relevance with much more nuance than first-stage retrieval, including for queries that have multiple constraints or require reasoning.

### Using the Cohere Rerank API

You send a query, a list of candidate documents, and how many top results you want (`top_n`); you get back the documents reordered by relevance, each with a relevance score.

```python
import cohere
co = cohere.ClientV2()   # reads the COHERE_API_KEY environment variable

query = "how long until I get my refund?"
candidates = [
    "Our stores open at 9 am every day.",
    "Refunds are issued within 5 business days of approval.",
    "Premium members get free expedited shipping.",
    "To request a refund, visit the orders page.",
]

resp = co.rerank(
    model="rerank-v3.5",       # current general rerank model (verify the latest name)
    query=query,
    documents=candidates,
    top_n=3,                    # return the 3 most relevant
)

for r in resp.results:
    print(r.index, round(r.relevance_score, 3))   # index points back into `candidates`
```

A few practical details:
- **Current models** include `rerank-v3.5` (multilingual, strong on semi-structured data like tables, code, and JSON), with a newer "Rerank 4" generation (Pro and Fast variants) also available. Always check the docs for the latest model name.
- **Pricing is per "search."** One search is a single query against up to ~100 documents; longer documents get chunked, and each chunk counts. (As a rough figure, on the order of ~$0.001–$0.0025 per search depending on the model — verify current pricing.) This is often cheaper than re-embedding or asking a generative model to judge relevance.
- **Multilingual and privately deployable** (it can run in your own cloud for data-sensitive cases).

> **Gotcha — reranking adds latency, so rerank a *small* candidate set.** The reranker scores every (query, document) pair, so cost and latency grow with the number of candidates. There's a direct trade-off: rerank more candidates for slightly better recall, or fewer for lower latency and cost. A common pattern is retrieve ~50–100, rerank to ~5–20. Don't rerank thousands.

> **Gotcha — reranking can't fix bad retrieval.** A reranker only reorders the candidates it's *given*. If the right document never made it into the first-stage top-N, no reranking can surface it. Measure first-stage recall; if it's missing relevant documents, fix retrieval (better chunking, hybrid search) before blaming the reranker.

> **Gotcha — the `index` in results points back to your input list.** Cohere returns results by their position (`index`) in the documents you sent, ordered by relevance — not your original documents reordered in place. Map each result's `index` back to your candidate list (and its metadata) carefully.

### Hands-on exercise (Hour 3)

1. Get a Cohere API key (free trial tier works). Take a query and ~10 candidate documents where the best answer is *not* the one a naive method ranks first.
2. Call `co.rerank` with `top_n=3` and print each result's `index` and `relevance_score`. Confirm the most relevant document rises to the top.
3. Compare the reranked order to the order before reranking. Note which documents moved and by how much.
4. Time a rerank call over 10 candidates vs ~100 candidates and write 2–3 sentences on the latency/quality trade-off you observed.

### Interview questions — Hour 3

**1. (Conceptual) What is a reranker, and where does it sit in a retrieval pipeline?**
A reranker is a model that scores how relevant each candidate document is to a specific query by examining the query and document together, producing a sharper relevance ordering than first-stage retrieval. It sits as the second stage of a two-stage pipeline: a fast first stage (e.g. hybrid search) returns a candidate set, and the reranker reorders that set and keeps the most relevant top-K.

**2. (Conceptual) Why can a reranker judge relevance more accurately than vector search?**
Because it considers the query and document jointly — reading both texts together — rather than comparing two vectors that were created separately without awareness of each other. Seeing the pair directly lets it capture nuanced relevance, including multi-constraint or reasoning-heavy queries, at the cost of being slower than vector comparison.

**3. (Practical) How is Cohere Rerank priced, and what's the practical implication?**
It's priced per "search," where one search is a query against up to roughly 100 documents (with long documents chunked and each chunk counting). The implication is to rerank a modest candidate set rather than huge numbers of documents, and that reranking is often more cost-effective than re-embedding or using a generative model to score relevance.

**4. (Gotcha) Reranking didn't improve your results at all. What should you check first?**
Whether the relevant document is even in the first-stage candidate set. A reranker only reorders what it's given, so if retrieval never surfaced the right document into the top-N, reranking can't help. Check first-stage recall; if it's missing relevant documents, improve retrieval (chunking, hybrid search) before expecting the reranker to add value.

**5. (Practical) You need lower latency in a reranked pipeline. What's the main lever?**
Reduce the number of candidates sent to the reranker, since its cost and latency scale with the number of (query, document) pairs scored. Retrieve and rerank a smaller top-N (trading a little recall for speed), rather than reranking a very large candidate set. Tuning N up or down directly trades quality against latency and cost.

---

## Hour 4 — Cross-Encoders and Local Rerankers

### The problem we're solving

Cohere Rerank is convenient, but it's a paid API and sends your data to a third party. You can run a reranker *yourself* with an open-source model. To do that well, you need to understand the architecture behind every reranker — the **cross-encoder** — and how it differs from the embedding models you use for first-stage retrieval. This distinction is one of the most important concepts in retrieval.

### Bi-encoders vs. cross-encoders

There are two ways a model can compare a query and a document:

- **Bi-encoder** — this is what an *embedding model* is. It encodes the query into a vector and each document into a vector *separately*, then compares the vectors (e.g. cosine similarity). Because documents are encoded independently and ahead of time, you can **precompute** all document vectors once and search millions of them fast. The cost: the query and document never "see" each other during encoding, so the comparison is coarser.
- **Cross-encoder** — this is what a *reranker* is. It takes the query and one document *together* as a single joint input, lets the model attend across both texts at once, and outputs a single relevance score. Because it reads the pair jointly, it judges relevance much more accurately. The cost: you **can't precompute** anything — the model must run once per (query, document) pair at query time, which is far too slow to run over a whole corpus.

This explains the whole two-stage design: use the fast **bi-encoder** to retrieve a candidate set from the full corpus, then use the accurate **cross-encoder** to rerank just those candidates. (Cohere Rerank from Hour 3 is, under the hood, a hosted cross-encoder — running a local cross-encoder is the do-it-yourself version of the same thing.)

### Running a local cross-encoder

The `sentence-transformers` library provides a `CrossEncoder` class and access to many open reranker models.

```python
from sentence_transformers import CrossEncoder

# A small, fast, classic reranker trained on a passage-ranking dataset
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

query = "how long until I get my refund?"
candidates = [
    "Our stores open at 9 am every day.",
    "Refunds are issued within 5 business days of approval.",
    "To request a refund, visit the orders page.",
]

# Option A: score each (query, document) pair directly
scores = reranker.predict([(query, doc) for doc in candidates])
# scores is one relevance number per candidate; higher = more relevant

# Option B: let the library rank for you and return the top results
ranked = reranker.rank(query, candidates, top_k=3)
for item in ranked:
    print(item["corpus_id"], round(item["score"], 3))   # corpus_id indexes `candidates`
```

### Choosing a local reranker model

Common open cross-encoder rerankers:
- **`cross-encoder/ms-marco-MiniLM-L6-v2`** (and the L-12 variant) — small, fast, English, trained on a standard passage-ranking dataset; a great, lightweight default. (For these models, applying a sigmoid activation maps scores to a 0–1 range.)
- **`BAAI/bge-reranker-v2-m3`** — a popular multilingual reranker (and smaller `bge-reranker-base`/`large` variants), stronger than the MiniLM models at a larger size.
- Newer families (e.g. ModernBERT-based rerankers and reranker variants of recent model families) push quality further at various sizes — the landscape moves quickly, so benchmark a couple on your data.

For serving rerankers in production, dedicated inference servers (such as Hugging Face's Text Embeddings Inference) give better batching and throughput than calling the model directly.

### Local cross-encoder vs. hosted API: when to use which

- **Local cross-encoder** — free to run (no per-search fee), keeps data on your own infrastructure (privacy), and works offline. The cost: you need a **GPU** for good speed and you must operate and scale it yourself.
- **Hosted reranker (e.g. Cohere)** — no infrastructure to manage, often strong multilingual quality, and trivial to integrate. The cost: per-search fees and sending data to a third party.

Choose local when cost-at-scale, privacy, or offline operation dominate and you have the hardware and skills; choose a hosted API when you want minimal operational burden and are comfortable with usage-based cost.

> **Gotcha — never use a cross-encoder for first-stage retrieval over the whole corpus.** Because it can't precompute and must score every pair at query time, running it over thousands or millions of documents per query is impossibly slow. Cross-encoders are *only* for reranking a small candidate set produced by a fast first stage. Confusing the two roles is the central mistake here.

> **Gotcha — different rerankers output scores on different scales.** Some output unbounded logits, some 0–1 (with a sigmoid), some negative-to-positive. The *ranking* (sort order) is what matters for reranking; don't compare raw scores across different models or assume a fixed threshold transfers. If you need a probability-like score, apply the model's documented activation.

> **Gotcha — match the model to your language and domain.** An English-only reranker (like the MiniLM models) will underperform on other languages; use a multilingual reranker (like `bge-reranker-v2-m3`) for multilingual data. As always, evaluate candidates on your own queries rather than trusting a generic ranking.

### Hands-on exercise (Hour 4)

1. `pip install sentence-transformers`. Load `cross-encoder/ms-marco-MiniLM-L6-v2`.
2. Take the same query and ~10 candidates from the Hour 3 exercise. Score them with `reranker.predict` (or `reranker.rank`) and compare the top-3 to what Cohere Rerank produced — note where the local and hosted rerankers agree and differ.
3. Swap in `BAAI/bge-reranker-v2-m3` and compare its ranking and its speed to the MiniLM model.
4. Build a tiny two-stage pipeline: retrieve top-20 with BM25 (from Hour 1) or vector search, then rerank to top-5 with a cross-encoder. Write 3–4 sentences on whether reranking improved the final top-5 and what it cost in latency.

### Interview questions — Hour 4

**1. (Conceptual) Explain the difference between a bi-encoder and a cross-encoder.**
A bi-encoder (an embedding model) encodes the query and each document into vectors *separately* and compares the vectors, so document vectors can be precomputed and searched fast over a huge corpus, at the cost of coarser comparison. A cross-encoder (a reranker) takes the query and a document *together* as one input and outputs a relevance score, judging relevance more accurately but requiring a model run per pair at query time, which is too slow for whole-corpus search.

**2. (Conceptual) Why is a cross-encoder used for reranking rather than first-stage retrieval?**
Because it can't precompute document representations — it must process each (query, document) pair at query time — so running it over an entire corpus per query is prohibitively slow. It's reserved for reranking a small candidate set that a fast bi-encoder first stage has already narrowed down, where its superior accuracy pays off on a manageable number of pairs.

**3. (Practical) When would you run a local cross-encoder instead of using Cohere Rerank, and vice versa?**
Run a local cross-encoder when avoiding per-search costs at scale, keeping data on your own infrastructure for privacy, or operating offline matters, and you have GPU hardware and the skills to run it. Use a hosted reranker like Cohere when you want minimal operational burden, strong out-of-the-box (often multilingual) quality, and easy integration, and you're comfortable with usage-based cost and third-party data handling.

**4. (Gotcha) A teammate tries to use a cross-encoder as the only retrieval step over a million-document corpus and it's unusably slow. What's the conceptual error?**
They're using a cross-encoder for first-stage retrieval, but cross-encoders can't precompute and must score every query-document pair at query time, which doesn't scale to a whole corpus. The fix is the two-stage design: retrieve a candidate set with a fast bi-encoder (or BM25), then apply the cross-encoder only to rerank those candidates.

**5. (Gotcha) You compare raw relevance scores from two different rerankers and they look wildly different. Is one broken?**
Not necessarily — different rerankers output scores on different scales (unbounded logits, 0–1 with a sigmoid, negative-to-positive, etc.). What matters for reranking is the resulting sort order, not the absolute numbers, so you shouldn't compare raw scores across models or assume a threshold transfers. If you need comparable, probability-like scores, apply each model's documented activation.

---

## Gotchas Summary Table

| # | Gotcha | Where | Why it bites | How to avoid it |
|---|--------|-------|--------------|-----------------|
| 1 | Expecting BM25 to match meaning | BM25 | It matches literal words, not synonyms/paraphrases | Pair BM25 with semantic search (hybrid) |
| 2 | Inconsistent tokenization/preprocessing | BM25 | Mismatched indexing vs. query handling drops matches | Keep tokenization and casing identical both sides |
| 3 | Adding raw scores from different retrievers | Fusion | Different scales (cosine vs BM25) → one dominates | Normalize first, or use rank-based RRF |
| 4 | Treating RRF's `k`/weights as fixed | Fusion | Defaults aren't optimal for every dataset | Tune `k` and per-list weights on your eval set |
| 5 | Splitting a document's credit in fusion | Fusion | Different ids for the same doc across retrievers | Key both retrievers by one stable document id |
| 6 | Reranking a huge candidate set | Cohere Rerank | Latency/cost scale with number of candidates | Retrieve ~50–100, rerank to ~5–20 |
| 7 | Expecting rerank to fix bad retrieval | Cohere Rerank | Reranker only reorders what it's given | Measure first-stage recall; fix retrieval first |
| 8 | Misusing the result `index` | Cohere Rerank | Results point back into the input list by index | Map each result's `index` to your candidates/metadata |
| 9 | Cross-encoder for first-stage retrieval | Cross-encoders | Can't precompute; far too slow over a corpus | Use it only to rerank a small candidate set |
| 10 | Comparing raw scores across rerankers | Cross-encoders | Different models use different score scales | Use sort order; apply documented activation if needed |
| 11 | Wrong-language reranker | Cross-encoders | English-only models underperform on other languages | Use a multilingual reranker; evaluate on your data |

---

## Quick Reference Card

**The shape of a strong retrieval pipeline:** `BM25 + dense (vector) → fuse (RRF) → rerank (cross-encoder) → top-K to the LLM`.

**BM25 (keyword/lexical ranking)**
- TF-IDF idea: reward query words that are frequent in a doc (TF) *and* rare in the collection (IDF).
- BM25 adds: term-frequency **saturation** (param `k1` ≈ 1.2) and **document-length normalization** (param `b` ≈ 0.75).
- Strengths: exact codes/IDs, rare terms, out-of-domain; no training, fast, interpretable. Blind spot: synonyms/paraphrases.
```python
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
bm25.get_top_n("ZX-9920".lower().split(), corpus, n=3)
```

**Hybrid fusion (combine two ranked lists)**
- Don't add raw scores (different scales). Two options:
  - **Score-based:** normalize (e.g. min-max), then `alpha·dense + (1−alpha)·keyword` (alpha=1 semantic, 0 keyword).
  - **RRF (rank-based):** `score(d) = Σ 1/(k + rank_i(d))`, default `k=60`. No normalization; robust; ignores score magnitude.
```python
def rrf(lists, k=60):
    s = {}
    for L in lists:
        for rank, d in enumerate(L, 1):
            s[d] = s.get(d, 0) + 1/(k + rank)
    return sorted(s, key=s.get, reverse=True)
```

**Reranking (sharpen the top-N)** — two-stage retrieval
- **Hosted (Cohere Rerank):** `co.rerank(model="rerank-v3.5", query=..., documents=[...], top_n=K)`; priced per search (query + up to ~100 docs); results carry `index` + `relevance_score`.
- **Local (cross-encoder):**
```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")   # or BAAI/bge-reranker-v2-m3
ce.rank(query, candidates, top_k=5)                         # or ce.predict([(query, d), ...])
```

**Bi-encoder vs cross-encoder (the key distinction)**
- **Bi-encoder** = embedding model: encodes query & docs *separately* → precompute, fast, whole-corpus search (first stage).
- **Cross-encoder** = reranker: encodes query+doc *together* → accurate, slow, no precompute (rerank a small candidate set only).
- Cohere Rerank is a hosted cross-encoder. Never use a cross-encoder for first-stage retrieval.

**Rules of thumb**
- Retrieve wide (~50–100), rerank to narrow (~5–20). Reranking can't fix missing-from-retrieval documents.
- Rerank quality/score scales vary by model — trust the sort order, evaluate on your own data, match the language.

**Reference docs:** Elasticsearch BM25 documentation; the Reciprocal Rank Fusion paper and Elastic's hybrid-search write-ups; Cohere Rerank documentation; the Sentence Transformers cross-encoder documentation and model cards.

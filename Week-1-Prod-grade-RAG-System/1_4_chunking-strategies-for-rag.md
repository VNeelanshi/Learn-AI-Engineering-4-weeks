# Chunking Strategies for RAG: From Fixed-Size to Contextual Retrieval

How you cut documents into pieces is one of the highest-leverage decisions in any retrieval system — and one beginners almost always get wrong by accepting a default. This guide builds from the simplest splitting method up through the techniques that fix its weaknesses, including a recent approach that prepends generated context to every chunk. By the end you'll understand the trade-offs, know several advanced strategies, and be able to measure which one actually works on your data.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order.

> **A note on running the code:** the libraries here (LangChain, LlamaIndex) reorganize their import paths fairly often. The code reflects current packages, but if an import fails, check the library's current documentation — the *concepts* are stable even when import paths move.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers (a **vector**) representing the *meaning* of a piece of text. Similar meaning → vectors that sit close together. An **embedding model** produces these vectors.
- **Retrieval** means finding the stored texts most similar to a query, by comparing vectors. A **vector database** stores vectors and does this search quickly.
- **RAG** (Retrieval-Augmented Generation) is the pattern of retrieving relevant text and feeding it to a **large language model** (LLM — an AI that generates text) so its answer is grounded in your documents rather than its memory.
- A **chunk** is a small passage of a document. We split documents into chunks because whole documents are usually too big to embed well or to fit into an LLM's input, and because retrieving a focused passage beats retrieving an entire document.
- A **token** is roughly a word-piece; chunk sizes are measured in characters or tokens.

The core tension this whole guide circles: **a chunk small enough to retrieve precisely is often too small to make sense on its own.** Every strategy here is a different way to manage that tension.

---

## Hour 1 — Chunking Overview: Fixed-Size, Recursive, and Semantic

### The problem we're solving

You have a document. You need pieces. The naive instinct — "just cut every N characters" — works, but it cuts blindly, often mid-sentence or mid-word, destroying the meaning you're trying to preserve. The three strategies below represent increasing sophistication in *where* you choose to cut.

### Fixed-size chunking

**Fixed-size chunking** splits text into pieces of a set length (e.g. every 500 characters or 256 tokens), regardless of sentence or word boundaries. Two parameters define it: **chunk size** (how big each piece is) and **chunk overlap** (how many characters/tokens to repeat from the end of one chunk at the start of the next).

Why overlap? Because a hard cut can split a key sentence across two chunks, leaving neither chunk complete. **Overlap** softens this: by repeating a bit of text at the boundary, an idea that straddles the cut survives intact in at least one chunk.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",          # split purely by length, ignoring structure
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(document_text)
```

Fixed-size is fast, predictable, and simple. Its weakness is exactly its blindness: it has no respect for the document's structure, so it routinely severs sentences and ideas.

> **Gotcha — overlap costs storage and can duplicate results.** More overlap means more total chunks (each carries repeated text), which increases embedding and storage costs, and overlapping chunks can both match the same query and crowd out other results. A common starting point is overlap around 10–20% of chunk size; tune it, don't max it.

### Recursive chunking

**Recursive chunking** is the smarter default. Instead of cutting blindly, it tries to split on a *hierarchy* of natural boundaries — first paragraphs, then sentences, then words, then characters — only descending to a finer boundary when a piece is still too big. The result keeps semantically related text together far better than fixed-size.

The standard tool, LangChain's **RecursiveCharacterTextSplitter**, does this with a default list of separators (`["\n\n", "\n", " ", ""]`, i.e. paragraph → line → word → character). It tries the first separator; if a resulting chunk still exceeds the size limit, it re-splits *that* chunk on the next separator, and so on.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_text(document_text)          # -> list of strings
# or splitter.create_documents([document_text])      # -> list of Document objects
```

It also adapts to structured content. For code, `RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, ...)` uses language-aware separators (splitting on class and function boundaries) so you don't cut through the middle of a function.

> **Gotcha — recursive splitting respects boundaries but still enforces a size cap.** If a single paragraph is larger than `chunk_size`, it *will* be split — just at the next-best boundary (a sentence, then a word). Recursive chunking minimizes damage; it doesn't eliminate cuts. Set `chunk_size` large enough for your typical semantic unit.

### Semantic chunking

**Semantic chunking** goes further: instead of using punctuation or length, it uses *meaning* to decide where to cut. It embeds sentences and looks for points where the topic shifts — a large jump in the embeddings between consecutive sentences signals a natural boundary — and splits there. The idea is that each chunk should be about one coherent thing.

LangChain's **SemanticChunker** (in the experimental package) implements this:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings   # or any embeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",   # how aggressive the topic-shift cutoff is
)
chunks = splitter.create_documents([document_text])
```

Because it groups by topic rather than count, semantic chunking tends to produce *fewer, larger, variable-sized* chunks. It can shine on documents that wander across many topics — but it costs extra compute (you must embed while chunking), and it is **not always worth it**: on well-structured documents, recursive chunking often matches it at a fraction of the cost.

> **Gotcha — semantic chunking is not a free win.** It adds embedding cost at ingestion time and produces unpredictable chunk sizes (some may be too large for your LLM's input). Treat it as a candidate to *evaluate against* recursive chunking on your data, not an automatic upgrade.

### The size trade-off (the heart of the hour)

Whatever the method, **chunk size is a genuine trade-off**:
- **Too large:** the chunk covers several ideas, so its single embedding is a blurry average — it matches queries weakly, and you may overflow the LLM's input with mostly-irrelevant text.
- **Too small:** the chunk is precise but loses the surrounding context needed to interpret it (a sentence referring to "this approach" is useless without knowing which approach).

There's no universal best size — it depends on your documents and queries. Treat it as something to *measure* (Hour 4), not guess.

### Hands-on exercise (Hour 1)

1. `pip install langchain-text-splitters`. Take one real document (an article, a manual page, a few pages of a PDF as text).
2. Chunk it three ways: `CharacterTextSplitter` (fixed-size), `RecursiveCharacterTextSplitter`, and — if you have an embedding model — `SemanticChunker`. Use the same `chunk_size` where applicable.
3. Print the first few chunks from each. Note concretely where fixed-size cuts mid-sentence and how recursive avoids it.
4. Count how many chunks each method produced and eyeball their size variation. Write 2–3 sentences on which method best preserved complete ideas for *your* document, and why.

### Interview questions — Hour 1

**1. (Conceptual) Why do we chunk documents for RAG instead of embedding whole documents?**
Whole documents are usually too large to embed effectively (their single vector becomes a blurry average of many topics) and too large to fit into an LLM's input. Chunking produces focused embeddings and lets retrieval return the specific passage that answers a query, improving both retrieval precision and the quality of context handed to the model.

**2. (Conceptual) What is chunk overlap and why is it useful?**
Overlap repeats a portion of text from the end of one chunk at the start of the next. It's useful because a hard cut can split a key idea or sentence across two chunks, leaving neither complete; overlap ensures the straddling idea survives intact in at least one chunk, reducing boundary information loss.

**3. (Conceptual) How does recursive chunking differ from fixed-size chunking?**
Fixed-size cuts at a set length regardless of structure, often mid-sentence. Recursive chunking tries a hierarchy of natural boundaries — paragraphs, then sentences, then words, then characters — descending only when a piece still exceeds the size limit, so it keeps semantically related text together far better while still respecting a size cap.

**4. (Practical) When might semantic chunking be worth its extra cost, and when not?**
It's worth considering for documents that cover many distinct topics, where cutting at true topic boundaries meaningfully improves chunk coherence. It's often not worth it for well-structured documents, where recursive chunking achieves similar quality without the added embedding cost and where semantic chunking's unpredictable chunk sizes can complicate things. The deciding factor is evaluation on your own data.

**5. (Gotcha) A teammate sets chunk size very small to "make retrieval more precise" and quality drops. What likely happened?**
Chunks that are too small lose the surrounding context needed to interpret them — a fragment referring to "this method" or "the above" becomes meaningless alone — so retrieval may match keywords but the retrieved text can't actually support a good answer. The fix is to balance size against context, or to decouple the retrieval unit from the context unit using an advanced strategy (Hour 2).

---

## Hour 2 — Advanced Chunking: Parent-Document, Small-to-Big, and Sentence-Window

### The problem we're solving

Hour 1 ended on a dilemma: small chunks retrieve precisely but lack context; large chunks have context but retrieve imprecisely. The advanced strategies in this hour resolve it with one shared insight: **the unit you search and the unit you read don't have to be the same.** You can *retrieve* on small, precise chunks but *feed the LLM* a larger, context-rich passage.

We'll use **LlamaIndex** here (a popular RAG library; its splitter components are called **node parsers**, where a **node** is its term for a chunk). A **post-processor** in LlamaIndex is a step that modifies retrieved nodes before they reach the LLM.

### Parent-document / small-to-big

The **parent-document** (or **small-to-big**) strategy keeps two linked versions of your content: small **child** chunks and the larger **parent** chunk each child came from. You embed and search the *small* children (precise matching), but when a child is retrieved you hand the LLM its *parent* (full context). Retrieval precision and reading context, both satisfied.

### Auto-merging (a hierarchical generalization)

LlamaIndex generalizes this with a **HierarchicalNodeParser** that builds *multiple* levels of chunks — large parents, medium children, small grandchildren — each linked to its parent. Paired with an **AutoMergingRetriever**, the system retrieves on the smallest nodes, but if *enough* small children of the same parent get retrieved, it automatically "merges" them — replacing them with the single parent node so the LLM sees one coherent larger passage instead of scattered fragments.

```python
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

# Build a hierarchy: 2048-token parents -> 512 -> 128-token leaves
node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
nodes = node_parser.get_nodes_from_documents(documents)

leaf_nodes = get_leaf_nodes(nodes)   # the smallest chunks — these get embedded & searched
# Index leaf_nodes in a vector store; keep all nodes in a docstore so parents can be fetched.
# An AutoMergingRetriever then promotes children to their parent when enough are retrieved.
```

### Sentence-window

The **sentence-window** strategy takes precision to the extreme: it splits documents into *individual sentences* and embeds each one alone (maximally precise matching). But it stores, in each sentence's metadata, a **window** of the surrounding sentences — and at query time, *after* retrieving a matching sentence, it swaps in that surrounding window before sending text to the LLM. So you match on one sentence but read it in context.

This uses LlamaIndex's **SentenceWindowNodeParser** plus a **MetadataReplacementNodePostProcessor** (which performs the swap from single sentence to window). The window text is kept in metadata, invisible to the embedding model — so the embedding stays tightly scoped to the one sentence.

```python
from llama_index.core.node_parser import SentenceWindowNodeParser

node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,                              # sentences captured on each side
    window_metadata_key="window",               # metadata holding the surrounding window
    original_text_metadata_key="original_sentence",
)
nodes = node_parser.get_nodes_from_documents(documents)
# At retrieval time, a MetadataReplacementNodePostProcessor replaces the matched
# single sentence with its stored window before the text reaches the LLM.
```

(LangChain offers an equivalent to parent-document via its `ParentDocumentRetriever`; the pattern is the same regardless of library.)

> **Gotcha — these strategies need a place to store the "big" version.** Because the retrieved unit (small chunk/sentence) differs from the fed unit (parent/window), the system must keep both and maintain the link. In practice that means a document store alongside the vector index. Skipping this — or losing the parent↔child link — breaks the whole approach.

> **Gotcha — auto-merging behavior depends on a threshold.** Whether children get merged into a parent depends on how many of that parent's children were retrieved and on the threshold setting. Set it poorly and you either rarely merge (back to fragmented context) or merge too eagerly (back to bloated, imprecise context). Tune it on your data.

> **Gotcha — what gets embedded vs. what the LLM sees can silently diverge.** In sentence-window, the embedding is of a single sentence while the LLM sees a window. That's the point — but it means a chunk can be retrieved for a sentence whose surrounding window turns out unhelpful. Inspect both the retrieved sentence and its window when debugging.

### Hands-on exercise (Hour 2)

1. `pip install llama-index`. Load one longer document.
2. Parse it two ways: with `SentenceWindowNodeParser` (window_size 2–3) and with `HierarchicalNodeParser` (`chunk_sizes=[1024, 256, 64]` or similar).
3. For sentence-window, retrieve for a question and print both the single matched sentence *and* its surrounding window — see how the window adds the context the lone sentence lacked.
4. For the hierarchical parse, print a leaf node and then its parent node, and describe in 2–3 sentences when feeding the parent would help the LLM answer versus when the leaf alone would suffice.

### Interview questions — Hour 2

**1. (Conceptual) What single insight underlies parent-document, small-to-big, and sentence-window strategies?**
That the unit you search and the unit you read don't have to be the same. You can embed and retrieve small, precise chunks for accurate matching, then feed the LLM a larger, context-rich passage (the parent chunk or surrounding window) so it has enough context to answer well — getting precision and context simultaneously.

**2. (Conceptual) How does the sentence-window strategy work?**
It splits documents into individual sentences and embeds each sentence alone for very precise matching, while storing the surrounding sentences as a "window" in each sentence's metadata (invisible to the embedding model). At query time, after a sentence is retrieved, a post-processing step replaces it with its stored window before the text reaches the LLM, so matching is precise but the model reads the sentence in context.

**3. (Practical) When would you choose auto-merging (hierarchical) over a simple parent-document setup?**
When documents have meaningful structure at multiple scales and queries may need varying amounts of context, a multi-level hierarchy lets the system promote scattered small matches up to the right-sized parent automatically, rather than always returning a single fixed parent size. Auto-merging consolidates several related small hits into one coherent larger passage when enough of them are retrieved.

**4. (Gotcha) What infrastructure does a small-to-big strategy require that plain chunking doesn't?**
A place to store the larger "parent" units and a maintained link between each small retrieved chunk and its parent — typically a document store alongside the vector index. Because the retrieved unit and the fed unit differ, losing that storage or that parent↔child link breaks the strategy.

**5. (Practical) In sentence-window, why is the surrounding window kept out of the embedding?**
So the embedding stays tightly scoped to the single sentence, giving maximally precise matching against queries. If the window were embedded, the vector would blur across multiple sentences and lose precision. The window is only added back after retrieval, when the goal shifts from precise matching to giving the LLM enough context.

---

## Hour 3 — Contextual Retrieval (Anthropic's Technique)

### The problem we're solving

Every strategy so far fights the same enemy: **chunks lose the context of the document they came from.** A chunk reading "The rate increased to 3.2% in the third quarter" doesn't say *whose* rate, *which* year, or *what* document — so it may not match a query like "what was Acme's Q3 2023 interest rate," and even if retrieved, it's ambiguous. The advanced strategies in Hour 2 fix this by *retrieving differently*. **Contextual Retrieval**, a technique published by Anthropic, fixes it by *changing what you embed*: it prepends a short, chunk-specific description of context to each chunk before embedding it.

### How it works

For each chunk, you ask an LLM to write a brief (typically 50–100 token) explanation situating that chunk within the whole document — for example, "This chunk is from Acme Corp's Q3 2023 financial report; it discusses the company's interest rate, which rose to 3.2%." You **prepend** that context to the chunk, and *then* embed it. Now the chunk's vector carries the document-level context it was missing, so it matches relevant queries far better.

Anthropic's full technique has two parts that both use this contextualization:
- **Contextual Embeddings** — prepend the generated context before creating the vector embedding (as above).
- **Contextual BM25** — prepend the same context before building a **BM25 index**. *BM25* is a classic keyword-matching ranking method (an improved form of term-frequency scoring) that excels at exact matches like names, codes, and technical terms. Adding context to the BM25 side helps it match exact terms that only appear in the document's broader context.

These two are combined with **rank fusion** (merging and de-duplicating the two ranked result lists into one), and optionally a **reranking** step — a model that re-scores the merged candidates and keeps only the most relevant — is added on top.

### Why it's newly practical: prompt caching

Generating a context blurb for *every chunk* sounds expensive — you'd send the whole document to the LLM once per chunk. The technique becomes affordable through **prompt caching**, a feature that lets the model cache the document once and reuse it cheaply across all of that document's chunks, rather than re-processing it every time. That's what turned "contextualize every chunk" from impractical into routine.

### The results (and how to read them)

Anthropic reported, on their internal evaluations across domains like codebases, scientific papers, and fiction, reductions in the top-20 retrieval failure rate of roughly:
- **35%** from Contextual Embeddings alone,
- **49%** from Contextual Embeddings + Contextual BM25 together,
- **67%** when reranking is added on top.

These are meaningful gains, but read them as Anthropic does: as evidence the approach helps across varied data, not as a guarantee of the same numbers on *your* corpus. The technique's own guidance stresses validating on your data.

```python
# Sketch of the contextualization step (pseudocode-ish).
# For each chunk, generate a short context blurb with an LLM, then prepend it before embedding.

CONTEXT_PROMPT = """Here is the whole document:
{whole_document}

Here is a chunk from it:
{chunk}

Write a short (1-2 sentence) context that situates this chunk within the document,
to improve search retrieval. Answer with only the context."""

def contextualize(chunk, whole_document, llm):
    context = llm.complete(
        CONTEXT_PROMPT.format(whole_document=whole_document, chunk=chunk)
        # In practice, cache `whole_document` so every chunk of it reuses the cached copy.
    )
    return f"{context}\n\n{chunk}"     # prepend context, THEN embed this combined text
```

### When you might not need any of this

Anthropic notes a simpler truth worth remembering: if your entire knowledge base is small — under roughly 200,000 tokens (about 500 pages) — you may not need retrieval at all. You can just put the whole knowledge base directly into the LLM's prompt (made affordable by prompt caching) and skip chunking and retrieval entirely. Reach for chunking strategies when your corpus is too big for that.

> **Gotcha — Contextual Retrieval adds a one-time LLM cost at ingestion.** You run an LLM call per chunk to generate context (mitigated by prompt caching, but not free). It's a one-time indexing cost, not a per-query cost — but budget for it, especially on large corpora.

> **Gotcha — prepend context for indexing, but be deliberate about what the LLM finally reads.** The generated context improves *retrieval*. Whether you also keep that prepended context in the text shown to the LLM at answer time is a design choice; the context is primarily there to make the chunk findable, not necessarily to pad the final prompt.

> **Gotcha — it complements chunking, it doesn't replace it.** You still need a base chunking strategy (Hours 1–2); Contextual Retrieval is a layer applied *to* those chunks. Garbage chunks with context prepended are still garbage chunks.

### Hands-on exercise (Hour 3)

1. Take 5–10 chunks from a document where the chunks are individually ambiguous (e.g. they use pronouns or undefined references to the document's subject).
2. For each chunk, write — by hand or with an LLM — a 1–2 sentence context that situates it in the document, and prepend it to the chunk.
3. Embed both the original chunks and the contextualized chunks. Write 3–4 queries that are ambiguous without document context and compare which version retrieves the right chunk. Note where context made the difference.
4. If you have an LLM API with prompt caching, sketch how you'd cache the full document so generating context for all its chunks is cheap. Estimate the one-time cost for a 100-document corpus.

### Interview questions — Hour 3

**1. (Conceptual) What problem does Contextual Retrieval address, and how?**
It addresses chunks losing the context of their source document, which hurts both retrieval and interpretability. It fixes this by using an LLM to generate a short, chunk-specific blurb situating each chunk within its document, prepending that context to the chunk before embedding it (and before BM25 indexing), so the chunk's representation carries the document-level context it otherwise lacked.

**2. (Conceptual) What are the two sub-techniques of Contextual Retrieval, and what is each for?**
Contextual Embeddings — prepend the generated context before creating the vector embedding, improving semantic matching. Contextual BM25 — prepend the same context before building a BM25 keyword index, improving exact-term matching (names, codes, technical terms). Their results are combined with rank fusion, and reranking can be added on top.

**3. (Practical) Why is prompt caching central to making Contextual Retrieval affordable?**
Because generating a context blurb for every chunk would otherwise require sending the whole document to the LLM once per chunk, which is expensive. Prompt caching lets the model cache the document once and reuse it cheaply across all of that document's chunks, turning per-chunk contextualization from impractical into routine.

**4. (Practical) When would you skip retrieval entirely, according to the technique's guidance?**
When the entire knowledge base is small — under roughly 200,000 tokens (about 500 pages). In that case you can place the whole knowledge base directly in the LLM's prompt (made cost-effective by prompt caching) and skip chunking and retrieval altogether. Chunking strategies are for corpora too large to fit in context.

**5. (Gotcha) A colleague reports the "67% improvement" as a guarantee for their project. How do you respond?**
I'd explain that the ~67% (and the ~35%/~49% figures) come from Anthropic's internal evaluations across specific domains and are evidence the approach helps broadly, not a guaranteed result for our data. Retrieval gains depend on the corpus, queries, embedding model, and base chunking. We should implement it and measure the improvement on our own labeled queries before quoting any number.

---

## Hour 4 — Implement and Compare Strategies

### The problem we're solving

You now know several strategies. The only way to choose is to **measure** them on the same data — because the best chunking strategy is empirical, not theoretical. This hour is about building a fair comparison.

### How to compare retrieval quality

You need three things: a corpus, a **gold set**, and a metric.

A **gold set** (or evaluation set) is a collection of queries paired with the document(s) or chunk(s) that *should* be retrieved for each — the "right answers." You build it by hand: write realistic questions about your corpus and mark which passage answers each.

The standard metrics for retrieval quality:
- **Recall@k** — of the truly relevant items, what fraction appear in the top *k* retrieved results. (If every query has one right chunk, recall@5 is simply "how often the right chunk was in the top 5.") This is usually the headline number for RAG.
- **MRR** (Mean Reciprocal Rank) — rewards ranking the first relevant result higher (a hit at position 1 scores better than a hit at position 5).
- **NDCG** (Normalized Discounted Cumulative Gain) — a ranking-quality score that gives more credit for relevant results near the top.

### The comparison procedure

1. Fix everything *except* the chunking strategy: same corpus, same embedding model, same vector database, same top-*k*, same gold set. Only the chunker changes — otherwise you can't attribute differences to chunking.
2. For each strategy, chunk the corpus, embed, index, and run every gold-set query.
3. Compute recall@k (and optionally MRR/NDCG) per strategy.
4. Compare — and also look at *which* queries each strategy wins or loses, not just the averages.

```python
# A minimal recall@k harness comparing strategies on the same gold set.
def recall_at_k(retrieve_fn, gold_set, k=5):
    """gold_set: list of (query, set_of_relevant_chunk_ids)."""
    hits = 0
    for query, relevant_ids in gold_set:
        retrieved_ids = retrieve_fn(query, k)          # top-k chunk ids for this strategy
        if relevant_ids & set(retrieved_ids):          # at least one relevant chunk retrieved
            hits += 1
    return hits / len(gold_set)

# Build one retrieve_fn per chunking strategy (fixed / recursive / semantic or sentence-window),
# keeping the embedding model, vector DB, and k identical across all of them.
for name, retrieve_fn in strategies.items():
    print(name, round(recall_at_k(retrieve_fn, gold_set, k=5), 3))
```

> **Gotcha — change one variable at a time.** If you switch chunking strategy *and* embedding model *and* top-*k* at once, you can't tell what caused a score change. Hold everything constant except the one thing you're testing. This is the single most common evaluation mistake.

> **Gotcha — a tiny gold set gives noisy results.** Ten queries can swing wildly on luck. Aim for at least ~30–50 queries (more is better) so a few lucky or unlucky hits don't dominate. Also make sure the gold queries resemble *real* user questions, not just keyword echoes of the chunks.

> **Gotcha — retrieval quality ≠ final answer quality, but it caps it.** Good retrieval doesn't guarantee a good generated answer, but bad retrieval guarantees a bad one (the LLM never sees the right facts). Measuring retrieval first isolates the chunking decision from the separate question of how well the LLM writes.

### Hands-on exercise (Hour 4)

This is the capstone: apply three strategies to one corpus and compare.

1. Assemble a corpus you know well (20+ documents or a few long ones) and build a **gold set** of 30+ realistic questions, each marked with its correct passage.
2. Implement three chunking strategies from this guide — e.g. fixed-size, recursive, and one advanced strategy (semantic, sentence-window, or contextual). Keep embedding model, vector database, and top-*k* identical across all three.
3. Compute **recall@5** for each strategy using a harness like the one above. Optionally add MRR or NDCG.
4. Identify 2–3 queries where the strategies disagree and inspect *what* each retrieved. Write a short conclusion: which strategy won on your data, by how much, and your hypothesis for *why* (e.g. did contextual help the ambiguous queries? did recursive beat fixed by preserving sentences?).

### Interview questions — Hour 4

**1. (Conceptual) What is a "gold set" and why is it essential for comparing chunking strategies?**
A gold set is a collection of queries paired with the passages that should be retrieved for each — the known-correct answers. It's essential because it lets you measure retrieval quality objectively with metrics like recall@k, turning the choice of chunking strategy from an opinion into a measurement on your actual data.

**2. (Conceptual) Define recall@k and explain why it's a common headline metric for RAG retrieval.**
Recall@k is the fraction of truly relevant items that appear in the top *k* retrieved results. It's a common headline metric because RAG only works if the right context reaches the LLM: recall@k directly measures how often the relevant passage made it into the top results the model will see, which caps the system's possible answer quality.

**3. (Practical) You want to compare three chunking strategies fairly. What must you hold constant, and why?**
Everything except the chunking strategy: the corpus, embedding model, vector database, top-*k*, and gold set. If more than one variable changes at once, you can't attribute a score difference to chunking rather than, say, a different embedding model — so isolating the single variable under test is what makes the comparison valid.

**4. (Gotcha) Your evaluation on 10 queries shows strategy A beating B by a wide margin. How much should you trust it?**
Not much on its own — ten queries are too few, so a handful of lucky or unlucky hits can dominate and the gap may be noise. I'd expand the gold set to at least ~30–50 realistic queries, re-measure, and look at per-query wins/losses before concluding, since small evaluation sets give unstable results.

**5. (Conceptual) Why measure retrieval quality separately from final answer quality?**
Because they're distinct failure modes, and retrieval caps answer quality: if the right passage isn't retrieved, the LLM can't possibly answer correctly, no matter how well it writes. Measuring retrieval (recall@k, MRR, NDCG) isolates the chunking decision, so when an answer is wrong you can tell whether the cause was bad retrieval (a chunking/embedding problem) or bad generation (an LLM/prompting problem).

---

## Gotchas Summary Table

| # | Gotcha | Where | Why it bites | How to avoid it |
|---|--------|-------|--------------|-----------------|
| 1 | Maxing out chunk overlap | Fixed-size | More chunks, higher cost, duplicate results | Use ~10–20% overlap; tune, don't max |
| 2 | Expecting recursive chunking to never cut | Recursive | A unit bigger than chunk_size is still split | Set chunk_size to fit your typical semantic unit |
| 3 | Treating semantic chunking as a free upgrade | Semantic | Adds embedding cost; unpredictable chunk sizes | Evaluate it against recursive on your data |
| 4 | Chunks too small | All | Lose context needed to interpret them | Balance size, or decouple retrieval vs. reading unit |
| 5 | No store for the "big" unit | Parent-doc / sentence-window | Retrieved unit ≠ fed unit needs both kept + linked | Keep a docstore alongside the vector index |
| 6 | Mis-tuned auto-merge threshold | Auto-merging | Rarely merges (fragmented) or over-merges (bloated) | Tune the threshold on your data |
| 7 | Embedding/LLM-input divergence unexamined | Sentence-window | Matched sentence's window may be unhelpful | Inspect both sentence and window when debugging |
| 8 | Budgeting Contextual Retrieval as per-query | Contextual | It's a one-time ingestion cost (LLM per chunk) | Use prompt caching; budget the one-time indexing cost |
| 9 | Treating Contextual Retrieval as a chunker | Contextual | It's a layer on top of base chunking | Keep a solid base chunking strategy underneath |
| 10 | Quoting "67%" as a guarantee | Contextual | Those are Anthropic's internal-eval numbers | Measure the gain on your own corpus |
| 11 | Changing multiple variables when comparing | Implementation | Can't attribute score changes | Hold everything constant except the chunker |
| 12 | Tiny gold set | Implementation | Noisy, luck-dominated results | Use ≥30–50 realistic queries |
| 13 | Confusing retrieval quality with answer quality | Implementation | Different failure modes; retrieval caps answers | Measure retrieval (recall@k) separately first |

---

## Quick Reference Card

**The core tension:** a chunk small enough to *retrieve* precisely is often too small to *read* in context. Every strategy manages this trade-off.

**Basic chunking (LangChain `langchain_text_splitters`)**
```python
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
CharacterTextSplitter(separator="", chunk_size=500, chunk_overlap=50)          # fixed-size
RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)                # recommended default
# semantic (experimental):
from langchain_experimental.text_splitter import SemanticChunker
SemanticChunker(embeddings=..., breakpoint_threshold_type="percentile")
```
- **Fixed-size:** simple, fast, cuts blindly. **Recursive:** respects paragraph→sentence→word boundaries (default). **Semantic:** splits at topic shifts via embeddings; costs more, not always worth it.
- Size trade-off: too big = blurry/irrelevant; too small = no context. No universal best — measure it.

**Advanced (LlamaIndex `llama_index.core.node_parser`)** — *decouple search unit from read unit*
```python
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
SentenceWindowNodeParser.from_defaults(window_size=3)          # embed 1 sentence, read a window
HierarchicalNodeParser.from_defaults(chunk_sizes=[2048,512,128])  # + AutoMergingRetriever
```
- **Parent-document / small-to-big:** embed small children, feed the LLM the parent.
- **Auto-merging:** multi-level chunks; promote children → parent when enough are retrieved.
- **Sentence-window:** embed single sentences, swap in surrounding window at answer time (MetadataReplacementNodePostProcessor).
- (LangChain equivalent: `ParentDocumentRetriever`.) Needs a docstore for the big units.

**Contextual Retrieval (Anthropic)** — *change what you embed*
- Prepend an LLM-generated, chunk-specific context blurb to each chunk **before** embedding (Contextual Embeddings) and before BM25 indexing (Contextual BM25). Fuse results; optionally rerank.
- Reported failure-rate reductions: ~35% (embeddings) → ~49% (+BM25) → ~67% (+reranking) — *on Anthropic's data; verify on yours.*
- **Prompt caching** makes per-chunk contextualization affordable (cache the doc once).
- Small KB (<~200k tokens / ~500 pages)? Skip RAG — put the whole thing in the prompt.

**Comparing strategies (the only way to choose)**
- Build a **gold set** (≥30–50 query → correct-passage pairs).
- Measure **recall@k** (headline), optionally MRR / NDCG.
- **Hold everything constant except the chunker.** Inspect per-query wins/losses, not just averages.
- Retrieval quality caps answer quality — measure it first.

**Reference docs:** LangChain text-splitter documentation; LlamaIndex node-parser documentation; Anthropic's Contextual Retrieval post and the accompanying cookbook. Validate every strategy on your own dataset.

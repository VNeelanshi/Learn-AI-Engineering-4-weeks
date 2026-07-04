# Advanced RAG Patterns: HyDE, Multi-Query, Self-Querying, GraphRAG, and Agents

Basic retrieval — embed the query, fetch the nearest chunks, generate — works until it doesn't. This guide covers five patterns that fix specific failure modes of basic retrieval, and then a framework for choosing among them. By the end you'll know what each pattern is *for*, how to implement it, what it costs, and how to pick the right combination for a real system.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order.

> **A note on running the code:** these libraries reorganize imports and rename classes frequently, and graph/agent tooling moves especially fast. The code reflects current APIs, but if an import fails, check the library's current docs — the *patterns* are stable even when the APIs shift.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers (a **vector**) representing the *meaning* of text; similar meaning → vectors close together. An **embedding model** produces them.
- **Retrieval** = finding, for a query, the most relevant stored chunks by comparing vectors. This is **semantic search**. A **chunk** is a small passage of a document.
- **RAG** (Retrieval-Augmented Generation) = retrieve relevant text, then feed it to a **large language model** (LLM — an AI that generates text) so its answer is grounded in your documents.
- **Metadata** = structured key-value extras attached to each chunk (author, date, type) that you can **filter** on.
- **Reranking** = re-scoring retrieved candidates with a sharper model to improve their ordering.
- The pipeline this guide upgrades is **"basic RAG"**: embed the user's query → retrieve top-k chunks → put them in a prompt → generate.

The theme: basic RAG makes several silent assumptions — that the query is phrased like the answer, that one phrasing suffices, that the query has no structured constraints, that answers live in individual passages, and that one retrieval step is enough. Each pattern below removes one of those assumptions.

---

## Hour 1 — HyDE (Hypothetical Document Embeddings)

### The problem we're solving

Queries and answers often *look different*. A user asks a short, interrogative question — "What caused the 1929 crash?" — but the document that answers it is a long, declarative passage that may never use the words "caused" or "1929 crash" together. Because semantic search matches the *query's* embedding against *document* embeddings, this **query–document mismatch** can cause the right document to rank poorly: the question simply doesn't embed near the answer.

**HyDE** (Hypothetical Document Embeddings) is a clever fix: instead of embedding the question, first have an LLM *write a hypothetical answer* to it, then embed *that* and use it to search. The made-up answer "looks like" a real answer document — same style, same vocabulary — so its embedding lands much closer to the genuine answer passages than the bare question would.

### How it works

1. Take the user's query.
2. Ask an LLM to generate a plausible answer document for it (a few sentences). It doesn't matter if details are wrong or invented — this text is *only* used for searching.
3. Embed the hypothetical document.
4. Retrieve real chunks using that embedding.
5. Generate the final answer from the *real* retrieved chunks (never from the hypothetical text).

The key insight: the LLM's hypothetical answer is a better *search probe* than the question, because it inhabits the same "shape" as the documents you're searching. This is a form of **query transformation** — rewriting the query into something that retrieves better.

```python
from langchain.chains import HypotheticalDocumentEmbedder   # may live under langchain_classic in newest versions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Wrap an LLM + a base embedding model. The LLM writes a hypothetical answer,
# then `base_embeddings` embeds that answer instead of the raw query.
hyde = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(temperature=0),
    base_embeddings=OpenAIEmbeddings(),
    prompt_key="web_search",            # a built-in prompt; others exist, or pass custom_prompt
)

# Use `hyde` as the embedding function for QUERIES into your vector store.
query_vector = hyde.embed_query("What caused the 1929 stock market crash?")
# -> retrieve with query_vector against your index, then generate from the real chunks.
```

### When it helps (and when it doesn't)

HyDE shines when queries are short, vague, or phrased very differently from your documents, and when you have no labeled training data (it works zero-shot). It tends *not* to help — and can even hurt — when queries are already long and detailed (they're good probes already), or in narrow domains where the LLM's hypothetical answer is likely to be confidently wrong in ways that mislead retrieval.

> **Gotcha — the hypothetical answer can hallucinate, and that mostly doesn't matter — except when it does.** Because you only use the hypothetical text to *search* (not to answer), minor invented details are usually harmless. But if the LLM hallucinates a *wrong entity or fact central to the query* (a wrong name, date, or place), the search probe points at the wrong region of vector space and retrieval degrades. Watch for this on niche or factual queries.

> **Gotcha — HyDE adds an LLM call to every query.** You're generating text before every search, which adds latency and cost on the query path (not just at indexing time). For high-traffic systems, weigh whether the recall gain justifies the per-query LLM call.

### Hands-on exercise (Hour 1)

1. Take a small corpus and write 5 short, question-style queries whose answer passages use different wording than the questions.
2. For each query, retrieve top-5 two ways: (a) embedding the raw query, and (b) embedding an LLM-generated hypothetical answer (HyDE).
3. Compare which approach surfaced the correct passage higher. Note any case where HyDE *hurt* (e.g. the hypothetical answer invented a wrong fact).
4. Write 3–4 sentences on which query types benefited most from HyDE for your data.

### Interview questions — Hour 1

**1. (Conceptual) What problem does HyDE address?**
It addresses query–document mismatch: short, interrogative queries often embed far from the long, declarative passages that answer them, so the right document ranks poorly. HyDE fixes this by generating a hypothetical answer to the query and embedding that instead, since the hypothetical answer resembles real answer documents and lands closer to them in vector space.

**2. (Conceptual) Why doesn't it matter much if the hypothetical document contains errors?**
Because the hypothetical document is used only as a search probe to retrieve real chunks, not as the final answer — the answer is generated from the genuine retrieved documents. So minor invented details don't reach the user. The exception is a hallucinated central fact (wrong entity/date), which can point retrieval at the wrong region and degrade results.

**3. (Practical) For what kind of query is HyDE most valuable, and when would you skip it?**
Most valuable for short, vague, or oddly-phrased queries, and when you have no labeled data (it's zero-shot). Skip it when queries are already long and detailed (they're good probes as-is) or in narrow domains where the LLM is likely to confidently hallucinate misleading specifics, and when the per-query LLM latency/cost isn't justified by the recall gain.

**4. (Conceptual) HyDE is described as a "query transformation." What does that mean?**
It means HyDE rewrites or replaces the original query with something that retrieves more effectively — here, swapping the literal question for a generated hypothetical answer before embedding. The user's intent is preserved, but the text actually sent to the retriever is transformed to better match the documents.

**5. (Gotcha) What's the cost consideration that makes HyDE different from an indexing-time technique?**
HyDE runs an LLM call on the *query* path — once per user query — adding latency and cost to every search, unlike techniques whose cost is paid once at indexing. For high-traffic systems this per-query overhead must be weighed against the retrieval improvement it provides.

---

## Hour 2 — Multi-Query Retrieval

### The problem we're solving

A single query is a single phrasing, and phrasing is fragile. "How does the heart pump blood?" and "What is the mechanism of cardiac circulation?" mean nearly the same thing, but they embed differently — so each retrieves a somewhat different set of chunks, and each *misses* chunks the other would find. Relying on one phrasing means you only ever see one slice of the relevant material.

**Multi-query retrieval** fixes this by generating *several* reworded versions of the query with an LLM, retrieving for each, and combining the results. Casting the net from multiple angles improves **recall** (the fraction of relevant chunks you actually find).

### How it works

1. Send the user's query to an LLM and ask for several alternative phrasings (different words, same intent).
2. Run retrieval for each phrasing (including, optionally, the original).
3. **Union** the results — combine all retrieved chunks and remove duplicates.
4. Generate from the combined set (often after reranking, since you now have more candidates).

```python
from langchain.retrievers.multi_query import MultiQueryRetriever   # may live under langchain_classic in newest versions
from langchain_openai import ChatOpenAI

# Generates several query variations (3 by default), retrieves for each, unions + dedupes.
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(temperature=0),
)

docs = retriever.invoke("How does the heart pump blood?")
# Internally generates variations like:
#  - "What is the mechanism by which the heart circulates blood?"
#  - "Describe how cardiac muscle moves blood through the body."
#  - "Explain the pumping action of the heart."
```

A popular variant, **RAG-Fusion**, goes one step further: instead of a plain union, it fuses the multiple ranked lists with **Reciprocal Rank Fusion** (a rank-based method that rewards chunks ranked highly across several lists), producing a single, better-ordered result set.

### When it helps (and the trade-off)

Multi-query helps whenever a single phrasing is too narrow — broad or ambiguous questions, or corpora where the same idea is expressed many ways. The trade-off is the flip side of casting a wider net: more retrieval calls (latency and cost), and a wider net pulls in some irrelevant chunks too, which can *lower precision*. That's why multi-query pairs naturally with reranking.

> **Gotcha — more recall can mean less precision.** Generating more queries surfaces more candidates, including off-target ones. Without a reranking or filtering step, the extra noise can dilute the context you feed the LLM. Treat multi-query as a recall booster that usually needs a precision step (reranking) after it.

> **Gotcha — deduplicate by a stable id.** Different query variations will retrieve the *same* chunk repeatedly. You must merge duplicates by a consistent chunk identifier; otherwise the same passage occupies several slots in your top-k and crowds out diversity.

> **Gotcha — each variation is another retrieval (and an upfront LLM call).** You pay for one LLM call to generate variations, then N retrievals. On large vector stores or high traffic, that multiplies query-time work. Tune the number of variations.

### Hands-on exercise (Hour 2)

1. Pick a broad query over your corpus. Manually write three rephrasings of it.
2. Retrieve top-5 for each rephrasing separately and note how the result sets differ — confirm each finds chunks the others missed.
3. Union and deduplicate the results, then (if you have a reranker) rerank to a final top-5. Compare to single-query retrieval.
4. Write 2–3 sentences on whether the wider net improved coverage and whether it pulled in noise that reranking had to remove.

### Interview questions — Hour 2

**1. (Conceptual) What problem does multi-query retrieval solve?**
It solves the fragility of a single query phrasing: because differently-worded but equivalent queries embed differently and retrieve different chunks, one phrasing misses relevant material. Multi-query generates several rephrasings, retrieves for each, and combines the results, improving recall by approaching the question from multiple angles.

**2. (Conceptual) What is RAG-Fusion and how does it differ from plain multi-query?**
RAG-Fusion is a multi-query variant that, instead of taking a plain union of results, fuses the several ranked lists using Reciprocal Rank Fusion — rewarding chunks that rank highly across multiple query variations. The difference is the combination step: plain multi-query unions and dedupes, while RAG-Fusion produces a single re-ranked result set from the multiple lists.

**3. (Practical) Why is multi-query often paired with a reranker?**
Because multi-query boosts recall by widening the candidate net, which also pulls in irrelevant chunks and can lower precision. A reranker re-scores the larger candidate set and keeps only the most relevant, restoring precision. So the two complement each other: multi-query finds more, reranking filters to the best.

**4. (Gotcha) After adding multi-query, the same passage appears several times in your results. What went wrong and how do you fix it?**
Different query variations retrieved the same chunk, and the results weren't deduplicated by a stable identifier, so the passage occupies multiple top-k slots. Fix it by merging results on a consistent chunk id before taking the final top-k, so each unique passage appears once and doesn't crowd out other relevant material.

**5. (Practical) What are the cost implications of multi-query retrieval?**
It adds one LLM call to generate the query variations plus one retrieval per variation, multiplying query-time work relative to a single search. On large vector stores or high-traffic systems this can be significant, so the number of variations should be tuned to balance recall gains against latency and cost.

---

## Hour 3 — Self-Querying Retrievers

### The problem we're solving

Many real queries contain *two kinds* of information at once. "Find papers about transformers published after 2021" has a **semantic** part ("about transformers" — a meaning to match) and a **structured** part ("published after 2021" — a hard constraint on a metadata field). Basic semantic search embeds the *whole* sentence and ignores the constraint, so it might return a perfectly on-topic 2019 paper — semantically relevant but violating the explicit filter.

A **self-querying retriever** solves this by using an LLM to *parse* the natural-language query into its two components: a clean semantic query string, and a structured **metadata filter**. It then runs the vector search with that filter applied — honoring both the meaning and the constraint.

### How it works

You tell the retriever, up front, what metadata fields exist (their names, types, and what they mean) and what the documents contain. Then, for each user query, the LLM:
1. Extracts the semantic search string (e.g. "transformers").
2. Extracts a structured filter from the constraints (e.g. `year > 2021`).
3. The retriever runs the vector search for the semantic string *with* the metadata filter.

```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo   # import path may vary by version
from langchain_openai import ChatOpenAI

# Describe the metadata fields so the LLM knows what filters it can build
metadata_field_info = [
    AttributeInfo(name="year",  description="The year the paper was published", type="integer"),
    AttributeInfo(name="topic", description="The paper's primary topic",        type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(temperature=0),
    vectorstore=vectorstore,
    document_contents="Abstracts of machine learning research papers",
    metadata_field_info=metadata_field_info,
)

# The LLM splits this into: semantic="transformers", filter=(year > 2021)
docs = retriever.invoke("papers about transformers published after 2021")
```

### When it helps

Self-querying is the right tool whenever your documents have **rich, well-defined metadata** and users ask free-form questions that naturally include constraints — dates, authors, categories, price ranges, ratings, document types. It turns "blue running shoes under $100 reviewed after January" into a semantic search for "running shoes" filtered by `color = blue AND price < 100 AND review_date > ...`, without you writing brittle parsing code.

> **Gotcha — the LLM can only build filters it's been told about.** The field descriptions you provide are what the LLM uses to construct filters; vague or missing descriptions lead to wrong or absent filters. Describe each metadata field precisely (name, type, meaning, and example values where helpful).

> **Gotcha — the vector store must support the operators the LLM emits.** If the LLM generates a filter using an operator or field your vector database doesn't support, the query errors or silently misbehaves. Confirm your store supports the comparison/logical operators the retriever will produce, and that field types match.

> **Gotcha — the LLM can hallucinate fields or malformed filters.** It may invent a metadata field that doesn't exist or produce an invalid filter expression. Validate generated filters, constrain the allowed fields tightly, and handle parse failures gracefully (e.g. fall back to unfiltered search).

### Hands-on exercise (Hour 3)

1. Build a small corpus where each chunk has 2–3 metadata fields (e.g. `year`, `category`, `rating`).
2. Define `metadata_field_info` with clear descriptions and create a `SelfQueryRetriever`.
3. Ask natural-language queries that mix meaning and constraints ("highly rated articles about climate from 2023"). Inspect the filter the retriever generated and confirm results respect both parts.
4. Deliberately ask about a field you *didn't* define and observe how the retriever handles it. Write 2–3 sentences on why precise metadata descriptions matter.

### Interview questions — Hour 3

**1. (Conceptual) What does a self-querying retriever do that basic semantic search can't?**
It separates a natural-language query into a semantic search string and a structured metadata filter using an LLM, then applies the filter during retrieval. Basic semantic search embeds the whole query and ignores explicit constraints, so it can return semantically relevant results that violate stated conditions (like a date range); self-querying honors both meaning and constraints.

**2. (Conceptual) What information must you give a self-querying retriever ahead of time, and why?**
You provide a description of the document contents and a schema of the metadata fields — each field's name, type, and meaning. The LLM uses this to decide which filters it can construct and how, so accurate, precise descriptions are essential; without them the LLM can't build correct filters from the user's constraints.

**3. (Practical) Give an example use case where self-querying is clearly the right pattern.**
Product or document search with rich metadata — e.g. e-commerce ("blue running shoes under $100 reviewed this year") or a research library ("papers about transformers published after 2021 by a specific group"). The queries naturally combine a semantic topic with structured constraints (price, date, category), which self-querying maps to a filtered vector search automatically.

**4. (Gotcha) The retriever generates a filter your vector database rejects. What's the likely issue?**
The LLM emitted an operator or referenced a field type that the vector store doesn't support, or referenced a field that doesn't exist. The fix is to ensure the store supports the operators the retriever produces, that field types match the schema, to constrain the allowed fields tightly, and to validate or gracefully handle invalid generated filters.

**5. (Gotcha) Why must you validate the filters a self-querying retriever produces?**
Because the LLM can hallucinate non-existent metadata fields or produce malformed filter expressions, which cause errors or silently wrong results. Validating generated filters (and falling back to unfiltered search on failure) keeps the system robust, rather than trusting the LLM to always emit a correct, supported filter.

---

## Hour 4 — GraphRAG (Knowledge Graphs for RAG)

### The problem we're solving

Basic RAG retrieves individual passages, which works for questions answered *in* a passage ("What's the refund window?"). But it fails on two kinds of questions: **global / sensemaking** questions about the whole corpus ("What are the main themes across these 5,000 documents?") and **multi-hop** questions that require *connecting* facts scattered across many documents ("How is person A related to project C?"). No single passage holds the answer, so nearest-neighbor retrieval can't assemble it.

**GraphRAG** (a method from Microsoft Research) tackles this by building a **knowledge graph** from the corpus and using its structure. A **knowledge graph** is a network of **entities** (people, places, organizations, concepts — the nodes) connected by **relationships** (the edges), e.g. *Alice —works_on→ Project C*. Reasoning over this graph lets the system "connect the dots" and summarize across the whole dataset in ways flat retrieval cannot.

### How it works

**Indexing (expensive, done once):**
1. Split the corpus into text units (chunks).
2. Use an LLM to extract **entities, relationships, and claims** from each unit, building an entity knowledge graph (entities deduplicated by similarity; edges weighted by how often things co-occur).
3. Run **community detection** (the **Leiden** algorithm — a method that finds clusters of densely-connected nodes) to group related entities into **communities** at multiple levels of granularity. A **community** is a tightly-interconnected cluster of the graph.
4. Use an LLM to write a **community summary** (also called a community report) for each community — a summary of what that cluster of entities and relationships is about.

**Querying (two modes):**
- **Global search** — for whole-corpus, sensemaking questions. The system feeds the relevant community summaries to the LLM (often via a *map-reduce* process: summarize over batches, then combine) to synthesize an answer about the dataset as a whole.
- **Local search** — for entity-specific questions. The system finds the relevant entities (via vector search on entity descriptions) and "fans out" to their neighbors, related chunks, and communities to gather focused context.

(A hybrid mode, **DRIFT search**, starts global and then drills into local follow-ups.)

```bash
# GraphRAG ships as a library + CLI
pip install graphrag

graphrag init --root ./ragproject          # scaffold config (settings.yaml) + prompts
# put your documents in ./ragproject/input, set your LLM API key in the config/.env
graphrag index --root ./ragproject          # build the graph, communities, and summaries (the costly step)

# Query: GLOBAL for whole-corpus questions, LOCAL for entity-specific ones
graphrag query --root ./ragproject --method global "What are the recurring themes across the documents?"
graphrag query --root ./ragproject --method local  "What is Project C and who works on it?"
```

### When it helps (and the big caveat: cost)

GraphRAG is worth it when your questions are genuinely global ("summarize the themes," "catch me up") or require multi-hop reasoning across many documents, typically on corpora large enough to have rich structure (roughly 1,000+ documents). For simple point lookups, **basic vector RAG is better and far cheaper** — GraphRAG's advantage is structure and synthesis, not pinpoint fact retrieval.

The caveat is **indexing cost**. Extracting a graph and LLM-summarizing every community across a large corpus involves a very large number of LLM calls, which can be expensive and slow. This has driven cheaper variants — **LazyGraphRAG**, **LightRAG**, and **Fast GraphRAG** — that defer or skip the community-summarization step and cut indexing cost dramatically (reported reductions of many-fold). If GraphRAG's quality appeals but its cost doesn't, evaluate these.

> **Gotcha — GraphRAG is the wrong tool for simple lookups.** If your queries are mostly "what does the document say about X," the graph machinery is expensive overkill and plain vector RAG (or hybrid search) will be faster, cheaper, and just as accurate. Match the pattern to the *question shape*: graph for global/multi-hop, vector for point lookups. Many production stacks route by query type and use both.

> **Gotcha — indexing cost and time are substantial.** Building the graph and summaries is the costly part and scales with corpus size and model choice. Budget for it, use cheaper models for the easier sub-tasks (e.g. relevance rating) where possible, and consider the lighter-weight variants before committing.

> **Gotcha — reported quality gains deserve your own evaluation.** GraphRAG's headline improvements come from specific evaluations, and independent audits have questioned how large the gains are under rigorous testing. Treat published win-rates as motivation to try it, not proof it will win on your data — validate on your own questions.

### Hands-on exercise (Hour 4)

1. Assemble a modest corpus with interconnected entities (e.g. a set of related articles, meeting notes, or a small wiki). Install `graphrag` and run `graphrag init`.
2. Index it (start *small* to control cost), then run a **global** query ("What are the main themes?") and a **local** query about a specific entity.
3. Compare the global answer to what plain vector RAG gives for the same global question — note whether GraphRAG synthesized across the corpus where vector RAG only fetched scattered passages.
4. Write 3–4 sentences classifying which of your likely real questions are "global/multi-hop" (suited to GraphRAG) versus "point lookups" (suited to vector RAG), and note the cost you observed.

### Interview questions — Hour 4

**1. (Conceptual) What two kinds of questions does GraphRAG handle that basic vector RAG struggles with?**
Global/sensemaking questions about the whole corpus ("what are the main themes?") and multi-hop questions that require connecting facts scattered across many documents. Basic RAG retrieves individual passages, but neither of these is answered by any single passage, so nearest-neighbor retrieval can't assemble the answer; GraphRAG uses graph structure and community summaries to synthesize across the dataset.

**2. (Conceptual) Walk through GraphRAG's indexing process.**
It chunks the corpus, uses an LLM to extract entities, relationships, and claims to build an entity knowledge graph (deduplicating entities and weighting edges by co-occurrence), runs Leiden community detection to cluster related entities into multi-level communities, and uses an LLM to write a summary (community report) for each community. These structures are then used at query time.

**3. (Conceptual) What's the difference between GraphRAG's global and local search?**
Global search answers whole-corpus, sensemaking questions by feeding relevant community summaries to the LLM (often via map-reduce) to synthesize a dataset-wide answer. Local search answers entity-specific questions by finding relevant entities and fanning out to their neighbors, related chunks, and communities for focused context. Global is for breadth; local is for depth around specific entities.

**4. (Practical) When should you NOT use GraphRAG?**
When questions are mostly simple point lookups answered by a single passage. The graph extraction and community summarization make GraphRAG expensive and slow to index, so for pinpoint retrieval, plain vector or hybrid RAG is faster, cheaper, and equally accurate. GraphRAG's value is in global synthesis and multi-hop reasoning, not fact lookup.

**5. (Gotcha) A stakeholder wants GraphRAG everywhere because of its published accuracy gains. What's your measured response?**
I'd note that GraphRAG's main cost is expensive, slow indexing, and that its advantage is specific to global/multi-hop questions — for point lookups, vector RAG is better and cheaper. I'd also flag that independent audits question how large the published gains are under rigorous evaluation, so we should validate on our own questions and consider cheaper variants (LazyGraphRAG, LightRAG) and a query-type router that uses graph and vector approaches where each fits.

---

## Hour 5 — Agentic RAG

### The problem we're solving

Basic RAG is a fixed assembly line: every query triggers exactly one retrieval, then one generation. That rigidity breaks down for real questions. Some questions need *no* retrieval (a greeting, a math step). Some need retrieval from *different sources* depending on the topic. Some are *compound* ("compare our refund policy to our shipping timelines and flag conflicts") and need to be broken into parts. Some need *iteration* — retrieve, realize it's insufficient, retrieve again. A fixed pipeline can't make any of these decisions.

**Agentic RAG** makes retrieval a *decision* rather than a fixed step. It uses an **agent** — an LLM that can choose to call **tools** (functions or services it has access to) and reason about the results — and treats **retrieval as one of those tools**. The agent decides *whether* to retrieve, *which* source to query, *how many* times, and *how* to combine results, planning its approach to the specific question.

### Core concepts

- An **agent** is an LLM equipped with tools and a loop: it reads the query, decides on an action (often calling a tool), observes the result, and repeats until it can answer.
- A **tool** is a capability the agent can invoke — a Python function, an API, or, crucially here, a **retrieval/query engine** wrapped as a tool.
- **Query planning** is the agent breaking a complex question into sub-steps (e.g. sub-questions) and sequencing tool calls to answer it.
- **Routing** is the agent (or a router component) choosing *which* of several retrieval tools/sources fits the question.

### Implementing retrieval-as-a-tool

In LlamaIndex (a RAG/agent framework), you wrap a RAG **query engine** as a `QueryEngineTool` and hand it to an agent. Two agent types: a **FunctionAgent** (uses the LLM's native tool-calling) and a **ReActAgent** (uses step-by-step "reason then act" prompting, working even with models lacking native tool calling).

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# Build a RAG query engine over some documents
docs = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

# Wrap the RAG engine as a TOOL the agent may choose to call
docs_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="company_docs",
    description="Answers questions about the company's internal policies and documentation.",
)

agent = FunctionAgent(tools=[docs_tool], llm=OpenAI(model="gpt-4o-mini"))
ctx = Context(agent)   # holds session state/memory across turns

# The agent decides whether and how to use the tool (async API)
response = await agent.run(
    "Summarize our refund policy and compare it to our shipping timelines.",
    ctx=ctx,
)
```

For richer setups: give the agent *multiple* query-engine tools (one per data source) and it will **route** to the right one; or use a **sub-question** approach that decomposes a compound query into sub-questions answered separately and then synthesized; or compose several specialized agents into a **multi-agent workflow** where each handles part of the task and can hand off to others.

### When it helps (and the trade-off)

Agentic RAG is the right pattern when questions are heterogeneous, span multiple sources, are compound, or require multi-step reasoning — situations a single fixed retrieval can't serve. The trade-off is significant: agents make *more LLM calls* (higher latency and cost), behave *less predictably* (the path varies per query), and are *harder to debug and evaluate* than a deterministic pipeline. You're buying flexibility with complexity.

> **Gotcha — agents add latency, cost, and nondeterminism.** Each reasoning step and tool call is another LLM round-trip, and the agent may take different paths on different runs. Don't reach for an agent when a fixed pipeline would do; use it only when the question genuinely requires decisions. Simpler is more reliable.

> **Gotcha — tool descriptions are load-bearing.** The agent chooses tools based on their *descriptions*. Vague or overlapping descriptions cause it to pick the wrong tool or none. Write precise, distinct descriptions for each tool (what it covers, when to use it), and test that the agent routes correctly.

> **Gotcha — bound the loop.** An agent can loop — retrieving, deciding it's insufficient, retrieving again — and occasionally fail to terminate or rack up cost. Set limits (maximum steps/iterations), add guardrails, and monitor for runaway loops. Evaluate agentic systems on whether they reach the goal, not just on output text.

### Hands-on exercise (Hour 5)

1. Build two small RAG query engines over two *different* topics (e.g. "HR policies" and "engineering docs"). Wrap each as a `QueryEngineTool` with a clear, distinct description.
2. Give both tools to a single agent. Ask a question that clearly belongs to one source and confirm the agent routes to the correct tool (inspect which tool it called).
3. Ask a *compound* question spanning both sources and observe the agent make multiple tool calls and synthesize.
4. Ask a question needing *no* retrieval (e.g. simple arithmetic or a greeting) and confirm the agent answers without calling a retrieval tool. Write 3–4 sentences on where the agent's flexibility helped and where it added latency or unpredictability.

### Interview questions — Hour 5

**1. (Conceptual) How does agentic RAG differ fundamentally from basic RAG?**
Basic RAG is a fixed pipeline: every query triggers exactly one retrieval and one generation. Agentic RAG makes retrieval a decision made by an LLM agent that can call tools — including retrieval/query engines — choosing whether to retrieve, which source to use, how many times, and how to combine results. It replaces a fixed assembly line with dynamic, query-specific planning.

**2. (Conceptual) What does it mean to treat "retrieval as a tool," and why is it powerful?**
It means wrapping a retrieval/query engine as a capability the agent can choose to invoke, rather than a mandatory step. It's powerful because the agent can decide *not* to retrieve when unnecessary, pick among multiple sources, retrieve iteratively, and combine retrieval with other tools (calculations, APIs) — handling heterogeneous, compound, or multi-step questions a fixed pipeline can't.

**3. (Practical) Give an example of a question that needs agentic RAG rather than basic RAG.**
A compound, multi-source question like "Compare our refund policy to our shipping timelines and flag any conflicts" — it requires retrieving from (possibly) different sources, reasoning across the results, and synthesizing, which a single fixed retrieval step can't do. Questions that sometimes need no retrieval, or that require iterative retrieval, are other examples.

**4. (Gotcha) Why shouldn't you use an agent for every RAG use case?**
Because agents add latency and cost (each step is another LLM call), behave nondeterministically (paths vary per run), and are harder to debug and evaluate than a fixed pipeline. For straightforward single-retrieval questions, a deterministic pipeline is faster, cheaper, and more reliable. Agents should be reserved for questions that genuinely require decisions and multi-step reasoning.

**5. (Gotcha) An agent keeps choosing the wrong retrieval tool. What's the most likely cause and fix?**
The tool descriptions are probably vague or overlapping, since the agent selects tools based on their descriptions. The fix is to write precise, distinct descriptions stating what each tool covers and when to use it, and to test routing on representative queries. Bounding iterations and monitoring tool-call accuracy also help diagnose and contain the problem.

---

## Hour 6 — Pattern Selection for Your Capstone

### The problem we're solving

You now have five patterns. The mistake would be to stack them all — every pattern adds latency, cost, and complexity. The skill is **matching each pattern to a specific failure mode**, adding only what a measured problem demands. This hour is about choosing deliberately and sketching a design for a real corpus.

### Match the pattern to the problem

Each pattern fixes a *different* weakness of basic RAG. Diagnose your failure mode, then apply the matching pattern:

| If basic RAG fails because… | The query/data looks like… | Reach for… |
|---|---|---|
| Queries don't embed near their answers | Short, vague, oddly-phrased questions | **HyDE** (query → hypothetical answer → embed) |
| One phrasing misses relevant chunks | Broad/ambiguous questions; varied wording in docs | **Multi-query** (or RAG-Fusion) |
| Queries carry hard constraints that get ignored | "X about Y after 2021 / under $100 / by author Z" | **Self-querying** (semantic + metadata filter) |
| Answers require whole-corpus synthesis or connecting scattered facts | "Main themes?", "How is A related to C?" | **GraphRAG** (mind cost; consider LightRAG) |
| One fixed retrieval can't serve the question | Heterogeneous, compound, multi-source, multi-step | **Agentic RAG** (retrieval as a tool) |

### Key principles for choosing

- **Start simple.** A solid baseline — good chunking, hybrid (keyword + semantic) search, and a reranker — solves most cases. Add advanced patterns only where this baseline measurably fails.
- **The patterns are largely composable.** They're mostly orthogonal: you can run multi-query *and* self-querying *and* reranking together, or use agentic RAG with HyDE inside one of its tools. But each layer costs latency and money — compose intentionally, not reflexively.
- **Measure before and after.** For each pattern you consider, build a small evaluation set (queries with known-correct results) and check whether the pattern actually improves retrieval/answer quality on *your* data. If it doesn't move the metric, don't ship it.
- **Mind the cost surface.** HyDE and multi-query add per-query LLM/retrieval calls; GraphRAG adds heavy indexing cost; agentic RAG adds variable per-query calls and nondeterminism. Weigh each against its benefit.

> **Gotcha — don't stack patterns reflexively.** Each added pattern compounds latency, cost, and failure surface. Three patterns that each "might help" can together make a system slow, expensive, and hard to debug while barely beating the baseline. Add one pattern at a time, measure, keep only what earns its place.

> **Gotcha — the baseline matters more than the patterns.** No advanced pattern compensates for bad chunking, a weak embedding model, or missing hybrid search. Get the foundation right first; advanced patterns are refinements, not rescues.

### Hands-on exercise (Hour 6 — capstone design)

This is the capstone: choose patterns for a real corpus and sketch the design.

1. Describe your corpus and its likely questions: size, document types, whether questions are mostly point lookups, global/sensemaking, constraint-laden, compound, or multi-source. Note any rich metadata.
2. Establish the baseline you'd build first (chunking strategy, embedding model, hybrid search, reranker).
3. For each likely *failure mode* of that baseline, identify which pattern from the table addresses it — and justify why, naming the trade-off you accept.
4. Sketch the end-to-end design (a diagram or numbered flow): how a query travels from input to answer, which patterns engage and when. Explicitly list patterns you *considered and rejected*, and why.
5. Define how you'd *evaluate* whether each chosen pattern earns its place (the metric and a small gold set).

### Interview questions — Hour 6

**1. (Conceptual) Why is "add every advanced pattern" the wrong approach to RAG design?**
Because each pattern adds latency, cost, and failure surface, so stacking them can make a system slow, expensive, and hard to debug while barely improving over the baseline. Patterns fix *specific* failure modes; applying ones your system doesn't suffer from adds complexity for no benefit. The right approach is to diagnose the actual failure mode and add only the matching pattern.

**2. (Practical) Your users mostly ask constraint-heavy questions like "reports about X from last quarter." Which pattern, and why?**
Self-querying retrieval, because these queries combine a semantic topic ("about X") with a structured constraint ("from last quarter" → a date filter) that basic semantic search would ignore. A self-querying retriever uses an LLM to split the query into a semantic string plus a metadata filter and applies the filter during retrieval, honoring both parts — provided the documents have well-described date metadata.

**3. (Practical) A product needs both "summarize the key themes" and "what's our exact refund window" answered well. How would you design for that?**
Route by question shape: use GraphRAG (or a lighter graph variant) for the global "summarize themes" questions, which need whole-corpus synthesis, and basic vector or hybrid RAG for the point-lookup "exact refund window" questions, which a single passage answers cheaply and accurately. An agent or router can classify the query and dispatch to the appropriate retrieval approach.

**4. (Conceptual) The advanced patterns are described as "composable." What does that mean and what's the catch?**
It means they're largely orthogonal and can be combined — e.g. multi-query plus self-querying plus reranking, or HyDE inside an agent's tool — because each addresses a different weakness. The catch is that every layer adds latency, cost, and complexity, so composition should be deliberate and measured, not reflexive; combine only patterns that each earn their keep on your data.

**5. (Practical) How do you decide whether a given pattern "earns its place" in your system?**
Build a small evaluation set of queries with known-correct results, measure retrieval/answer quality on your data with and without the pattern, and keep the pattern only if it meaningfully improves the metric enough to justify its added latency and cost. If it doesn't move the metric, it doesn't ship — patterns are validated empirically, not assumed.

---

## Gotchas Summary Table

| # | Gotcha | Pattern | Why it bites | How to avoid it |
|---|--------|---------|--------------|-----------------|
| 1 | Hallucinated central fact in the hypothetical doc | HyDE | Wrong entity/date points the probe at the wrong region | Watch niche/factual queries; the hypothetical is only a probe |
| 2 | Per-query LLM call overhead | HyDE | Adds query-path latency and cost | Weigh recall gain vs. cost for high-traffic systems |
| 3 | More recall, less precision | Multi-query | Wider net pulls in off-target chunks | Pair with a reranking/filtering step |
| 4 | Duplicate chunks across variations | Multi-query | Same passage fills multiple top-k slots | Deduplicate by a stable chunk id |
| 5 | Poor metadata descriptions | Self-querying | LLM can't build correct filters | Describe each field precisely (name, type, meaning) |
| 6 | Unsupported filter operators | Self-querying | Generated filter errors or misbehaves | Confirm the store supports the operators/types produced |
| 7 | Hallucinated/malformed filters | Self-querying | Invalid fields or expressions | Constrain allowed fields; validate; fall back gracefully |
| 8 | Using GraphRAG for point lookups | GraphRAG | Expensive overkill vs. vector RAG | Match pattern to question shape; route by query type |
| 9 | Underestimating indexing cost/time | GraphRAG | Graph + community summaries are costly | Budget it; use cheaper models per sub-task; try LightRAG/LazyGraphRAG |
| 10 | Trusting published win-rates | GraphRAG | Audits question the gains | Validate on your own questions |
| 11 | Agents everywhere | Agentic RAG | Latency, cost, nondeterminism | Use agents only when decisions are genuinely needed |
| 12 | Vague tool descriptions | Agentic RAG | Agent picks the wrong tool/none | Write precise, distinct tool descriptions; test routing |
| 13 | Unbounded agent loops | Agentic RAG | Runaway cost / non-termination | Cap iterations; add guardrails; monitor |
| 14 | Stacking patterns reflexively | Selection | Compounds cost/complexity for little gain | Add one at a time, measure, keep only what earns its place |
| 15 | Neglecting the baseline | Selection | Patterns can't rescue bad chunking/embeddings | Get chunking + hybrid + reranking right first |

---

## Quick Reference Card

**Each pattern removes one assumption of basic RAG (embed query → retrieve top-k → generate).**

**HyDE** — *the query doesn't embed near its answer*
```python
from langchain.chains import HypotheticalDocumentEmbedder
hyde = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, prompt_key="web_search")
qvec = hyde.embed_query("short question")   # embed an LLM-written hypothetical answer instead
```
Generate the *answer* from real retrieved chunks, never the hypothetical text. Best for short/vague queries.

**Multi-query** — *one phrasing misses chunks*
```python
from langchain.retrievers.multi_query import MultiQueryRetriever
r = MultiQueryRetriever.from_llm(retriever=vs.as_retriever(), llm=llm)   # ~3 variations, unions results
```
Boosts recall; pair with reranking for precision; dedupe by id. RAG-Fusion = multi-query + RRF.

**Self-querying** — *hard constraints get ignored*
```python
from langchain.retrievers.self_query.base import SelfQueryRetriever
r = SelfQueryRetriever.from_llm(llm, vs, document_contents="...", metadata_field_info=[...])
r.invoke("about Y published after 2021")    # LLM → semantic string + metadata filter
```
Needs rich, well-described metadata; validate generated filters.

**GraphRAG** — *answers need whole-corpus synthesis / multi-hop*
```bash
pip install graphrag
graphrag init --root ./proj && graphrag index --root ./proj           # build graph + Leiden communities + summaries
graphrag query --root ./proj --method global "main themes?"           # whole-corpus
graphrag query --root ./proj --method local  "what is entity X?"      # entity-centric
```
Global vs local search. Costly indexing — consider LightRAG/LazyGraphRAG. Not for point lookups.

**Agentic RAG** — *one fixed retrieval can't serve the question*
```python
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent   # or ReActAgent
tool = QueryEngineTool.from_defaults(query_engine=index.as_query_engine(), name="docs", description="...")
agent = FunctionAgent(tools=[tool], llm=OpenAI(model="gpt-4o-mini"))
await agent.run("compound, multi-source question", ctx=Context(agent))
```
Retrieval-as-a-tool; routing, query planning, multi-agent. Adds latency/cost/nondeterminism; bound the loop; precise tool descriptions.

**Choosing (Hour 6)**
- Diagnose the failure mode → apply the *matching* pattern (see the table above).
- **Start simple:** chunking + hybrid search + reranker solves most cases.
- Patterns are **composable** but each costs latency/money — add one at a time, **measure** on a gold set, keep only what earns its place.
- The **baseline** matters more than any pattern.

**Reference docs:** the HyDE paper and explainers; LangChain's MultiQueryRetriever and SelfQueryRetriever docs; the Microsoft GraphRAG repository and "From Local to Global" paper; LlamaIndex agent documentation. Validate every pattern on your own corpus.

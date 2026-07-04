# Choosing a Vector Database: Weaviate, Chroma, and pgvector

Three vector databases, three completely different philosophies — and a framework for deciding which one fits a given project. By the end you'll understand a feature-rich open-source option, a dead-simple local-first option, and a "just use the database you already have" option, and you'll be able to defend a choice between them (and the major managed alternatives) on the dimensions that actually matter.

This guide assumes **zero prior knowledge**. Every term is defined the first time it appears. Work through it in order.

> **A note on running the code:** vector database client libraries change often, and two of these (Weaviate especially) have had major rewrites. The code here reflects current APIs, but if something doesn't match what you see, the official Quickstart for that database is the source of truth — trust the live docs over any tutorial when versions disagree.

## What you need to know first (60-second primer)

- An **embedding** is a list of numbers — a **vector** — that represents the *meaning* of a piece of text. Text with similar meaning produces vectors that are numerically close together. A trained program called an **embedding model** turns text into these vectors.
- **Similarity search** (also called **vector search** or **semantic search**) means: given a query vector, find the stored vectors closest to it. "Closest" is measured by a **distance/similarity metric** — most commonly **cosine** (which compares the *angle* between vectors, i.e. their meaning, ignoring their length).
- A **vector database** is a system built to store many vectors and run similarity search over them quickly.
- **Keyword search** (also called **lexical search**) matches the literal words instead of the meaning. **Hybrid search** combines semantic + keyword search to get the strengths of both.

That's enough to begin. Each database below stores vectors and searches them — the differences are in *how you run it*, *what extra features it gives you*, and *what it costs you in money and effort*.

---

## Hour 1 — Weaviate

### The problem we're solving

Some projects need more than a bare "store vectors, search vectors" box. They need structured data (each item has named fields), built-in hybrid search, and the ability to plug in embedding models and even text generation. **Weaviate** is an open-source vector database built around exactly that: structured objects, a defined schema, pluggable modules, and first-class hybrid search.

"Open-source" means the code is free and you can run it yourself (commonly via **Docker**, a tool that packages and runs software in isolated containers) or use Weaviate's hosted cloud service.

### Core concept: the schema

A **schema** is the definition of your data's structure — what kinds of objects you store and what fields (called **properties**) each one has. In Weaviate, data lives in a **collection** (a named group of objects of the same kind, like a table in a traditional database). When you create a collection you declare its properties and their data types (text, number, boolean, etc.).

Defining a schema up front is more work than just dumping vectors in, but it pays off: you get typed fields you can filter on, validation, and predictable structure.

> **Gotcha — schema mismatches break first runs.** A very common beginner failure is inserting an object whose fields don't match the declared schema (wrong property name, wrong type). Define the schema deliberately and make your inserted data match it exactly.

### Core concept: modules

A **module** is a pluggable component that adds capability to Weaviate. The two kinds you'll meet first:

- **Vectorizer modules** turn your text into vectors *automatically* at insert and query time, so you don't have to embed text yourself. Examples include modules that call OpenAI, Cohere, or a local model. The alternative is to embed text yourself and hand Weaviate the finished vectors.
- **Generative modules** let Weaviate send retrieved results to a large language model to generate an answer — useful for building question-answering directly on top of search.

This modularity is a defining trait: you assemble the embedding and generation pieces you want rather than wiring them together by hand.

### Core concept: hybrid search and GraphQL

Weaviate has **hybrid search** built in. It blends semantic (vector) search with **BM25** — a classic keyword-scoring method that ranks results by how well their literal words match the query. A weighting knob called **alpha** controls the mix: `alpha=1` is pure vector search, `alpha=0` is pure keyword search, and values in between blend them. The two result lists are merged using a **fusion** method (for example, a rank-based method that combines the two rankings).

**GraphQL** is a query language for APIs (a structured way to ask a service for exactly the data you want). Weaviate historically exposed its queries through GraphQL, and you can still drop down to raw GraphQL for advanced cases. The modern Python client wraps the common queries in Python methods so you usually don't write GraphQL by hand.

> **Important — the Python client was rewritten (v3 → v4).** The current client is **collection-based**: you get a collection object and call query methods on it. Older tutorials use a different, now-outdated style. The current client also talks to Weaviate over **gRPC** (a fast binary communication protocol) under the hood, which means a gRPC port must be reachable on your Weaviate server, not just the HTTP port.

### Putting it together in code

```python
import weaviate
import weaviate.classes.config as wc

# Connect to a local Weaviate (via Docker), or use connect_to_weaviate_cloud(...)
client = weaviate.connect_to_local()

try:
    # 1) Define a collection (schema). Here we supply our OWN vectors,
    #    so we set the vectorizer to None.
    client.collections.create(
        "Article",
        properties=[
            wc.Property(name="title", data_type=wc.DataType.TEXT),
            wc.Property(name="body",  data_type=wc.DataType.TEXT),
        ],
        vectorizer_config=None,   # we'll provide vectors ourselves
    )

    articles = client.collections.get("Article")

    # 2) Insert an object with a vector you computed elsewhere
    articles.data.insert(
        properties={"title": "Refund policy", "body": "Refunds take 5 business days."},
        vector=[0.10, 0.22, 0.31],  # your embedding here
    )

    # 3) Hybrid search: blend keyword (query text) with vector (your query embedding)
    res = articles.query.hybrid(
        query="how long for a refund",     # keyword side
        vector=[0.11, 0.20, 0.30],         # semantic side (your query embedding)
        alpha=0.5,                          # 0 = keyword only, 1 = vector only
        limit=3,
    )
    for o in res.objects:
        print(o.properties)
finally:
    client.close()   # always close the client
```

If instead you configure a vectorizer module, you can skip computing vectors yourself and pass only text — Weaviate embeds it for you on both insert and query.

> **Gotcha — the vectorizer config helper has moved across client versions.** The exact path for configuring a vectorizer module (for example, the namespace it lives under) has changed between minor versions of the v4 client. If a vectorizer config line errors, check the current Quickstart for the exact helper name for your installed version. The *concepts* (you either configure a vectorizer module or supply your own vectors) are stable.

> **Gotcha — always close the client.** The v4 client holds open connections. Forgetting `client.close()` (or using it as a context manager) can leak connections and cause confusing warnings.

### Hands-on exercise (Hour 1)

1. Run Weaviate locally with Docker (follow the Weaviate Quickstart), or sign up for a free Weaviate Cloud sandbox.
2. Embed 6–10 short sentences yourself with a model like `all-MiniLM-L6-v2` (384 dimensions).
3. Create an `Article` collection with `title` and `body` text properties and `vectorizer_config=None`. Insert your sentences with their vectors.
4. Run a `hybrid` query: pass a natural-language `query` string *and* the query's embedding as `vector`. Print the results.
5. Re-run the same query three times with `alpha=0`, `alpha=0.5`, and `alpha=1`. Write 2–3 sentences describing how the results shift from keyword-driven to meaning-driven.

### Interview questions — Hour 1

**1. (Conceptual) What is a schema in a vector database like Weaviate, and what does declaring one buy you?**
A schema defines the structure of your stored objects — the collection they belong to and the named, typed properties each object has. Declaring it gives you typed fields to filter on, input validation, and predictable structure, at the cost of a little setup work compared to dumping in unstructured vectors.

**2. (Conceptual) What is a module in Weaviate? Give two kinds.**
A module is a pluggable component that adds capability. Vectorizer modules automatically turn text into vectors at insert and query time so you don't embed manually; generative modules pass retrieved results to a language model to produce generated answers (e.g. for question-answering).

**3. (Practical) When would you let Weaviate vectorize your text for you versus supplying your own vectors?**
Let Weaviate vectorize when you want simplicity and are happy to depend on a configured embedding module — fewer moving parts. Supply your own vectors when you need precise control over the embedding model, want to reuse embeddings computed elsewhere, or must keep embedding off the database for cost, privacy, or consistency reasons.

**4. (Conceptual/Practical) What does the alpha parameter control in hybrid search, and how would you set it for a query full of exact codes?**
Alpha controls the blend between keyword (BM25) and vector search: 0 is pure keyword, 1 is pure vector. For queries dominated by exact codes or identifiers, you'd lower alpha toward the keyword end so literal matches are weighted more heavily.

**5. (Gotcha) A teammate's old Weaviate script throws errors after upgrading. What's your first hypothesis?**
The Python client was rewritten from v3 to the collection-based v4, so old-style code no longer works. They likely need to migrate to the v4 API. A close second: the v4 client needs a reachable gRPC port, so a connection that only exposes HTTP can also fail.

---

## Hour 2 — Chroma

### The problem we're solving

When you're prototyping — experimenting on your laptop, building a quick demo, learning — you don't want to run Docker containers or stand up a server. You want to `pip install` one thing and have vector search working in five lines. **Chroma** (the `chromadb` package) is built for exactly that: it's **local-first**, meaning it's designed to run right inside your Python process on your own machine, with no separate server required.

### Core concept: client modes

Chroma can run three ways, and picking the right one is the single most important Chroma decision:

- **In-memory client** (`chromadb.Client()` or `EphemeralClient()`): stores everything in RAM. Fast and zero-setup, but **all data is lost when your program exits**. Good for tests and throwaway experiments.
- **Persistent client** (`chromadb.PersistentClient(path="./chroma_db")`): stores data on disk (backed by SQLite, a lightweight file-based database) in a folder you choose. Data **survives restarts**. This is what you want for anything you care about keeping.
- **HTTP client** (`chromadb.HttpClient(host=..., port=...)`): connects to a Chroma server running as a separate process, for multi-process or production-style use.

**Persistence** here means data is written to durable storage so it outlives the program that created it.

> **Gotcha #1 — `Client()` silently loses your data on restart.** Beginners copy `chromadb.Client()` from a tutorial, add documents, restart, and find everything gone. For anything you want to keep, use `PersistentClient(path=...)`.

### Core concept: collections and automatic embedding

Like Weaviate, Chroma groups data into a **collection**. The friendliest detail: Chroma has a built-in **embedding function** (by default a local Sentence Transformers model), so you can add *raw text* and Chroma embeds it for you. You can also pass your own precomputed vectors via an `embeddings` argument, or configure a different embedding function (e.g. an OpenAI one).

Use `get_or_create_collection(...)` rather than `create_collection(...)`: it's **idempotent**, meaning it returns the existing collection if it already exists instead of raising an error — so re-running your script doesn't blow up.

```python
import chromadb
from chromadb.utils import embedding_functions

# Persistent: data is saved to ./chroma_db and survives restarts
client = chromadb.PersistentClient(path="./chroma_db")

# Explicitly choose the embedding function (do this on EVERY open of the collection)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection(name="docs", embedding_function=ef)

# Add raw text — Chroma embeds it for you using `ef`
col.add(
    documents=[
        "Refunds are issued within 5 business days.",
        "Standard shipping takes 5 to 7 business days.",
        "Premium members get free expedited shipping.",
    ],
    ids=["d1", "d2", "d3"],
    metadatas=[{"topic": "refunds"}, {"topic": "shipping"}, {"topic": "shipping"}],
)

# Query by meaning. `where` filters on metadata; `where_document` filters on text content.
res = col.query(
    query_texts=["how soon do I get my money back"],
    n_results=2,
    where={"topic": "refunds"},                 # metadata filter
    # where_document={"$contains": "business"},  # optional full-text filter
)

# Results come back as nested lists (one inner list per query text)
for doc, dist in zip(res["documents"][0], res["distances"][0]):
    print(round(dist, 3), doc)
```

Note the two filter types: `where` matches on **metadata** (the key-value extras you attached), while `where_document` matches on the **document text itself** (e.g. `{"$contains": "word"}`).

> **Gotcha #2 — the embedding function must match every time you open a collection.** If you create a collection with one embedding function but later reopen it with a different one (for instance, relying on a different default), every query embeds with the wrong model and returns nearest-*noise* — wrong results, with no error raised. Pass the same `embedding_function` explicitly each time you get the collection.

> **Gotcha #3 — a collection's dimension is fixed on first write.** Once you add vectors of a certain length, you can't add differently sized vectors (e.g. switching embedding models from 384 to 3072 dimensions). This is intentional: mixing embedding spaces produces garbage similarity. Switching models means re-indexing into a fresh collection.

> **Gotcha #4 — don't call `get()` without a `limit` on large collections.** A bare `collection.get()` loads every document, embedding, and metadata into memory at once, which can crash on large datasets. Paginate with `limit` and `offset`.

### Where Chroma fits

Chroma shines for prototyping, development, learning, demos, and small-to-moderate datasets, especially as the vector store inside a question-answering pipeline you're building locally. Its trade-off is the flip side of its simplicity: it's not aimed at very large scale or heavy concurrent production traffic the way a dedicated managed service or a hardened server is. You start with it because it gets you moving in minutes.

### Hands-on exercise (Hour 2)

1. `pip install chromadb sentence-transformers`. Create a **`PersistentClient`** pointing at `./chroma_db`.
2. Using `get_or_create_collection` with an explicit Sentence Transformers embedding function, add 8–10 short documents with a `topic` metadata field.
3. Query by meaning and print results with distances.
4. Add a `where` filter on `topic` and confirm results narrow.
5. **Prove persistence and the embedding-function gotcha:** exit Python, restart, reopen the collection *with the same embedding function*, and query again — confirm your data is still there. Then (for learning) reopen it *without* specifying the embedding function and observe how results degrade.

### Interview questions — Hour 2

**1. (Conceptual) What does "local-first" mean, and why is it appealing for prototyping?**
Local-first means the database is designed to run inside your own process on your own machine without a separate server. It's appealing for prototyping because setup is trivial (one install), there's no infrastructure to manage, and you can iterate quickly before committing to anything heavier.

**2. (Practical) A colleague says "I add documents to Chroma but they're gone every time I restart." What happened and how do you fix it?**
They're using the in-memory client (`chromadb.Client()`/`EphemeralClient()`), which keeps data only in RAM and loses it on exit. Switch to `chromadb.PersistentClient(path="./chroma_db")` so data is written to disk and survives restarts.

**3. (Conceptual) What's the difference between Chroma's `where` and `where_document` filters?**
`where` filters on the metadata you attached to each item (key-value pairs), while `where_document` filters on the actual document text (for example, requiring it to contain a substring). One narrows by structured attributes, the other by textual content.

**4. (Gotcha) Queries return irrelevant results even though the data looks fine and no error is raised. What's a likely Chroma-specific cause?**
The embedding function differs between when the collection was created and when it's being queried — for example it was created with one model but reopened relying on a different default. The query then embeds with the wrong model and returns nearest-noise. Fix by passing the same embedding function explicitly every time you open the collection.

**5. (Practical) For what kind of project would you reach for Chroma, and when would you outgrow it?**
Reach for Chroma for local development, demos, learning, and small-to-moderate datasets where simplicity wins. You'd outgrow it when you need very large scale, heavy concurrent production traffic, or operational features that a dedicated managed service or hardened server provides.

---

## Hour 3 — pgvector

### The problem we're solving

What if you already run **PostgreSQL** — a popular, mature, open-source **relational database** (a database that stores data in tables of rows and columns and is queried with SQL)? Adding a brand-new, separate vector database means another system to operate, back up, secure, and keep in sync with your real data. **pgvector** avoids that: it's an **extension** (an add-on that plugs new capabilities into PostgreSQL) that teaches Postgres to store vectors and run similarity search. Your vectors live right alongside your normal data, in the database you already trust.

The appeal of **self-hosting** (running the database yourself rather than paying a vendor to run it) here is unification: one system for both your relational data and your vectors, with all of Postgres's transactions, joins, and backups applying to both.

### Core concept: the vector type and distance operators

After enabling the extension, Postgres gains a new `vector` data type. You declare a column's dimension, e.g. `vector(384)`, and all vectors in it must have that length.

pgvector adds **distance operators** you use in queries to measure how far apart two vectors are:

- `<=>` — **cosine distance** (smaller = more similar in meaning). This is the usual choice for text.
- `<->` — **L2 (Euclidean) distance** (straight-line distance).
- `<#>` — **negative inner product**.

> **Gotcha — cosine *distance* is not cosine *similarity*.** Cosine distance equals `1 − cosine similarity`, so **smaller means closer**. You sort ascending (`ORDER BY ... LIMIT k`) to get the most similar rows. Mixing up distance and similarity (sorting the wrong direction) is an easy way to get exactly-backwards results.

> **Gotcha — `<#>` returns the *negative* inner product.** pgvector negates it because Postgres index scans go in ascending order. If you use this operator, account for the sign or your ranking will be inverted.

### Core concept: exact vs approximate search, and indexes

By default, pgvector does **exact** nearest-neighbor search: it compares your query against every row. This gives **perfect recall** (it never misses the true closest vectors — **recall** is the fraction of the genuinely-nearest results you actually retrieve), but it gets slow as the table grows.

To go faster at scale you add an **index** that enables **approximate nearest neighbor (ANN)** search — it finds *almost* the closest vectors much faster, trading a little recall for a lot of speed. pgvector offers two index types:

- **HNSW** (Hierarchical Navigable Small World): a graph-based index with excellent query speed-versus-recall, but slower to build and more memory-hungry. It can be built on an empty table.
- **IVFFlat**: partitions vectors into lists; faster to build and lighter on memory, but generally lower query performance for the same recall. It needs some data present first to build well.

```sql
-- 1) Enable the extension (once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2) A table with a 384-dimensional vector column
CREATE TABLE documents (
    id        bigserial PRIMARY KEY,
    content   text,
    embedding vector(384)
);

-- 3) Insert rows (embedding generated by your model, written as a bracketed list)
INSERT INTO documents (content, embedding)
VALUES ('Refunds take 5 business days.', '[0.10, 0.22, 0.31]');

-- 4) Exact search: perfect recall, slower at scale. Smaller cosine distance = closer.
SELECT content, embedding <=> '[0.11, 0.20, 0.30]' AS distance
FROM documents
ORDER BY embedding <=> '[0.11, 0.20, 0.30]'
LIMIT 5;

-- 5) Add an HNSW index for COSINE distance to make queries fast (approximate)
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

> **Gotcha — the index's operator class must match the query operator.** `vector_cosine_ops` builds an index for cosine-distance (`<=>`) queries; `vector_l2_ops` is for L2 (`<->`); `vector_ip_ops` is for inner product (`<#>`). If you index with one operator class but query with a different operator, your query simply won't use the index and falls back to slow exact search — no error, just unexplained slowness.

From Python you typically use the `psycopg` driver with pgvector's adapter so Python lists convert to the `vector` type automatically:

```python
# pip install "psycopg[binary]" pgvector
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect("postgresql://user:password@localhost/mydb")
register_vector(conn)   # lets you pass Python lists/np arrays as vectors

conn.execute(
    "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
    ("Premium members get free shipping.", query_embedding),
)
rows = conn.execute(
    "SELECT content FROM documents ORDER BY embedding <=> %s LIMIT 5",
    (query_embedding,),
).fetchall()
```

### When self-hosting with pgvector makes sense

pgvector is a strong fit when you already run Postgres and want one system instead of two; when your scale is small-to-moderate; when you value being able to **join** vector results against your other relational tables and wrap everything in database **transactions** (all-or-nothing units of work); and when you want full control over where your data lives. It's less ideal when you need the absolute highest-end vector performance at very large scale or the turnkey operational features of a purpose-built managed vector service — at the extreme end, dedicated systems pull ahead.

> **Gotcha — adding an approximate index changes your results.** Unlike normal database indexes (which only speed things up), an ANN index can return slightly *different* rows than exact search, because it's approximate. This is expected; just be aware your results may shift the moment you add the index. Also note: zero or NULL vectors aren't indexed for cosine distance.

### Hands-on exercise (Hour 3)

1. Get a PostgreSQL with pgvector installed (a local Docker image with pgvector, or a hosted Postgres that offers the extension). Run `CREATE EXTENSION vector;`.
2. Create a `documents` table with a `content text` column and an `embedding vector(384)` column.
3. Embed 10–15 short sentences with `all-MiniLM-L6-v2` and insert them (use the Python `psycopg` + `register_vector` approach so you can pass lists directly).
4. Run an **exact** cosine-distance query (`ORDER BY embedding <=> query LIMIT 5`) and confirm sensible results.
5. Add an HNSW index with `vector_cosine_ops`, re-run the same query, and note whether the top results changed at all (they may, slightly — that's approximate search). Then write 2–3 sentences on why putting vectors in Postgres might be attractive for a team already running Postgres.

### Interview questions — Hour 3

**1. (Conceptual) What is pgvector, and what's the core appeal of using it instead of a separate vector database?**
pgvector is a PostgreSQL extension that adds a vector data type and similarity-search operators to Postgres. Its core appeal is unification: vectors live in the same database as your relational data, so you operate one system, and Postgres's transactions, joins, and backups apply to your vectors too — avoiding a second system to run and keep in sync.

**2. (Conceptual) Explain the difference between exact and approximate nearest-neighbor search, and the trade-off.**
Exact search compares the query against every vector, guaranteeing it finds the true closest ones (perfect recall) but slowing down as data grows. Approximate (ANN) search uses an index to find *almost* the closest vectors much faster, trading a small amount of recall for a large speed gain. You choose based on how much recall you can sacrifice for latency.

**3. (Practical) You're storing text embeddings and will query by meaning. Which distance operator and index operator class do you use, and which way do you sort?**
Use cosine distance (`<=>`) for the query and build the index with `vector_cosine_ops`. Sort ascending with `ORDER BY embedding <=> query LIMIT k`, because cosine distance is `1 − similarity`, so smaller values are more similar.

**4. (Gotcha) Your pgvector queries are unexpectedly slow even though you created an index. What do you check?**
Whether the index's operator class matches the operator in your query. An index built with `vector_cosine_ops` only accelerates `<=>` queries; query with a different operator (`<->` or `<#>`) and the planner ignores the index and falls back to slow exact search, with no error. Align the operator class with the operator you actually query with.

**5. (Conceptual) When would HNSW be preferable to IVFFlat, and vice versa?**
HNSW gives better query speed-versus-recall and can be built on an empty table, at the cost of slower builds and higher memory — preferable when query performance matters most. IVFFlat builds faster and uses less memory but generally gives lower query performance for the same recall, and needs data present to build well — preferable when build time and memory are the bigger constraints.

---

## Hour 4 — A Decision Framework

### The problem we're solving

"Which vector database should I use?" has no universal answer — it depends on your constraints. The goal of this hour is a repeatable way to decide, instead of picking whatever a benchmark blog praised this month. We'll compare the three databases above plus the major **managed** alternatives (a *managed* service is one a vendor runs for you, so you don't operate servers — the best-known being Pinecone, a hosted, auto-scaling vector database).

### The dimensions that actually matter

Evaluate any vector database along these axes:

- **Deployment model.** *Embedded/local* (runs in your process, e.g. Chroma), *self-hosted* (you run the server or extension, e.g. Weaviate or pgvector), or *fully managed* (a vendor runs it, e.g. Pinecone). This drives most other trade-offs.
- **Operational burden.** How much work is it to run, scale, monitor, and back up? Managed services minimize this; self-hosting maximizes your control *and* your responsibility.
- **Latency.** How fast are queries? This depends heavily on scale, your recall target, network distance, and whether an index is used — so treat any single latency number with suspicion.
- **Cost.** Self-hosted/open-source software is free to license but you pay for the infrastructure and the engineering time to run it. Managed services charge for usage but save you that operational effort. Costs change frequently — always verify current pricing.
- **Scale.** How many vectors and how much query traffic can it comfortably handle?
- **Features.** Hybrid search, metadata filtering, multi-tenancy (isolating many users/customers), multi-modal support (images, audio), built-in embedding/generation.
- **Ecosystem and integration.** Does it fit what you already run? (If you already operate Postgres, pgvector starts with a huge advantage.)

### Comparison table

Treat the latency and cost columns as *qualitative shapes*, not promises — your real numbers depend on your data, scale, configuration, and the provider's current pricing.

| Dimension | Chroma | pgvector (Postgres) | Weaviate | Pinecone (managed) |
|---|---|---|---|---|
| Deployment | Embedded / local-first (also has a server mode) | Extension inside your existing Postgres (self-hosted or hosted Postgres) | Self-hosted (Docker) or vendor cloud | Fully managed, serverless |
| Operational burden | Very low for local use | Low *if you already run Postgres*; otherwise you run Postgres | Medium — you run a server, or pay for cloud | Lowest — vendor runs everything |
| Setup speed | Fastest (one `pip install`) | Fast if Postgres exists | Medium (Docker/cloud + schema) | Fast (sign up, get key) |
| Hybrid search | Limited / evolving | Vector-only by default (combine with Postgres full-text yourself) | Built-in, first-class (alpha + fusion) | Supported (sparse-dense, `dotproduct` metric) |
| Metadata filtering | Yes (`where`) | Yes — full SQL `WHERE`, joins, transactions | Yes | Yes |
| Best-fit scale | Small–moderate | Small–moderate (large with tuning) | Moderate–large | Moderate–very large |
| Cost shape | Free (local) | Free license; pay for your Postgres infra | Free if self-hosted; pay for cloud | Pay per usage; verify pricing |
| Standout strength | Simplicity for prototyping | One system; SQL + joins + transactions on your vectors | Rich features, modules, hybrid, multi-modal | Zero-ops scaling, no infra to manage |
| Main trade-off | Not aimed at heavy production scale | Top-end vector performance trails dedicated systems | You operate it (or pay for cloud) | Vendor lock-in; cost grows with scale |

### A decision tree

Ask these in order and stop at the first "yes":

1. **Are you prototyping, learning, or building a local demo on a small dataset?** → **Chroma.** Minimal setup, fastest path to a working pipeline.
2. **Do you already run PostgreSQL, with small-to-moderate scale, and want one system (plus SQL joins/transactions on your vectors)?** → **pgvector.** Avoid operating a second database.
3. **Do you want a feature-rich open-source database — first-class hybrid search, pluggable embedding/generation modules, multi-modal — and you're willing to self-host or pay for its cloud?** → **Weaviate.**
4. **Do you want zero operational burden and easy scaling, and you're comfortable with a managed vendor and usage-based cost?** → **Pinecone** (or another managed service).

> **Gotcha — don't choose on benchmarks alone.** A latency or throughput number from someone else's benchmark was measured on their data, their hardware, their recall target, and their configuration. Yours differ. Benchmarks are a starting hint, not a decision.

> **Gotcha — switching later is expensive.** Migrating between vector databases generally means re-ingesting all your data and often re-embedding it, plus rewriting integration code. Choose with your *next year* in mind, not just today's prototype — but also don't over-engineer a demo into a production cluster.

> **Gotcha — "managed" trades effort for lock-in and cost.** A managed service removes operational work but ties you to a vendor's API and pricing, and costs typically grow with scale. Self-hosting is the inverse: more control and potentially lower cost, but you own the operations. Name the trade-off explicitly when you decide.

### Hands-on exercise (Hour 4)

Build your own decision artifact for a concrete project.

1. Write down a real (or realistic) project: roughly how many documents, expected query volume, whether you already run Postgres, your team's appetite for operations, and any privacy/data-residency constraints.
2. Recreate the comparison table above, but **add a final column "Fit for my project"** and fill each cell with a short note specific to your project.
3. Rank the seven dimensions (deployment, operational burden, latency, cost, scale, features, ecosystem) from most to least important *for your project*.
4. Walk the decision tree and pick one database. Write a 4–6 sentence justification that explicitly names what you're optimizing for and what trade-off you're accepting.
5. Sanity-check against the gotchas: are you choosing on a borrowed benchmark? Have you accounted for the cost of switching later?

### Interview questions — Hour 4

**1. (Conceptual) What are the main deployment models for a vector database, and why does deployment model drive so many other trade-offs?**
The main models are embedded/local (runs in your process), self-hosted (you run the server or extension), and fully managed (a vendor runs it). Deployment model drives operational burden, cost shape, and scaling: managed minimizes effort but adds vendor dependence and usage cost; self-hosting maximizes control and potential cost savings but makes you responsible for running it; embedded is simplest but aimed at smaller scale.

**2. (Practical) A team already runs PostgreSQL for their app and needs to add semantic search over a few hundred thousand documents. What would you recommend and why?**
pgvector is a natural fit: they add vector search to the database they already operate instead of standing up a second system, they get SQL filtering, joins, and transactions over their vectors, and a few hundred thousand documents is well within range — especially with an HNSW index. It minimizes new operational burden.

**3. (Practical) When would you recommend a fully managed service like Pinecone over self-hosting?**
When the team wants minimal operational burden and easy scaling, lacks the appetite or staff to run and tune a database, and is comfortable with usage-based cost and depending on a vendor's API. Managed services trade money and some lock-in for removing the work of operating and scaling the system.

**4. (Gotcha) A stakeholder insists on database X because a blog benchmark showed it was "fastest." How do you respond?**
I'd note that benchmark latency depends on the data, hardware, recall target, and configuration used in that test, none of which match ours, so it's a hint rather than a decision. I'd push to evaluate against our actual constraints — deployment model, operational burden, features we need, scale, cost, and fit with our existing stack — and, if it matters, to benchmark on our own data.

**5. (Conceptual) Why is the choice of vector database worth careful thought up front rather than something to "just change later"?**
Because switching later is costly: you typically re-ingest and often re-embed all your data and rewrite integration code. The deployment model and feature set also ripple through your architecture. So it's worth choosing with the next year's needs in mind — while still avoiding over-engineering a simple prototype.

---

## Gotchas Summary Table

| # | Gotcha | Where | Why it bites | How to avoid it |
|---|--------|-------|--------------|-----------------|
| 1 | Inserting data that doesn't match the schema | Weaviate | Type/field mismatch fails or misbehaves | Define the schema deliberately; make inserts match exactly |
| 2 | Old (v3) client code after upgrade | Weaviate | Client was rewritten to collection-based v4 | Migrate to the v4 API; ensure a gRPC port is reachable |
| 3 | Vectorizer config helper moved between versions | Weaviate | Config path changed across v4 minors | Check the current Quickstart for the exact helper name |
| 4 | Forgetting to close the client | Weaviate | Leaks open connections | Call `client.close()` or use a context manager |
| 5 | Using `Client()` and losing data on restart | Chroma | In-memory client keeps data only in RAM | Use `PersistentClient(path=...)` for anything to keep |
| 6 | Mismatched embedding function on reopen | Chroma | Query embeds with wrong model → nearest-noise, no error | Pass the same embedding function every time you open the collection |
| 7 | Changing embedding dimension after first write | Chroma / all | A collection's dimension is fixed; mixing spaces is garbage | Re-index into a fresh collection when changing models |
| 8 | `get()` without a `limit` | Chroma | Loads everything into memory; can crash at scale | Always paginate with `limit`/`offset` |
| 9 | Confusing cosine *distance* with *similarity* | pgvector | Distance = 1 − similarity; smaller is closer | Sort ascending (`ORDER BY ... LIMIT k`) |
| 10 | `<#>` returns negative inner product | pgvector | Sign is flipped for ascending index scans | Account for the negation in ranking |
| 11 | Index operator class ≠ query operator | pgvector | Planner ignores the index → silent slow exact search | Match `vector_cosine_ops`/`_l2_ops`/`_ip_ops` to your operator |
| 12 | Expecting an ANN index to be exact | pgvector | Approximate indexes can return slightly different rows | Expect a small result shift; that's normal for ANN |
| 13 | Choosing on someone else's benchmark | Decision | Their data/hardware/recall/config aren't yours | Evaluate against your own constraints; benchmark your data |
| 14 | Underestimating migration cost | Decision | Switching means re-ingest, often re-embed, rewrite code | Choose for ~next year; don't over-build a prototype either |

---

## Quick Reference Card

**The three philosophies**
- **Chroma** → local-first simplicity. One `pip install`, runs in your process.
- **pgvector** → vectors inside the Postgres you already run. One system, SQL + joins + transactions.
- **Weaviate** → feature-rich open-source. Schema, modules (vectorizers + generative), first-class hybrid search.
- *(plus)* **Pinecone** → fully managed, zero-ops, scales for you.

**Weaviate (v4 client)**
```python
import weaviate, weaviate.classes.config as wc
client = weaviate.connect_to_local()
try:
    client.collections.create("Article",
        properties=[wc.Property(name="title", data_type=wc.DataType.TEXT)],
        vectorizer_config=None)              # or configure a vectorizer module
    col = client.collections.get("Article")
    col.data.insert(properties={"title": "..."}, vector=[...])
    res = col.query.hybrid(query="text", vector=[...], alpha=0.5, limit=3)
    for o in res.objects: print(o.properties)
finally:
    client.close()
```
- alpha: 0 = keyword (BM25), 1 = vector. Talks gRPC under the hood.

**Chroma**
```python
import chromadb
from chromadb.utils import embedding_functions
client = chromadb.PersistentClient(path="./chroma_db")   # persists to disk
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection("docs", embedding_function=ef)  # idempotent
col.add(documents=[...], ids=[...], metadatas=[...])     # auto-embeds text
res = col.query(query_texts=["..."], n_results=2, where={"k": "v"})
res["documents"][0], res["distances"][0]
```
- Client modes: `Client()`/`EphemeralClient()` = memory (lost on exit); `PersistentClient(path=)` = disk; `HttpClient(host=,port=)` = server.
- `where` = metadata filter; `where_document={"$contains": "..."}` = text filter.

**pgvector**
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (id bigserial PRIMARY KEY, content text, embedding vector(384));
-- query (smaller cosine distance = closer):
SELECT content FROM documents ORDER BY embedding <=> '[...]' LIMIT 5;
-- speed it up:
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```
- Operators: `<=>` cosine distance · `<->` L2 · `<#>` negative inner product.
- Match index op class to query operator: `vector_cosine_ops` ↔ `<=>`.
- Default = exact (perfect recall, slow at scale); index = approximate (fast, slight recall trade-off).

**Decision tree**
1. Prototype / local / small → **Chroma**
2. Already on Postgres + moderate scale + want one system → **pgvector**
3. Want rich open-source features (hybrid, modules, multi-modal) → **Weaviate**
4. Want zero-ops managed scaling → **Pinecone** (or similar)

**Evaluate on:** deployment model · operational burden · latency · cost · scale · features · ecosystem fit.
**Watch out for:** borrowed benchmarks, migration cost, managed-vs-self-host trade-offs (lock-in/cost vs control/effort).

**Reference docs:** Weaviate Quickstart; Chroma documentation; pgvector GitHub; independent vector database comparison write-ups (verify pricing and benchmarks against your own workload).

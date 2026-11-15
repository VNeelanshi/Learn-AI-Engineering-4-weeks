# MCP & Agent Memory

The four parts: the **MCP protocol** (what it is, why it matters, and how it is structured), **building an MCP server**, **short-term memory**, and **long-term memory**.

> **Note on version names and churn.** Model and package names in the examples (`claude-sonnet-4-6`, `gpt-5.5`) are current examples and change often. This area moves fast, so where a common older approach is now deprecated (meaning it still works but is on its way out and should be avoided in new code), the tutorial says so and teaches the current one.

---

## Part 1. The MCP protocol: what it is, why it matters, and how it is structured

### Why this exists (the problem)

Before MCP, connecting an AI application to the outside world was what people call an **M×N problem**. Suppose you have *M* different AI applications and *N* different tools or data sources you want them to use (GitHub, Slack, a database, the local filesystem, and so on). If every application needs its own custom code to talk to every tool, you end up building and maintaining M times N separate integrations. Worse, each of those integrations tended to invent its own way of describing what it could do, its own way of logging in, and its own way of returning data, so nothing was reusable and the model had no reliable way to *discover* what was even available.

The **Model Context Protocol (MCP)** is an open standard (a shared, published specification that anyone can implement) introduced by Anthropic in late 2024 and now supported broadly across the industry. It turns that M×N problem into an **M+N problem**. Tool builders write *N* MCP **servers** (one per system they want to expose), AI-application builders write *M* MCP **clients** (one per application), and because everyone follows the same protocol, any client can talk to any server. You build a server once, and every MCP-capable application can use it. A helpful way to picture it: MCP is like a universal plug. Once a device and a socket agree on the same plug shape, any device can connect to any socket. MCP is that agreed-upon shape between AI applications and the tools and data they need.

### The client-server architecture

MCP defines three roles:

- **Host:** the AI application the person actually interacts with, such as a chat app, a coding tool, or a custom agent. The host is in charge overall and manages one or more clients.
- **Client:** a piece of software that lives *inside* the host and holds a single, dedicated connection to one server. If a host talks to three servers, it runs three clients, one per server.
- **Server:** a program that offers up capabilities (tools, data, and templates) to a client. A server can run **locally** (for example, a filesystem server on your own machine) or **remotely** (a service hosted somewhere on the internet).

An important property is **isolation**, which means each client-to-server connection is kept separate. A server cannot reach through the client to touch another server, and the host decides which servers are even allowed to connect. That separation is a deliberate security feature, not an accident.

### Two layers: data and transport

MCP is described in two layers that are worth keeping separate in your head:

- **Data layer:** the actual protocol, built on **JSON-RPC 2.0**. JSON-RPC is a simple, widely-used convention for packaging a "call this method with these arguments, and send back a result" message as JSON (the common text format for structured data). This layer defines the lifecycle of a connection and the core building blocks described below. In the current stable version of the spec, MCP is a *stateful* protocol, meaning a session is set up when a connection opens and kept alive for as long as the connection lasts. (The newest spec revision introduces a *stateless* core, which is easier to scale across ordinary web infrastructure. That is an advanced detail you do not need in order to start.)
- **Transport layer:** how those JSON-RPC messages physically travel between client and server, plus how they authenticate. The same messages work identically no matter which transport carries them.

### The three primitives (what a server exposes)

Everything an MCP server offers falls into one of three types. Each type comes with standard "list" and "get" or "call" methods, so a client can automatically discover what is available without any custom code:

- **Tools** (model-controlled): callable functions the LLM can invoke to *take an action or fetch changing data*, such as `search_web(query)`, `create_issue(title, body)`, or `get_customer(id)`. Each tool has a name, a description, and a schema for its inputs (see the definition of JSON Schema in Part 2). In most real agents, tools are the bulk of what MCP is used for.
- **Resources** (application-controlled): read-only data identified by a URI. A URI is a unique address for a piece of data, in the same spirit as a web page's URL, for example `file:///notes.txt`. A resource is fetched when it is needed, much like reading from a fixed address. Examples include a file's contents, a database record, or a configuration document. Resources let a server hand the model data to *reason about* without the model having to make a live function call.
- **Prompts** (user-controlled): reusable, fill-in-the-blank prompt templates the server offers, such as a "summarize this ticket using these fields" template. The client fills in the blanks before sending the finished prompt to the model.

There are also a few extra capabilities worth knowing by name, because you will see them in documentation and older tutorials.

Two of them are current:

- **Notifications:** one-way messages that announce that something changed, for example "the list of available tools just changed." They keep the client's picture of the server up to date without the client having to keep asking.
- **Roots:** a way for the client to tell a server which files or directories it is allowed to work within. Think of it as the client drawing a boundary on a map and saying "you may operate inside here, and nowhere else."

Two others are now deprecated - sampling and elicitation.

### The two transports

- **stdio** (short for standard input/output): the host starts the server as a local **subprocess**, which just means a separate program that the host launches and controls, and the two exchange messages over that program's normal input and output streams. There is no network involved, latency is roughly a millisecond, and it is the natural choice for local tools such as file access, a local database, or shell commands. Any secrets the server needs are passed in through environment variables (configuration values handed to a program when it starts).
- **Streamable HTTP:** the server runs as a web service, and the client sends JSON-RPC messages to it over HTTP POST requests. It can optionally stream responses back using Server-Sent Events, a web standard that lets a server push a series of updates to a client over one long-lived connection. This transport is for remote servers, for serving many clients at once, and for cloud deployment. It uses standard web authentication, with OAuth recommended (OAuth is a widely-used standard for granting an application limited access to an account without handing over a password), and it requires HTTPS (encrypted connections). This transport replaced the older HTTP+SSE design, which is now deprecated.

Because the message format is identical on both transports, **a server written for stdio can move to Streamable HTTP without changing its core logic.** Only the single line that selects the transport changes.

### The end-to-end flow

Putting it together, a full interaction runs in five steps:

1. **Initialize:** the client and server perform a handshake and tell each other what they support.
2. **Discovery:** the client asks what is available (for example, "list your tools"), and the server responds with names, descriptions, and input schemas.
3. **Exposure:** the host converts those tools into the format its LLM expects for function calling.
4. **Invocation:** the LLM decides to call a tool, and the host routes that request to the correct server.
5. **Execution and response:** the server runs the real logic, returns the result, and the host feeds that result back into the model, which then continues.

> **Gotcha: every connected tool consumes context on every request.** When a host is connected to several servers, the definitions of *all* their tools get loaded into the model's context window on each request. Three servers with fifty tools each means one hundred and fifty tool descriptions taking up space before the person's question is even read. That costs tokens (the chunks of text a model processes, which you pay for) and can dilute the model's focus. Keep each server narrowly scoped, and connect only the servers a given agent actually needs.

> **Gotcha: MCP inherits the usual security risks, and tools make them sharper because tools take real actions.** Validate tool inputs *on the server side* rather than trusting the model to produce well-formed, safe arguments. Treat third-party servers as untrusted and run them in a sandbox (an isolated environment such as a container). Guard against prompt injection, which is an attack where malicious text hidden inside data the model reads tricks it into doing something harmful, like calling a tool it should not. Log every tool call so you have an audit trail. For remote servers, use OAuth and HTTPS with a strict list of allowed origins. And remember that a stdio server runs with the same privileges as the process that launched it, so be careful what you launch.

> **Gotcha: MCP is not the same thing as function calling, nor as agent-to-agent protocols.** Plain function calling is the mechanism a single application uses to let its model call tools. MCP is the reusable protocol layer *on top* of that, which lets any host share any server's tools. Separately, MCP handles the *vertical* connection from an agent down to its tools and data, whereas Agent-to-Agent (A2A) protocols handle the *horizontal* connection between one agent and another. They solve different problems and are often used together.

### Hands-on exercise (Part 1)

No code here, just a modeling exercise to lock in the architecture.

1. Pick a real integration you can imagine, for example "let my assistant read my GitHub issues." Identify the **host**, the **client**, and the **server** in that setup.
2. For each capability you would expose, decide whether it is a **tool**, a **resource**, or a **prompt**. For instance, "open a pull request" is a tool, "the contents of README.md" is a resource, and "a template for writing release notes" is a prompt.
3. Choose a **transport**: would this server run locally (stdio) or remotely (Streamable HTTP)? Say why.
4. **Stretch:** estimate the context cost. If you connected this server (say twenty tools) plus two others (thirty and fifty tools), how many tool descriptions load on every request? Notice how that number argues for keeping servers narrow.

---

## Part 2. Build a small MCP server exposing a tool

### Why this exists (the problem)

If you wrap one of your own functions as an MCP server, then any MCP-capable host, whether a chat app, a coding tool, or a custom agent, can use it with no special glue code. You write the capability once, correctly, and stop hand-building a new adapter for every application that wants it.

### The Python SDK: FastMCP

The official MCP Python SDK (software development kit, meaning the official library for building MCP programs in Python) includes a high-level helper called **FastMCP** that hides the JSON-RPC plumbing. You install it and describe your tools as ordinary Python functions with type annotations, and the SDK generates the MCP-compatible schema for you.

```bash
pip install "mcp[cli]"      # the MCP SDK plus its command-line and developer tools
```

```python
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")          # give the server a name

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""     # this docstring becomes the tool's description
    # the type hints (city: str, returns str) let the SDK build the input schema for you
    return f"{city}: 18C, cloudy"

@mcp.resource("config://units")
def units() -> str:
    """The unit system this server reports in."""
    return "metric"

if __name__ == "__main__":
    mcp.run()                     # stdio transport by default
    # mcp.run(transport="streamable-http")   # switch to remote; core logic unchanged
```

Two things are doing important work here. First, the **type hints**, which are the `: str` and `-> str` annotations that tell Python (and the SDK) what type each value is. From them the SDK builds the tool's **JSON Schema**, which is a standard, machine-readable description of what an input looks like: which fields exist, their types, and which are required. Second, the **docstring**, which is the triple-quoted sentence just under the function definition. It becomes the tool's description. The model reads both of these to decide *when* to use the tool and *how* to call it, so they are not just documentation, they are the interface the model sees.

> **Gotcha: your type hints and docstrings are the interface.** If you leave off the type hints, the SDK cannot build a proper schema and the model may call the tool with the wrong shape of arguments. If you leave the docstring vague, the model will not understand when the tool applies and will either misuse it or ignore it. Write precise types and a clear, action-oriented description.

> **Gotcha: never print to standard output from a stdio server.** With the stdio transport, the JSON-RPC messages travel over standard output (the stream a program writes its normal output to). A stray `print(...)` injects junk into that stream and corrupts the protocol. Send any logging to standard error (a separate output stream meant for diagnostics) or to a file, and leave standard output for MCP alone.

> **Gotcha: validate inputs on the server side.** The arguments arrive from an LLM, which can invent values or produce the wrong shape. Check them inside your tool before doing anything real (a common way is with Pydantic, a popular Python library that validates data against a declared shape and gives a clear error when it does not match), and return a clear message rather than crashing. Never assume the arguments are safe or well-formed.

### Testing and connecting it

The SDK ships with the **MCP Inspector**, a tool for poking at your server (listing its tools and resources, and calling them) without having to wire up a full host first:

```bash
mcp dev server.py      # launches your server with the Inspector attached
```

To use the server from a real **host**, you register it in that host's configuration. For a local server, that usually means giving the command that launches it, such as `python server.py`. To drive it yourself from a **client**, you open a session over your chosen transport and go through the same initialize, discover, call flow from Part 1:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])

async def main():
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                       # handshake
            tools = await session.list_tools()               # discovery
            result = await session.call_tool("get_weather", {"city": "Paris"})
            print(result)
```

> **Gotcha: secure the remote path before you ship it.** Switching to Streamable HTTP is one line, but a remote server exposed with no authentication is an open door for anyone who can reach its address. Add OAuth or token-based authentication, require HTTPS, and check the request's origin against an allowed list before you expose anything beyond your own machine.

### Hands-on exercise (Part 2)

1. Write `server.py` with a `FastMCP` server named `"Weather"`, a `get_weather(city: str) -> str` tool (typed, with a clear docstring), and one resource (for example `config://units`).
2. Run `mcp dev server.py`, open the Inspector, list the tools, and call `get_weather` with a city. Confirm the schema and description came through from your type hints and docstring.
3. Add **input validation**: reject an empty or non-string city inside the tool and return a clear error instead of throwing an exception.
4. **Stretch:** add a second tool (say `get_forecast(city, days)`), switch `mcp.run()` to `transport="streamable-http"`, and notice that only that one line changed. If you have an MCP-capable host, register the local server and call the tool from it.

---

## Part 3. Short-term memory: conversation buffers and summarization

### Why this exists (the problem)

An LLM is **stateless** between calls, meaning it keeps no memory of your previous message once a call finishes. To hold a coherent conversation, you have to send the earlier turns back to it on every new call. That is what **short-term memory** is. It is also called **thread-scoped** memory, because it tracks one ongoing conversation (a "thread"). The complication is that you cannot simply resend *everything* forever. The context window is finite, so long histories cost more tokens, add latency (they take longer to process), and, even while they still fit, tend to degrade the model's performance, because models reason less reliably over very long stretches of text.

So short-term memory is really the craft of deciding **which slice of the conversation to keep in front of the model** on each turn.

### The classic concepts (timeless)

There are four standard strategies, and they are worth understanding no matter which framework you use:

- **Buffer:** keep the *entire* conversation, word for word, and resend all of it every turn. This is the simplest and most faithful approach, but the number of tokens grows steadily with every message, so you eventually hit the context limit (a hard error) and pay more and more along the way.
- **Window:** keep only the most recent *k* messages, sliding the window forward as the conversation grows. This bounds the token count and latency, at the cost of forgetting anything older than the window.
- **Summary:** replace the older history with a short, running summary written by the LLM, and update that summary as the conversation continues. It grows slowly, so it scales better for long chats, but producing the summary costs an extra LLM call and inevitably loses some detail.
- **Summary-buffer (hybrid):** keep the most recent messages word for word *and* a summary of everything older. This is the practical sweet spot: recent turns stay exact while older ones are compressed rather than thrown away.

### How to implement this today (important)

The current recommended approach is **LangGraph persistence**, the same mechanism used for durable state in general. LangGraph is a library for building LLM applications as a graph of steps with a shared state object flowing through them. Three of its pieces matter here:

- Store the conversation in the application's **state**, specifically in a `messages` list that *appends* new messages rather than replacing them.
- Attach a **checkpointer**, which is an object that automatically saves that state after each step. Use `InMemorySaver` (which keeps state in memory) during development, and `SqliteSaver` or `PostgresSaver` (which write to a database) for state that must survive restarts.
- Key everything by a **`thread_id`** in the configuration, so each conversation is its own separate, resumable thread.

That gives you buffer memory out of the box, since the full history persists per thread. You then keep its size under control with two mechanisms that implement the classic strategies above:

- **`trim_messages`** cuts the message list down to a token or message budget. This is the *window* strategy.
- **A summarization step** (a node in the graph, or LangChain's summarization middleware, which condenses the history as it approaches the token limit) implements the *summary* and *summary-buffer* strategies.

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[],
    checkpointer=InMemorySaver(),          # short-term memory: persists per thread
)

config = {"configurable": {"thread_id": "user-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Hi, I'm Bob."}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
# Because the same thread_id was used, the agent still has the earlier turn and answers "Bob."
```

To cap the history so it cannot overflow the context window:

```python
from langchain_core.messages import trim_messages

# Keep only the most recent messages that fit within a token budget (the "window" strategy)
recent = trim_messages(messages, max_tokens=1000, strategy="last", token_counter=model)
```

> **Gotcha: a full buffer will eventually overflow the context window.** For any long-running conversation, plain buffer memory keeps growing until it exceeds the model's limit and throws an unrecoverable error, and the quality drops noticeably well before that point. Always add trimming or summarization for anything beyond short chats.

> **Gotcha: summarization costs a call and loses detail.** A summary is a lossy compression produced by another LLM call. If you summarize too aggressively, you will drop the very fact the person expected you to remember, such as a flight number they mentioned earlier. Keep recent turns word for word (the hybrid approach) and be conservative about what you compress.

> **Gotcha: one `thread_id` per conversation.** The `thread_id` is what separates one chat's history from another's. Reuse it across different people or sessions and their conversations will bleed together.

### Hands-on exercise (Part 3)

1. Build a LangGraph chat agent with an `InMemorySaver` checkpointer. On one `thread_id`, run three turns: introduce a fact ("My name is Sam and I fly out Friday"), then ask about it, then ask a follow-up. Confirm it remembers within the thread.
2. Start a **second** `thread_id` and ask the same "what did I say?" question. Confirm it knows nothing, which shows that threads are isolated.
3. Add `trim_messages` with a small budget and watch older turns drop out. That is the window strategy in action.
4. **Stretch:** add a summarization step that condenses the oldest turns into a short summary while keeping the last couple word for word. You have now built summary-buffer memory. Test that a fact from an old, now-summarized turn is still recalled, at least roughly.

---

## Part 4. Long-term memory: vector-store, episodic, and semantic

### Why this exists (the problem)

Short-term memory dies when the session ends. But a useful assistant should remember you *next week*: your name, your preferences, the issue you reported last month, across different conversations. That is **long-term memory**: information kept **across sessions and threads**, stored under a label and recalled on demand, rather than tied to a single conversation.

### The types of memory (a human analogy)

The standard way to organize memory borrows from how people remember:

- **Semantic memory:** timeless *facts*. Examples: "the person's name is Bob," "they prefer metric units," "their company is headquartered in Amsterdam." (Short-term or working memory, meaning the current context window, was the subject of Part 3.)
- **Episodic memory:** specific *events or experiences*, usually attached to a time. Example: "on Tuesday the person said checkout was broken, and on Wednesday support marked it resolved." Being aware of time is what makes episodic memory distinct from plain facts.
- **Procedural memory:** *rules and skills*, meaning how the agent should behave. This is often the agent's own set of instructions that it refines over time.

Stripped of any particular product's branding, every memory system answers four questions in order: **What do I keep? When do I write it down? Where does it live? How do I find it again and rank what I find?**

The "when to write" question splits into two common styles. The **hot path** style means the agent decides to save something *during* the conversation, before it replies. That is immediate but adds a little latency to each turn. The **background** or **consolidation** style means a separate process extracts and compresses memories *after the fact*. That is cheaper per turn, but the memory is not available immediately.

### Vector-store memory (the dominant mechanism)

The most common way to store and recall long-term memories uses **embeddings**. An embedding is what you get when you run a piece of text through a model that turns it into a **vector**, which is simply a list of numbers that captures the text's meaning. The useful property is that texts with similar meaning end up with vectors that are close together, and texts with unrelated meaning end up far apart. You store each memory as a vector in a **vector database** (examples include Pinecone, Weaviate, pgvector, and Chroma), which is a database built to hold vectors and quickly find the ones most similar to a given vector. When a new question comes in, you turn *it* into a vector too, and retrieve the memories whose vectors are closest, which is called a **semantic search** because it matches on meaning rather than exact words. This lets an agent recall the *relevant* pieces of the past without cramming its entire history into the context window. It is the same idea as retrieval-augmented generation (fetching relevant text and feeding it to the model), applied to memories. Fancier systems combine several retrieval signals at once (meaning-based similarity, exact keyword matching, and matching on named entities), and some add graph or time-based structure for relationships and changing facts.

### Two representative tools

**Mem0** is a **framework-agnostic memory service**, meaning it works with any agent framework, and it exposes a deliberately small interface: you call `add()` to store something and `search()` to retrieve. Behind the scenes it turns content into vectors for semantic retrieval, and on each `add()` it runs an LLM pass to pull out discrete facts and reconcile them against what it already knows (adding, updating, or deleting entries to avoid duplicates and contradictions). Memories are **scoped** by identifiers you pass in, such as `user_id`, `agent_id`, or a session id, so different people's memories stay separate. It plugs into any framework without changing your agent's logic.

```python
from mem0 import Memory

m = Memory()
m.add("I prefer window seats and vegetarian meals.", user_id="bob")   # store a fact for Bob

hits = m.search("What are Bob's travel preferences?", user_id="bob")   # semantic retrieval
# returns the stored preference even though the query words are different

m.search("preferences", user_id="alice")   # a DIFFERENT user_id sees NONE of Bob's memories
```

**Letta** (formerly known as MemGPT) takes an **agentic, operating-system-inspired** approach. It gives the agent a *hierarchy* of memory and lets the agent manage that hierarchy itself using tools:

- **Core memory:** a few editable blocks kept *inside* the prompt, similar to a computer's fast working memory (RAM). This holds the person's profile and the current essentials. The agent rewrites these blocks with tool calls.
- **Recall memory:** the full conversation history, kept out of the prompt on "disk" and searched only when needed.
- **Archival memory:** an external vector store of long-term facts, which the agent inserts into and searches through tool calls.

The agent moves information between these tiers with named tool calls (insert into archival, search recall, edit a core block), which produces the illusion of unlimited memory even though the context window is fixed. It is powerful and flexible, but the quality of recall depends on the agent making good decisions about when to call those tools, it spends tokens making those calls, and your agent runs inside Letta's own runtime, which is a form of lock-in (dependence on one platform's way of doing things).

For completeness, two more names you will hear: **LangMem** together with the **LangGraph Store** is the long-term memory option when you are already using LangGraph, offering semantic, episodic, and procedural memory. **Zep** is built on a time-aware knowledge graph where facts carry validity windows (a "true from this date until that date" range), which suits questions like "what was true at a particular point in time."

### The gotchas that bite in production

> **Gotcha: cross-user memory pollution, the scariest one.** If your scoping is wrong at the storage layer, one person's facts can surface in another person's session, which is both a privacy failure and a correctness failure. Scope memories by a stable per-user identifier (a namespace, meaning a label that groups one user's memories and keeps them separate from everyone else's), enforce that scoping in the database itself, and write a test that specifically stores a fact as user A and then confirms user B cannot retrieve it.

> **Gotcha: retrieval misses when the wording does not match.** A fact stored in one phrasing but asked about in another (close in meaning, but different words) can slip past a naive vector search. Reduce this by storing both a literal and a reworded version, by trying more than one phrasing of the query, and by using hybrid retrieval (meaning plus keywords plus entities) rather than similarity alone.

> **Gotcha: duplicates, conflicts, and forgetting.** Memory piles up. Without deduplication and conflict resolution, you will store both "the person lives in New York" and, later, "the person lives in London," and then retrieve the stale one, which poisons answers. You need logic to update and delete (Mem0's reconciliation step does this) or validity windows (Zep's approach), plus some way to forget memories that are no longer relevant.

> **Gotcha: memory edits as a substitute for thinking.** In agentic memory systems, an agent can learn that calling a "save to memory" tool *feels* productive and start doing it reflexively instead of actually answering the person. Watch for this, and keep memory-writing purposeful.

> **Gotcha: memory is not free.** Every write and every recall adds LLM calls, retrieval latency, and storage. Do not try to remember everything. Decide deliberately what is worth keeping, and consolidate over time.

### Hands-on exercise (Part 4)

1. Using Mem0 (or a tiny hand-built vector store), `add` three facts about a person under `user_id="bob"`: a name, a preference, and a past issue.
2. `search` with a *differently worded* query (for example "what does Bob like?") and confirm that semantic retrieval returns the relevant fact despite the different wording.
3. `search` the same thing under `user_id="alice"` and confirm that none of Bob's memories appear. You have just verified the scoping boundary, which is the test against pollution.
4. **Stretch:** add a *contradicting* fact ("Bob now prefers aisle seats") and work through (or implement) how you would resolve the conflict so a later search returns the current preference rather than both. Then label each stored item as semantic, episodic, or procedural.

---

## Questions

These go beyond the definitions in the tutorial. Each one drops you into a concrete situation and asks you to apply the ideas, which is closer to what a real interview probes for.

**1. (MCP design) You are building one internal MCP server that three applications will share: a coding assistant, a support bot, and an analytics dashboard. How do you decide what to expose as tools versus resources, and how do you keep the three apps from interfering with one another?**

Expose anything that takes an action or fetches changing data as a tool (open a ticket, run a query), and anything that is read-only reference data as a resource addressed by a URI (a config document, a record). To keep the apps from interfering, lean on MCP's isolation: each application runs its own client with its own dedicated connection, so one app cannot reach another's session through the server. Then scope each app to only the tools it needs rather than exposing the full set to all three, which also keeps each app's context small and its tool selection accurate.

**2. (MCP judgment) A teammate wants to connect eight MCP servers to a single agent "to be safe." What do you push back on, and what would you measure before agreeing?**

Every connected server loads all of its tool descriptions into the model's context on every request, so eight servers can mean hundreds of tool schemas competing for attention before the person's question is even read, which raises token cost and hurts tool selection. Push back on connecting servers the agent will rarely use. Before agreeing, measure the total number of tool schemas being loaded, the token overhead they add per request, and tool-selection accuracy with the full set versus a trimmed set. Connect only the servers that earn their place.

**3. (MCP migration) You inherit an agent that depends on MCP sampling. It works today. What do you do, and on what timeline?**

Sampling still functions, but it is deprecated in the newest spec revision, so treat it as end-of-life rather than something to build more on. Stop adding new sampling usage immediately, and plan to migrate the existing usage to calling the LLM provider directly (or to the replacement request-based mechanism) instead of having the server ask the host to run completions. There is no need to panic-rewrite, because deprecated features stay in the spec for a defined window of at least twelve months from the deprecating revision, so schedule the change deliberately and complete it before that window closes.

**4. (Server debugging) Your MCP tool works when you call the underlying function directly in Python, but through the agent the model keeps passing the wrong arguments. Where do you look first?**

At the tool's type hints and docstring, because those are what the model actually sees. If the type hints are missing or loose, the generated schema will not pin down the argument shape, and if the docstring is vague, the model will not know how or when to call the tool. Tighten the parameter types and rewrite the description to be specific and action-oriented, then confirm the resulting schema in the Inspector.

**5. (Server debugging) Your stdio server runs fine until a colleague adds some logging, after which the connection starts failing. What happened, and how do you fix it without losing the logs?**

The logging is almost certainly writing to standard output, which is the same stream the stdio transport uses to carry JSON-RPC messages, so the log lines are corrupting the protocol. Redirect the logging to standard error or to a file, and leave standard output for MCP traffic alone. The logs are preserved; they simply travel on a different stream.

**6. (Short-term memory design) A support chatbot needs to handle troubleshooting sessions that can run 45 minutes without forgetting the original problem or overflowing the context window. Design its memory.**

Use thread-scoped memory: keep the conversation in state with a checkpointer, keyed by a per-session thread id, so the session persists across turns. To stay inside the context window, do not keep the whole transcript word for word; use summary-buffer memory, meaning keep the most recent turns verbatim and maintain a running summary of everything older, so the original problem statement survives in the summary while recent detail stays exact. Trim or summarize as the token budget approaches rather than waiting for an overflow error.

**7. (Short-term memory translation) You find a helpful tutorial that builds conversation memory with `ConversationBufferMemory`. How do you adapt its approach to something you would actually ship today?**

Take the concept (buffer memory, meaning keep the running conversation) but not the class, because `ConversationBufferMemory` and its relatives are deprecated. Reimplement it with LangGraph persistence: store the messages in state, attach a checkpointer (a durable one such as SqliteSaver or PostgresSaver for anything real), and key by a thread id. Then add `trim_messages` or a summarization step to bound the size, which the old buffer class would not have handled well on its own.

**8. (Long-term memory scenario) In March a user tells your assistant they live in New York; in October they mention moving to London. A week later they ask "what's the weather at home?" What must a well-designed memory system have done for this to answer correctly?**

It must have treated the two statements as a conflict rather than storing both as equally valid facts, and updated or superseded the March fact when the October one arrived (reconciliation, or a validity window that marks the New York fact as no longer current). At query time it then retrieves the current home location, London, rather than the stale one. Without conflict resolution, the system would hold both and might surface the outdated fact.

**9. (Long-term memory choice) You are choosing between Mem0 and Letta for a customer-support assistant whose main job is remembering each customer's account details and past tickets. Which fits better, and why?**

Mem0 fits this cleanly: it is a framework-agnostic service with a simple `add` and `search` interface, it extracts and reconciles discrete facts, and it scopes by a per-customer identifier, which matches "store and recall this customer's details." Letta's agent-managed memory hierarchy (core, recall, and archival tiers that the agent pages through with tool calls) is more powerful and more flexible, but it adds token cost and ties you to its runtime, which is more machinery than a straightforward per-customer fact store needs. You would reach for Letta when you need the agent itself to manage a layered, evolving memory.

**10. (Long-term memory debugging) Your system stored "the user is allergic to peanuts," but when the user asks "can I eat this granola bar?" the allergy never surfaces. Why might retrieval miss, and what are two fixes?**

The stored fact and the query are close in meaning but share almost no words, so a plain similarity search can rank the allergy note too low to retrieve. Two fixes: store the fact in more than one phrasing (a literal version plus a reworded one) so more query shapes match it, and use hybrid retrieval that combines meaning-based similarity with keyword and entity matching rather than similarity alone. Rephrasing the query a few different ways before searching also helps.

---

## Quick reference card

**MCP: architecture**

```text
Host (AI app) -> manages -> Client (one per server) -> connects to -> Server (tools/resources/prompts; local or remote)
data layer: JSON-RPC 2.0 (lifecycle + primitives)     transport layer: stdio (local) or Streamable HTTP (remote)
primitives: Tools (model-controlled actions) | Resources (read-only data by URI) | Prompts (fill-in templates)
current extras: notifications (change alerts), roots (allowed files/dirs)
deprecated extras: sampling, elicitation  (do not build on these; replaced by request-based multi round-trip)
flow: initialize -> discover -> expose to LLM -> LLM calls -> host routes -> execute -> result
MCP = vertical (agent to tools) ; A2A = horizontal (agent to agent)
```

**MCP: build a server (Python, FastMCP)**

```text
pip install "mcp[cli]"
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("Name")
@mcp.tool()                # type hints build the schema; docstring becomes the description
def my_tool(x: str) -> str: ...
@mcp.resource("uri://...") # read-only data
mcp.run()                  # stdio (default), or mcp.run(transport="streamable-http")
test: mcp dev server.py    # MCP Inspector
rules: log to standard error, not standard output; validate inputs; secure remote (OAuth + HTTPS)
```

**Short-term memory (thread-scoped)**

```text
concepts: buffer (full) | window (last k) | summary (running summary) | summary-buffer (recent + summary)
deprecated: ConversationBufferMemory and relatives (now under langchain_classic)
current: LangGraph persistence, meaning messages in state + a checkpointer + a thread_id
    dev checkpointer: InMemorySaver ; durable: SqliteSaver / PostgresSaver
bound the size: trim_messages(..., strategy="last")  (window) ; a summarization step (summary)
always trim or summarize long chats, or you will overflow the context window
```

**Long-term memory (across sessions)**

```text
types: semantic (facts) | episodic (time-stamped events) | procedural (the agent's own rules)
four questions: what to keep? when to write (hot path vs background)? where does it live? how to retrieve and rank?
mechanism: embed text into a vector -> store in a vector database -> recall by similarity (plus keywords/entities)
Mem0: framework-agnostic; add() and search(); scoped by user_id / agent_id / session; extracts facts and dedupes
Letta (formerly MemGPT): agentic tiers the agent manages via tools: core (in prompt) / recall (history) / archival (vectors)
others: LangMem + LangGraph Store (when on LangGraph) ; Zep (time-aware graph with validity windows)
guardrails: scope hard per user (with a boundary test) ; hybrid retrieval ; deduplicate and resolve conflicts ; remember deliberately
```

**Reference material:** the Model Context Protocol documentation (architecture, primitives, transports) and the MCP Python SDK and FastMCP examples; LangChain's memory concepts and its migration notes from the classic memory classes to LangGraph persistence; and the Mem0 and Letta documentation for long-term memory patterns.

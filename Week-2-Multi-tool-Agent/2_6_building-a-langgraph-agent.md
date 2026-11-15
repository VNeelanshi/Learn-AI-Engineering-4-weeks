# Building a LangGraph Agent

Throughout, we build one concrete example — a **research assistant**: an agent that answers questions by searching the web, consulting the user's own documents, doing calculations, and writing results to files, always citing its sources. (Where useful, a "code reviewer" is mentioned as an alternative target.)

### What you need to know first (a compact recap)

- **ReAct (Reasoning + Acting):** the agent pattern where the model reasons, acts by calling a tool, observes the result, and repeats until it can answer.
- **Checkpointer:** an object attached at compile time that persists state per conversation (`thread_id`), giving the agent memory.

The six parts: **design**, **state/tools/prompts**, **the core loop**, **adding real tools**, **testing happy paths**, and **iterating**.

> **Note on version names.** Model and package names (`claude-sonnet-4-6`, `langchain-tavily`) are current examples and change often; check the docs for the exact strings of the day.

---

## Part 1 — Agent design: picking a useful agent

### Why this comes first (the problem)

The highest-leverage decision you'll make is *what agent to build*. Pick the wrong target and no amount of good engineering saves it. A good agent target has three properties: it's **valuable** (worth automating), it's **achievable** with an LLM-plus-tools loop and most overlooked is **verifiable** (you can tell whether it did a good job). If you can't tell good output from bad, you can't debug or improve it, so you'll never trust it.

There's also a prior question: does this even need an *agent*? An agent (an LLM dynamically directing itself) is the right tool only when you *can't* hardcode the path in advance but the LLM *can* still check its progress against reality. If the steps are fixed, a plain **workflow** (LLM calls on predefined paths) is simpler, cheaper, and more reliable. Reach for an agent when flexibility outweighs the extra latency, cost, and unpredictability.

When an agent *is* warranted, its design reduces to three decisions (a useful framing from Anthropic's agent guidance):

- **Environment** — the world the agent perceives and acts in (for our research assistant: the web, the user's documents, a working folder).
- **Tools** — the documented, hard-to-misuse interfaces it acts through. Spend more effort here than on clever prompting.
- **System prompt** — its goals, constraints, and stop rules.

### Choosing well

Good first-agent targets share a shape: **bounded scope**, **clear success criteria**, and **tools that provide ground truth** (results the model can react to). Our research assistant qualifies: the task (find and synthesize sourced answers) is valuable, needs live data an LLM can't produce alone, and is verifiable (are the claims accurate and cited?). A code reviewer is another good fit — it reads files, runs tests/linters (ground truth: do the tests pass?), and comments.

Before writing any code, pin down three things:

1. **Mission** in one sentence ("Answer the user's question with an accurate, well-sourced summary").
2. **Success criteria** — what "good" concretely means ("every factual claim has a source; the answer directly addresses the question; it says so when it can't find something").
3. **Tools it needs** — the 3-5 capabilities required (search, document retrieval, calculation, file writing).

> **Gotcha — over-scoping kills agents.** "An assistant that can do anything" has no boundary, so it has no reliable behavior and no way to be evaluated. Give the agent one clear job. You can always add scope later once the core is solid.

> **Gotcha — no success criteria means you can't iterate.** If you haven't defined what good output looks like, every later step (testing, debugging, improving) becomes guesswork. Write the criteria *before* you build, so you have something to measure against.

> **Gotcha — a task with no ground truth can't be trusted.** If there's no way to check the agent's work (no tests, no sources, no verifiable result), you can't tell success from a confident hallucination. Prefer tasks where the environment gives feedback (search results, test outcomes, file contents).

### Hands-on exercise (Part 1)

1. Choose your agent (the research assistant, a code reviewer, or your own idea).
2. Write its **mission** in exactly one sentence.
3. List **5 concrete success criteria** — observable things a good run does.
4. List the **3-5 tools** it needs, and for each, one line on the ground truth it provides.
5. **Sanity check:** confirm (a) it's verifiable (you can check its work), and (b) it truly needs an agent rather than a fixed workflow. If either fails, adjust the target.

### Questions (Part 1)

1. **Conceptual — "What makes a good target for an agent, versus a bad one?"**
   A good target is valuable, achievable with an LLM-plus-tools loop, and verifiable (you can tell good output from bad). Bad targets are vague/unbounded, or produce results you can't check, so you can't debug or trust them.

2. **Practical — "How do you decide whether a task needs an agent or a workflow?"**
   Use an agent only when you can't hardcode the path in advance but the LLM can still verify its progress against the environment (search results, test outcomes). If the steps are fixed, a workflow on predefined paths is simpler, cheaper, and more reliable — reserve agents for cases where flexibility outweighs the added latency, cost, and unpredictability.

3. **Conceptual — "What three design decisions define an agent?"**
   The environment (the world it perceives and acts in), the tools (the documented interfaces it acts through — where you spend the most effort), and the system prompt (its goals, constraints, and stop rules).

4. **Gotcha — "Why is 'an assistant that does everything' a poor first agent?"**
   Because unbounded scope means no reliable, testable behavior and no clear success criteria, so you can't evaluate or improve it. A single, well-defined job produces predictable behavior you can measure and grow from.

5. **Practical — "Why insist on success criteria before building?"**
   Because testing, debugging, and iteration all measure output against a definition of "good." Without criteria defined up front, every later decision is guesswork and you can't tell whether a change helped or hurt.

---

## Part 2 — Define state, tools, and prompts

### Why this exists (the problem)

Before writing the loop, you define the three things the loop operates on: the **state** (what data flows through the agent), the **tools** (what it can do), and the **system prompt** (how it behaves). Getting these right up front is what makes the loop in Part 3 small and the debugging in Part 6 tractable. Skipping this and diving into code produces an agent that "sort of works" but can't be reasoned about.

### State: what flows through

For a ReAct agent, the state's backbone is the conversation. LangGraph's prebuilt `MessagesState` is a `TypedDict` with a single field, `messages`, that uses the **`add_messages`** reducer so new messages **append** rather than overwrite. You extend it with any extra fields your agent tracks.

```python
from langgraph.graph import MessagesState

class ResearchState(MessagesState):     # inherits `messages: Annotated[list, add_messages]`
    sources: list[str]                  # accumulate cited sources across the run (add a reducer if needed)
```

> **Gotcha — a missing accumulating reducer wipes history.** If you hand-roll a state with a plain `messages: list` (no `add_messages`), each node that returns `messages` replaces the whole list and the agent forgets everything. Use `MessagesState` (or annotate the field with `add_messages`) so the conversation accumulates.

> **Gotcha — don't stuff big blobs into state.** State is passed around (and, with a checkpointer, snapshotted) constantly. Putting large documents or files directly in state bloats every step. Store big data externally and keep a reference (an ID or path) in state.

### Tools: designing the interface (implement later)

You *design* the tool list now and *implement* it in Part 4. Each tool will be a **typed, documented function**: the type hints become the tool's input schema and the docstring becomes its description — and the model reads both to decide *when* and *how* to call it. So the design work here is choosing a **small, focused set** with **clear names and descriptions**.

For the research assistant, the list is: `web_search` (current info), `search_my_docs` (the user's own documents), `run_python` (calculations/data), and `write_file` (save output). Four tools, each with a distinct job.

> **Gotcha — vague tool descriptions cause mis-selection.** If two tools sound similar ("search" and "search documents") without descriptions that say *when* to use each, the model will pick the wrong one or supply wrong arguments. Write descriptions that disambiguate: "Search the public web for current information" vs "Search the user's uploaded documents."

> **Gotcha — too many tools confuse the model and bloat context.** Every tool's schema loads into the model's context each turn, and a long, overlapping list makes selection harder. Keep the set minimal; add tools only when a real gap appears.

### Prompts: how it behaves

The **system prompt** sets the agent's goals, its constraints, how to use its tools, and — critically — its **stop rule** (when to stop calling tools and answer). A vague prompt produces an agent that rambles, misuses tools, or never stops.

```python
SYSTEM_PROMPT = """You are a research assistant. To answer a question:
- Use `web_search` for current, public information and `search_my_docs` for the user's own documents.
- Use `run_python` only for calculations or data manipulation, not for general reasoning.
- Cite a source for every factual claim.
- When you have enough to answer, give a concise, well-sourced answer and STOP calling tools.
- If a tool fails or returns nothing useful, say so briefly — do not invent an answer."""
```

> **Gotcha — no stop rule means loops or over-tooling.** A ReAct agent stops when it replies *without* a tool call. If the prompt never tells it when it has "enough," it may keep searching or bounce between tools. State the stop condition explicitly ("when you can answer, stop").

### Hands-on exercise (Part 2)

1. Define your **state**: start from `MessagesState` and add any extra fields your agent tracks (with a reducer if they accumulate).
2. Write your **tool list**: 3-5 tools, each with a name and a one-line description that disambiguates it from the others. (Don't implement them yet.)
3. Write a first **system prompt** with explicit goals, tool-usage guidance, and a clear stop rule.
4. **Stretch:** for each tool, draft the exact function signature (name, typed parameters, return type) and a one-sentence docstring — this is the contract the model will read in Part 4.

### Questions (Part 2)

1. **Conceptual — "What three things do you define before building an agent's loop, and why?"**
   The state (what data flows through), the tools (what the agent can do), and the system prompt (how it behaves). Defining them up front keeps the loop small and makes later debugging tractable, because each piece is deliberate rather than accidental.

2. **Practical — "How should an agent's state be structured for a ReAct agent?"**
   Start from `MessagesState`, which appends to a `messages` list via the `add_messages` reducer, and extend it with any extra fields the agent tracks. Keep large data out of state (store a reference instead), since state is passed around and snapshotted constantly.

3. **Gotcha — "Your agent keeps 'forgetting' the conversation. What's a likely state bug?"**
   The `messages` field lacks an accumulating reducer, so each update overwrites the whole history. Using `MessagesState` (or annotating the field with `add_messages`) makes new messages append and preserves the conversation.

4. **Conceptual — "What belongs in an agent's system prompt?"**
   Its goals, constraints, how and when to use each tool, and an explicit stop rule (when to stop calling tools and answer). This prevents rambling, tool misuse, and loops.

5. **Gotcha — "Why keep the tool set small and the descriptions specific?"**
   Because every tool's schema loads into the model's context each turn (a long list bloats context and hurts selection), and similar-sounding tools without disambiguating descriptions get chosen wrongly. A small set with clear, distinct descriptions improves both accuracy and cost.

---

## Part 3 — Build the core ReAct loop in LangGraph

### Why this exists (the problem)

Now assemble the engine: the loop that lets the model reason, call a tool, read the result, and repeat until it can answer. This is the **ReAct** pattern, and it's inherently cyclical — the agent doesn't know in advance how many tool calls it'll need — which is exactly why a graph (with a cycle) is the right structure.

### The fast path: `create_react_agent`

LangGraph ships a prebuilt that builds the entire loop for you. You pass the model, the tools (a placeholder for now — real ones come in Part 4), the system prompt, and a checkpointer for memory:

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_anthropic import ChatAnthropic

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    tools=[],                       # placeholder; real tools added in Part 4
    prompt=SYSTEM_PROMPT,           # the system prompt from Part 2
    checkpointer=InMemorySaver(),   # thread-scoped memory
)

config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Hello — what can you help me with?"}]},
    config,
)
print(result["messages"][-1].content)
```

### What it's doing underneath (and when to build it yourself)

Understanding the internals matters for when you later need custom control. The prebuilt is equivalent to this manual graph: an **agent node** that calls the model, a **tool node** that runs any requested tools, a **conditional edge** that routes to the tool node if the model asked for a tool (or to the end if it didn't), and an edge **back** from the tool node to the agent (the cycle).

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

model_with_tools = ChatAnthropic(model="claude-sonnet-4-6").bind_tools(tools)   # bind_tools is required!

def agent_node(state: MessagesState):
    return {"messages": [model_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))            # runs requested tools, appends results
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)   # -> "tools" if tool calls, else -> END
builder.add_edge("tools", "agent")                    # THE CYCLE: back to the model to read results
graph = builder.compile(checkpointer=InMemorySaver())
```

Use `create_react_agent` for a standard tool-calling agent; drop to the manual graph when you need custom branching, validation, multiple agents, or non-standard state.

> **Gotcha — (manual path) you must `bind_tools`.** If you call the plain model instead of `model_with_tools`, it doesn't know the tools exist and will never call them — it'll just guess. Binding advertises the tools' names and schemas to the model.

> **Gotcha — route tool results back to the agent, not to `END`.** If the tool node edges straight to `END`, the model never sees the tool's output — it asked for something and never got to use it. The `tools → agent` edge is what lets the model read the result and continue.

> **Gotcha — the model stops by *not* calling a tool.** There's no explicit "done" signal; the loop ends when the model replies with no tool calls. (`tools_condition` handles this routing for you.) And remember LangGraph caps steps with a recursion limit (default 25) — hitting it usually means a loop that isn't terminating, not a limit that's too low.

### Hands-on exercise (Part 3)

1. Build your agent's core loop with `create_react_agent`, passing your model, your system prompt from Part 2, a `tools=[]` placeholder, and an `InMemorySaver` checkpointer.
2. Invoke it with a simple greeting and confirm it runs and responds.
3. Ask a two-turn question on the same `thread_id` and confirm it remembers the first turn (the checkpointer at work).
4. **Stretch:** rebuild the *same* loop manually with `StateGraph` + `ToolNode` + `tools_condition` so you can see the agent node, tool node, and the cycle. Confirm it behaves identically.

### Questions (Part 3)

1. **Conceptual — "What is the ReAct loop and why does it need a cyclical graph?"**
   ReAct is reason → act (call a tool) → observe → repeat until the model can answer. Because the number of tool calls isn't known in advance, the workflow must loop, and looping requires a cycle in the graph — a straight-line pipeline can't express it.

2. **Practical — "Walk through the nodes and edges of a minimal ReAct agent."**
   An agent node calls the model; a tool node runs requested tools and appends results. Edges: `START → agent`; a conditional edge from the agent that goes to the tool node if it requested a tool, else to `END`; and `tools → agent` to close the loop so the model reads the results.

3. **Practical — "When would you use `create_react_agent` versus building the graph manually?"**
   Use `create_react_agent` for a standard tool-calling agent — it builds the same loop with far less code. Build manually when you need custom control: branching, validation, multiple agents, retry logic, or a non-standard state.

4. **Gotcha — "Your agent answers as if it never used its tools. Two likely bugs?"**
   Either you called the model without `bind_tools` (so it never knew the tools existed), or you routed the tool node to `END` instead of back to the agent (so the model requested a tool but never saw the result).

5. **Gotcha — "Your agent errors out with a recursion/step-limit error. What does that usually mean?"**
   That a loop isn't terminating — the agent keeps calling tools without reaching a final answer. The fix is usually the prompt or routing (make the stop condition and tool usage clear), not raising the recursion limit.

---

## Part 4 — Add 3-5 real tools (search, retrieval, code exec, file IO)

### Why this exists (the problem)

An agent is only as capable as its tools — the loop is just plumbing until the tools give it real hands. Now we implement the four tools designed in Part 2 and attach them. Two quick definitions used below: an **embedding** turns text into a vector (numbers capturing meaning) so similar meanings sit near each other; a **vector store** holds those vectors so you can retrieve the most similar ones to a query (the basis of document retrieval).

### The tools

**1. Web search — Tavily.** Tavily is a search API built *for AI agents*: it returns clean, parsed text and source URLs (not ad-cluttered HTML), which is ideal for grounding answers.

```python
from langchain_tavily import TavilySearch        # pip install langchain-tavily; set TAVILY_API_KEY
web_search = TavilySearch(max_results=5)
# (No API key handy? DuckDuckGoSearchRun from langchain_community works without one, with rougher results.)
```

**2. Document retrieval — wrap a retriever as a tool.** `create_retriever_tool` turns a vector-store retriever into a tool the agent can call to search the user's own documents.

```python
from langchain_core.tools import create_retriever_tool
retriever = vectorstore.as_retriever()            # your vector store, built from the user's docs
search_my_docs = create_retriever_tool(
    retriever,
    name="search_my_docs",
    description="Search the user's uploaded documents for relevant passages.",   # disambiguates from web_search
)
```

**3. Code execution — Python REPL (handle with care).** A code tool lets the agent compute things reliably (arithmetic, data wrangling) instead of guessing.

```python
from langchain_experimental.tools import PythonREPLTool
run_python = PythonREPLTool()
```

> **Gotcha — a Python REPL runs arbitrary code (remote code execution risk).** `PythonREPLTool` will execute whatever the model produces, including file-system access and network calls. Never expose it unsandboxed in production. Run it inside a sandbox (a hosted code-execution service or a locked-down container) and restrict what it can touch. For a local learning build, keep the inputs trusted and be aware of the risk.

**4. File IO — a directory-scoped toolkit.** File tools let the agent save results or read local files.

```python
from langchain_community.agent_toolkits import FileManagementToolkit
file_tools = FileManagementToolkit(
    root_dir="./workspace",                       # <-- sandbox: the agent can ONLY touch this folder
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
```

> **Gotcha — file tools without a `root_dir` expose your whole filesystem.** If you don't set `root_dir`, read/write/delete tools can operate anywhere the process can reach. Always scope them to a dedicated working directory, and prefer read-only tools unless writing is required.

**5. (Optional) A custom tool.** Writing your own is just a decorated, typed, documented function:

```python
from langchain_core.tools import tool

@tool
def word_count(text: str) -> int:
    """Count the number of words in a piece of text."""     # docstring -> description; hints -> schema
    return len(text.split())
```

### Attaching them

```python
tools = [web_search, search_my_docs, run_python, *file_tools, word_count]

agent = create_react_agent(
    ChatAnthropic(model="claude-sonnet-4-6"),
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)
```

> **Gotcha — validate tool inputs; don't trust the model's arguments.** The arguments come from an LLM and can be malformed or unsafe. Validate inside each tool (types, ranges, allowed values — Pydantic helps) and return a clear error rather than crashing or acting on bad input.

### Hands-on exercise (Part 4)

1. Implement (real or stubbed) your agent's tools: a web search, a document retriever tool, a code/calculation tool, and a file-write tool. Scope the file tool to a `./workspace` folder.
2. Attach them to your agent via `create_react_agent(..., tools=tools)`.
3. Run a query that forces *at least two different tools* (e.g. "Search for X, then save a two-line summary to notes.txt"). Confirm it uses the right tools.
4. **Stretch:** add one custom `@tool` of your own with a clear name and docstring, and confirm the model calls it when appropriate. Then deliberately give two tools near-identical descriptions and watch the model pick wrongly — a preview of Part 6.

### Questions (Part 4)

1. **Practical — "What tools would a research assistant agent need, and why each?"**
   Web search (current public info), document retrieval (the user's own files), code execution (reliable calculation/data work rather than guessed arithmetic), and file IO (saving results). Each provides a distinct capability or source of ground truth the LLM lacks on its own.

2. **Conceptual — "How do you turn a function into a tool, and what does the model read?"**
   Decorate a typed function with `@tool`; the type hints become the input schema and the docstring becomes the description. The model reads both to decide when to call the tool and what arguments to pass, so precise types and a clear docstring are essential.

3. **Gotcha — "What's the danger of giving an agent a Python execution tool, and how do you mitigate it?"**
   It executes arbitrary model-generated code (a remote-code-execution risk with file/network access). Mitigate by running it in a sandbox (hosted code-execution service or a locked-down container), restricting its permissions, and never exposing it unsandboxed in production.

4. **Gotcha — "Why must file-management tools be given a `root_dir`?"**
   Without one, read/write/delete operations can reach anywhere the process can, which is dangerous. Scoping to a dedicated directory sandboxes the agent's file access; using read-only tools where possible reduces risk further.

5. **Practical — "How do you help the model choose between a web-search tool and a document-search tool?"**
   Give each a distinct, disambiguating description of *when* to use it ("current public information" vs "the user's uploaded documents"), keep the overall tool set small, and reinforce the guidance in the system prompt. Clear descriptions are how the model routes to the right tool.

---

## Part 5 — Test happy paths (run 5-10 queries, debug with traces)

### Why this exists (the problem)

Before worrying about weird inputs, confirm the agent handles the **happy path** — the normal, intended queries — correctly. And to debug an agent you must see *inside* it: an agent run is a tree of LLM calls, tool calls, and decisions, and `print` statements can't show you why it chose what it chose. The tool for this is **LangSmith**: a framework-agnostic platform that records a full **trace** of every step and lets you build **evaluations**.

### Turning on tracing

For LangChain/LangGraph, tracing is automatic once you set environment variables — no code changes:

```bash
export LANGSMITH_TRACING=true          # (older projects use LANGCHAIN_TRACING_V2=true)
export LANGSMITH_API_KEY=your-key
export LANGSMITH_PROJECT=research-agent # optional: groups traces under a named project
# For non-US accounts, also set LANGSMITH_ENDPOINT to your region's URL.
```

Now run the agent and open the trace in the LangSmith UI. A trace is the full nested run tree: every LLM call (with its exact prompt and output), every tool invocation (inputs and results), every state transition, plus token counts, latency, and cost. With LangGraph, the trace tree mirrors your graph's execution. (For non-LangChain code, wrap functions with the `@traceable` decorator or the provider client wrapper to get the same traces.)

### Reading a trace: the three failure modes

When an agent gives a wrong answer, a trace lets you pinpoint *which* of three things broke — a distinction you'll use constantly:

- **Tool failure** — the tool returned bad results or nothing (e.g. search found nothing useful).
- **Reasoning/prompt failure** — the tool returned good data but the model ignored it, misread it, or hallucinated anyway.
- **Orchestration failure** — the model called the *wrong* tool, or didn't call a needed tool, or looped.

Without a trace you'd only see "wrong answer"; with one you see exactly where the chain deviated, so you fix the right thing (Part 6).

### The process

1. Write **5-10 representative happy-path queries** — the normal things users will ask ("What's the latest on X? Summarize and cite.").
2. Run each through the agent.
3. For each, open the trace and check **two things**: the final answer (right?) *and* the path (did it call sensible tools in a sensible order?).
4. Turn those queries into a small **dataset** in LangSmith — pairs of `{input, expected output}` — and attach **evaluators** (exact match, semantic similarity, or an "LLM-as-judge" that scores against a rubric). Running this dataset gives you a repeatable score you can re-check after every change.

> **Gotcha — a right answer via a lucky path will break later.** If you only check the final answer, you'll miss that the agent reached it by accident (wrong tool, ignored a failure that happened not to matter). Always check the *trace*, not just the output — the path is what generalizes.

> **Gotcha — without a repeatable test set, you can't tell if changes help.** Ad-hoc manual testing can't detect regressions. Capture your happy-path queries as a dataset with expected outputs so every future change is measured against the same bar.

> **Gotcha — traces can capture sensitive data.** Prompts, tool inputs, and outputs may contain PII or secrets. In non-local environments, redact or filter what gets logged, and be deliberate about what you send to the tracing backend.

### Hands-on exercise (Part 5)

1. Enable LangSmith tracing (set the env vars) and run one query — confirm the trace appears.
2. Write 5-10 happy-path queries for your agent and run them all.
3. For each, open the trace and classify any problem as tool / reasoning / orchestration failure (or "clean"). Note the *step* where things went wrong.
4. Build a 5-example dataset (inputs + expected outputs) and run an evaluation with a simple evaluator (semantic similarity or an LLM judge). Record the baseline score.
5. **Stretch:** find one query where the answer is right but the *path* is questionable (an unnecessary tool call, a mis-ordered step) — proof that checking traces catches what output-only testing misses.

### Questions (Part 5)

1. **Conceptual — "What is a 'happy path' and why test it first?"**
   The happy path is the set of normal, intended inputs the agent is built to handle. You test it first to confirm the core behavior works before investing in edge cases — if the agent can't handle the expected queries, edge-case handling is premature.

2. **Practical — "How do you enable LangSmith tracing for a LangGraph agent, and what does a trace show?"**
   Set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (optionally `LANGSMITH_PROJECT`); tracing is then automatic with no code changes. A trace shows the full nested run tree — every LLM call with its prompt/output, every tool call with inputs/results, state transitions, tokens, latency, and cost — mirroring the graph's execution.

3. **Conceptual — "An agent returns a wrong answer. What three failure modes does a trace help you distinguish?"**
   Tool failure (the tool returned bad or no results), reasoning/prompt failure (good data but the model ignored or misused it), and orchestration failure (wrong tool, missing tool call, or a loop). The trace shows which one occurred so you fix the right layer.

4. **Gotcha — "Your agent passes a test with a correct answer, but you're still worried. Why?"**
   Because it may have reached the right answer by a lucky or wrong path (wrong tool, ignored failure) that won't generalize. You should inspect the trace to confirm the *reasoning path* was sound, not just the final output.

5. **Practical — "Why convert your test queries into a LangSmith dataset with evaluators?"**
   To make evaluation repeatable and objective: a dataset of inputs with expected outputs, scored by evaluators (exact/semantic/LLM-judge), gives a baseline you can re-run after every change to measure improvement and catch regressions — which ad-hoc manual testing can't do.

---

## Part 6 — Iterate: refine prompts and routing from failures

### Why this exists (the problem)

The first version of any agent has failures — that's expected, not a setback. **Iteration** is the disciplined loop that turns a rough agent into a reliable one: use the traces and evals from Part 5 to find the *specific* failure, fix the *right layer*, and re-measure to confirm it helped without breaking anything else. Doing this haphazardly (changing many things, guessing at the cause) makes the agent worse in ways you can't track.

### Match the failure mode to the fix

The three failure modes from Part 5 each point to a different layer:

- **Reasoning/prompt failure** (ignores tool output, rambles, doesn't stop, wrong tone) → fix the **system prompt**. Clarify goals, tell it to ground answers in tool results, and add or sharpen the **stop rule**.
- **Orchestration failure** (wrong tool, skips a needed tool, loops) → fix **tool descriptions** so the choice is unambiguous, tighten the prompt's tool-usage guidance, and ensure a real termination path (the stop rule; a step cap if needed).
- **Tool failure** (bad or missing results) → fix the **tool itself**: add input validation, handle errors and return a clear message, use a better source, or add a fallback.

The skill is diagnosing which layer is actually broken — the trace tells you — and changing *that* one.

### The iteration loop

1. Pick one failure from your traces/evals.
2. **Diagnose** the layer (prompt / routing / tool) from the trace.
3. Change **one thing**.
4. **Re-run your evaluation dataset** and compare to the baseline (LangSmith lets you compare runs side by side and version prompts).
5. Keep the change if it helped without regressions; revert if not. Repeat.
6. **Grow the dataset:** every new real-world failure becomes a permanent test case, so the agent can't regress on it later.

> **Gotcha — change one thing at a time.** If you edit the prompt, swap a tool, and change routing all at once and the score moves, you can't tell what caused it. Make single, isolated changes and re-evaluate after each, so cause and effect stay clear.

> **Gotcha — fix the right layer.** "Improving the prompt" won't help if the *tool* was returning garbage, and adding tool validation won't fix a model that ignores good results. Use the trace to identify the real failure point before touching anything.

> **Gotcha — don't overfit the prompt to your handful of examples.** Tuning the prompt until your 5 test queries pass can make it brittle on everything else. Keep the dataset representative and growing, and watch for a fix that helps your examples but hurts overall behavior.

> **Gotcha — always regression-test.** A change that fixes query A can break query B. Re-running the full dataset (not just the query you were fixing) is what catches this — which is exactly why the repeatable dataset from Part 5 matters.

### Hands-on exercise (Part 6)

1. Take one failing (or questionable-path) case from Part 5. Open its trace and **diagnose** the layer: prompt, routing/tool-selection, or tool.
2. Make **one** targeted change to that layer (e.g. add a stop rule to the prompt, or sharpen a tool description, or add validation to a tool).
3. Re-run your **whole** evaluation dataset and compare to the baseline. Confirm the case is fixed *and* nothing else regressed.
4. **Add the failing case to your dataset** so it becomes a permanent regression test.
5. **Stretch:** find a case where your first instinct about the failing layer is wrong (you think it's the prompt; the trace shows the tool returned nothing). Fix the actual layer, and note how the trace changed your diagnosis.

### Questions (Part 6)

1. **Conceptual — "Describe the iteration loop for improving an agent."**
   Pick a failure, diagnose the responsible layer from the trace, change one thing, re-run the evaluation dataset and compare to baseline, keep the change only if it helps without regressions, and add the failure as a permanent test case. Repeat.

2. **Practical — "Map each failure mode to what you'd fix."**
   Reasoning/prompt failures → fix the system prompt (goals, grounding in tool results, stop rule). Orchestration failures → fix tool descriptions and prompt guidance, and ensure a termination path. Tool failures → fix the tool (validation, error handling, better source, fallback).

3. **Gotcha — "Why change only one thing per iteration?"**
   Because if you change several things and the evaluation score moves, you can't attribute the effect. Single, isolated changes with re-evaluation after each keep cause and effect clear, so you know what actually helped or hurt.

4. **Gotcha — "You keep rewriting the prompt but the agent's answers don't improve. What might you be missing?"**
   You may be fixing the wrong layer — the trace might show the *tool* is returning bad or empty results, which no prompt change can fix. Diagnose the real failure point from the trace before editing, and fix that layer.

5. **Gotcha — "You tuned the prompt until all five test cases passed. What's the risk?"**
   Overfitting: the prompt may now be brittle and perform worse on inputs outside your small set. Keep the evaluation dataset representative and growing, and always regression-test the full set so a narrow fix doesn't degrade overall behavior.

---

## Gotchas summary table

| # | Gotcha | Where it bites | Fix / rule of thumb |
|---|---|---|---|
| 1 | Over-scoping the agent | Design | Give it one clear, bounded job; expand later |
| 2 | No success criteria | Design | Define what "good" means before building |
| 3 | Task with no ground truth | Design | Pick tasks the environment can verify (sources, tests, files) |
| 4 | Missing accumulating reducer wipes history | State | Use `MessagesState` / `add_messages` |
| 5 | Large blobs in state | State | Store externally; keep a reference in state |
| 6 | Vague/overlapping tool descriptions | Tools/prompt | Disambiguate when to use each; keep the set small |
| 7 | No stop rule in the prompt | Prompt | State explicitly when to stop calling tools and answer |
| 8 | (Manual loop) forgetting `bind_tools` | Core loop | Bind tools so the model knows they exist |
| 9 | Routing tool results to `END` | Core loop | Edge the tool node back to the agent |
| 10 | Hitting the recursion/step limit | Core loop | It's a non-terminating loop — fix prompt/routing, not the limit |
| 11 | Unsandboxed code-execution tool | Tools | Sandbox it (hosted service/container); never raw in prod |
| 12 | File tools without `root_dir` | Tools | Scope to a working directory; prefer read-only |
| 13 | Trusting model-supplied tool arguments | Tools | Validate inside the tool; return clear errors |
| 14 | Judging only the final answer | Testing | Inspect the trace — the path is what generalizes |
| 15 | No repeatable test set | Testing | Build a dataset with expected outputs; re-run it |
| 16 | Sensitive data in traces | Testing | Redact/filter what's logged in non-local envs |
| 17 | Changing many things at once | Iteration | One change per iteration, then re-evaluate |
| 18 | Fixing the wrong layer | Iteration | Diagnose from the trace before editing |
| 19 | Overfitting the prompt to a few examples | Iteration | Keep the dataset representative and growing; regression-test |

---

## Quick reference card

**Design (before code)**

```text
mission (1 sentence) + success criteria (what "good" is) + tools needed (3-5)
agent only if: can't hardcode the path BUT can verify progress; else use a workflow
three decisions: environment | tools | system prompt (spend effort on tools)
```

**State / tools / prompt**

```text
state:  class S(MessagesState): ...extra fields   # messages append via add_messages
tools:  small, focused; typed function + docstring = schema + description (model reads both)
prompt: goals + constraints + tool-usage guidance + explicit STOP rule
```

**Core loop**

```text
fast:   create_react_agent(model, tools=[...], prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())
manual: START -> agent -> [tools_condition] -> tools -> agent (cycle) ... -> END
        agent node: model.bind_tools(tools).invoke(messages)   # bind_tools required
        tools node: ToolNode(tools)   ;   route tool results BACK to agent (not END)
run:    agent.invoke({"messages":[...]}, {"configurable":{"thread_id":"..."}})
```

**Add tools**

```text
search:    from langchain_tavily import TavilySearch; TavilySearch(max_results=5)   # or DuckDuckGoSearchRun (no key)
retrieval: create_retriever_tool(vectorstore.as_retriever(), "search_my_docs", "...")
code exec: PythonREPLTool()   # DANGER: sandbox in prod (container / hosted service)
file IO:   FileManagementToolkit(root_dir="./workspace", selected_tools=[...]).get_tools()   # always set root_dir
custom:    @tool def f(x: str) -> str: """clear description"""
validate tool inputs (Pydantic); don't trust model args
```

**Test (happy paths)**

```text
enable: export LANGSMITH_TRACING=true ; LANGSMITH_API_KEY=... ; LANGSMITH_PROJECT=...
trace = every LLM call + tool call + state change + tokens/latency/cost (auto for LangChain/LangGraph; @traceable otherwise)
run 5-10 happy-path queries; check ANSWER and PATH
3 failure modes: tool failure | reasoning/prompt failure | orchestration failure
make a dataset (input + expected) + evaluators (exact / semantic / LLM-judge) -> baseline score
```

**Iterate**

```text
loop: trace -> diagnose layer -> change ONE thing -> re-run dataset -> keep/revert -> repeat
fix by layer: reasoning->prompt(+stop rule) | orchestration->tool descriptions/routing | tool failure->the tool
grow the dataset from real failures; always regression-test the full set
```

**Reference material:** existing agent examples for inspiration; LangGraph design-pattern docs (state, prebuilt ReAct agent, `ToolNode`/`tools_condition`); the LangChain tool library (Tavily search, `create_retriever_tool`, `PythonREPLTool`, `FileManagementToolkit`, the `@tool` decorator); and LangSmith for tracing and dataset-driven evaluation.

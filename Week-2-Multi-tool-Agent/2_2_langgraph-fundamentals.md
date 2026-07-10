# LangGraph Fundamentals

A complete, from-scratch tutorial on **LangGraph**: an open-source Python library you install (`pip install langgraph`) that lets you wire an LLM (and your own code) into a **graph** of steps, with shared memory ("state") flowing between them. This tutorial targets **LangGraph 1.0** (the stable line, released late 2025), which requires **Python 3.10 or newer**.

The four parts:

1. **Core concepts** — state, nodes, edges, and why you'd want a *graph* instead of a straight line.
2. **A simple agent** — building a "reason → act → observe → repeat" loop with one tool.
3. **Conditional edges & cycles** — routing to different steps and looping until a condition is met.
4. **Persistence & checkpoints** — saving state so runs can resume, remember, and be debugged by "time travel."

> **Note on version names.** Model names in examples (`claude-sonnet-4-6`, `gpt-5.5`) are just current examples and change often. The LangGraph API shapes shown here are the stable part; check the docs for the exact model string of the day.

---

## Part 1 — Core concepts: state, nodes, edges (and why a graph)

### Why this exists (the problem)

A lot of useful LLM work takes more than one step. You might classify a message, then handle it differently depending on the category. You might call a tool, look at the result, and decide whether to call another. The hard part isn't running one step — it's expressing *how steps connect*, *what data passes between them*, and especially *decisions and repetition*: "if the answer isn't good enough, try again," or "keep using tools until you can answer."

Before graphs, the common shape for multi-step LLM code was a **chain** — a fixed, straight-line sequence of steps: do A, then B, then C, every time, in that order. Chains are perfect when the path never varies. But they cannot naturally express two things real applications constantly need:

- **Branching** — go one way or another depending on what happened.
- **Looping** — repeat a step (or group of steps) an unknown number of times until some condition holds.

A straight line has no way to say "loop back to step 2 until done." **LangGraph exists to make branching, looping, stateful multi-step LLM workflows manageable.** It models your app as a *graph*: a set of steps (nodes) connected by links (edges) that can fork and loop, all sharing one evolving piece of data (state). That's the entire mental model; the rest is detail.

Four terms:

- **Graph:** the whole workflow — steps and the connections between them.
- **Node:** one step. In LangGraph, a node is just a Python function.
- **Edge:** a connection saying "after this node, go to that node." Edges can be unconditional ("always go to B") or conditional ("decide at runtime where to go").
- **State:** a single shared data structure that every node can read, and that nodes update as the graph runs. Think of it as a shared whiteboard the whole workflow writes on.

### State: the shared whiteboard

You declare the *shape* of your state with a **`TypedDict`** — a standard Python way to describe a dictionary with named fields and their types (it's just a dict with a documented structure). Every node receives the current state and returns a **dict of updates** (not the whole state — only the fields it wants to change). LangGraph merges those updates into the state.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    user_input: str
    response: str

def greet(state: State) -> dict:
    name = state["user_input"]
    return {"response": f"Hello, {name}!"}   # only returns the field it changed

builder = StateGraph(State)     # a "builder" you configure, then compile
builder.add_node("greet", greet)
builder.add_edge(START, "greet")   # START is the entry sentinel
builder.add_edge("greet", END)     # END is the exit sentinel
graph = builder.compile()          # turn the builder into a runnable graph

print(graph.invoke({"user_input": "Alice", "response": ""}))
# -> {'user_input': 'Alice', 'response': 'Hello, Alice!'}
```

Notice `greet` returned only `response`; LangGraph folded that into the existing state, leaving `user_input` untouched. This "return a partial update, LangGraph merges it" model is the core of how nodes cooperate.

By default, merging means **overwrite**: if two nodes both write `response`, the later write wins (last-write-wins), replacing the earlier value. That's usually what you want for a single value. But sometimes you want to *accumulate* — e.g. keep a growing list of steps, or a full conversation history. For that you attach a **reducer**: a function `(current_value, new_value) -> merged_value` that says how to combine an old value with an update. You declare it by annotating the field:

```python
from typing import Annotated, TypedDict
import operator

class State(TypedDict):
    log: Annotated[list[str], operator.add]   # operator.add concatenates lists
    value: int                                # no reducer -> overwritten each time
```

With `operator.add` on `log`, a node returning `{"log": ["step_one"]}` *appends* to the list instead of replacing it. Every time a node returns that field, the reducer runs to combine old and new.

> **Gotcha — the #1 beginner bug: no reducer silently wipes your history.** If a field is a list you expect to grow (like chat messages) but you *didn't* give it a reducer, each node that returns that field **replaces the whole list**, throwing away everything before it. Symptom: your agent "forgets" earlier messages. Fix: attach an accumulating reducer. For messages specifically, LangGraph provides the purpose-built `add_messages` reducer (from `langgraph.graph.message`), which appends new messages and correctly handles updates/IDs. There's even a ready-made state class, `MessagesState`, that is just a `TypedDict` with `messages: Annotated[list, add_messages]` — use it for chat/agent graphs.

> **Gotcha — keep reducers pure.** A reducer must only combine its two inputs and return the result. Don't make API calls, write files, or use randomness inside it. LangGraph may call it more than you expect, and side effects there cause chaos.

### Nodes and edges

A **node** is added with `builder.add_node("name", fn)` (you can also pass just the function and let its name be inferred). The function takes the state and returns a partial update. An **edge** connects nodes: `builder.add_edge("a", "b")` means "after `a` finishes, run `b`." Two special sentinels, `START` and `END` (imported from `langgraph.graph`), mark where execution begins and where it stops.

> **Gotcha — you must reach `END`.** Every path through the graph needs a way to terminate. If a node has no outgoing edge, LangGraph detects the dead end and raises an error. Always connect your final node(s) to `END`.

> **Gotcha — old tutorials use `set_entry_point`/`set_finish_point`.** Many search results and blog posts still call `builder.set_entry_point("first")` and `builder.set_finish_point("last")`. Those are older patterns. The current, canonical way is `builder.add_edge(START, "first")` and `builder.add_edge("last", END)`. Mixing stale patterns with new code is a common source of confusing errors.

### Compile, then run

`StateGraph` is a **builder** — a scaffold you configure. It cannot run until you call `.compile()`, which produces the runnable graph. The compiled graph exposes `invoke(input, config)` (run once, get the final state), `stream(...)` (get results step by step), and async versions (`ainvoke`, `astream`).

> **Gotcha — forgetting to compile.** Calling `.invoke()` on the builder instead of the compiled graph fails. The two-step "build, then compile" is deliberate: compiling validates the graph (checks for dead ends, etc.) before you run it.

### Hands-on exercise (Part 1)

Build a three-node **linear** pipeline that transforms a number and records what it did.

1. Define a `State` with `value: int` and `history: Annotated[list[str], operator.add]`.
2. Write three nodes: `add_five` (adds 5), `double_it` (multiplies by 2), `subtract_three` (subtracts 3). Each returns the new `value` **and** appends a one-line description to `history` (e.g. `"double_it: 10 -> 20"`).
3. Wire them `START → add_five → double_it → subtract_three → END` and compile.
4. Invoke with `{"value": 3, "history": []}` and print the result. Confirm `history` contains all three steps (proof your reducer works).
5. **Stretch:** remove the reducer (drop the `Annotated[..., operator.add]`) and re-run. Watch `history` collapse to only the last step — you've just reproduced the most common LangGraph bug on purpose, so you'll recognize it later.

### Interview questions (Part 1)

1. **Conceptual — "What problem does LangGraph solve that a straight-line pipeline can't?"**
   A straight-line pipeline (a chain) runs the same fixed steps in order and can't express branching (go different ways based on state) or looping (repeat steps an unknown number of times until a condition holds). LangGraph models the workflow as a graph with conditional edges and cycles, plus shared state, so those patterns become natural.

2. **Conceptual — "What are the three core pieces of a LangGraph graph?"**
   Nodes (steps, implemented as functions), edges (connections that route from one node to the next, possibly conditionally), and state (a shared data structure every node reads and updates via partial returns).

3. **Gotcha — "Your agent keeps forgetting earlier messages. What's the likely cause?"**
   The messages field has no accumulating reducer, so each node that returns `messages` replaces the whole list instead of appending. The fix is to use the `add_messages` reducer (or the prebuilt `MessagesState`), which appends new messages and preserves history.

4. **Practical — "A node returns `{'response': 'hi'}` but the state has three fields. What happens to the other two?"**
   They're left unchanged. Nodes return partial updates; LangGraph merges only the returned fields into the existing state, so any field the node didn't return keeps its previous value.

5. **Gotcha — "You call `.invoke()` and get an error before anything runs. Two common causes?"**
   Either you called `.invoke()` on the builder instead of the compiled graph (you must `.compile()` first), or the graph has a dead end — a node with no outgoing edge and no path to `END`.

---

## Part 2 — A simple agent: a ReAct-style loop with one tool

### Why this exists (the problem)

Suppose a user asks, "What's 4,891 times 237?" An LLM predicts text and is unreliable at exact arithmetic on unfamiliar numbers. The right move is to give it a calculator it can call. But that creates a *loop*: the model must (1) reason that it needs the calculator, (2) act by requesting it with arguments, (3) observe the result you return, and then (4) either answer or decide it needs another tool call — repeating until it can respond. This think→act→observe→repeat pattern is called **ReAct** (short for **Reasoning + Acting**). It's inherently cyclical, which is exactly why a graph — not a straight line — is the right structure.

Two terms:

- **Tool:** a function you expose to the model that it can request by name, supplying arguments. Critically, **the model does not run the tool** — it emits a structured "I want to call `calculator` with these arguments" request; *your code* runs the real function and returns the result. The model then continues, now with real data.
- **ReAct agent:** an LLM wired into a loop that lets it repeatedly reason, call tools, read results, and continue until it produces a final answer.

### Setting up the model and a tool

LangGraph is model-agnostic: it works with any provider through a LangChain-compatible wrapper. We'll use a chat model and attach ("bind") our tool to it so the model knows the tool exists and how to call it.

```python
from langchain_anthropic import ChatAnthropic   # or: from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '4891 * 237'."""
    return str(eval(expression))   # (eval is fine for a demo; never eval untrusted input in production)

model = ChatAnthropic(model="claude-sonnet-4-6")
model_with_tools = model.bind_tools([calculator])   # <-- tells the model the tool exists
```

> **Gotcha — you must `bind_tools`.** If you call the plain `model` instead of `model_with_tools`, the model has no idea the tool exists and will never request it (it'll just guess an answer). Binding is what advertises the tool's name and argument schema to the model.

### Building the loop by hand

We'll use `MessagesState` (the prebuilt state whose `messages` field appends via `add_messages`) so the growing conversation is preserved automatically. The loop needs two nodes:

- **agent node** — calls the model on the current messages. The model's reply (an "AI message") may contain tool-call requests.
- **tool node** — executes any tool calls the model requested and appends the results (as "tool messages").

LangGraph ships a prebuilt **`ToolNode`** that does the tool node's job: it reads the last AI message's tool calls, runs each corresponding tool, and returns the results as messages. We'll use it.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

def agent(state: MessagesState) -> dict:
    # Call the model on the full message history; return its reply to be appended.
    reply = model_with_tools.invoke(state["messages"])
    return {"messages": [reply]}

tool_node = ToolNode([calculator])   # runs requested tools, returns tool-result messages

# The routing decision: did the model ask for a tool, or is it done?
def should_continue(state: MessagesState) -> str:
    last = state["messages"][-1]
    if last.tool_calls:      # the AI message contains tool-call requests
        return "tools"       # -> go run them
    return END               # -> no tools requested, the model has answered

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")   # <-- THE CYCLE: after running tools, go back to the model
graph = builder.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "What's 4891 * 237?"}]})
print(result["messages"][-1].content)   # the final natural-language answer
```

Trace what happens: `agent` runs → model requests `calculator("4891 * 237")` → `should_continue` sees tool calls → routes to `tools` → `ToolNode` runs the calculator and appends the result → the `tools → agent` edge loops back → `agent` runs again, now the model sees the result and produces a final answer → `should_continue` sees no tool calls → routes to `END`.

> **Gotcha — route tool results *back to the agent*, not to `END`.** A frequent mistake is edging the tool node straight to `END`. Then the model never sees the tool's output — it requested a calculation and never got to use the answer. The tool node must loop back to the agent so the model can read the result and continue.

> **Gotcha — the model signals "done" by *not* calling a tool.** There's no explicit "stop" command. Termination happens when the model replies without any tool calls. Your router must handle that case (return `END`), or the loop never exits.

> **Gotcha — without `add_messages`, the loop breaks immediately.** Because this is a cycle that revisits the `messages` field repeatedly, an accumulating reducer is mandatory. `MessagesState` provides it. If you hand-roll a state with a plain `messages: list`, each pass overwrites the history and the model loses the very context the loop is meant to build.

### The one-line shortcut (and when to use it)

Building the loop by hand is worth doing once so you understand it — but for a standard tool-calling agent, LangGraph gives you the whole thing prebuilt:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools=[calculator])
result = agent.invoke({"messages": [{"role": "user", "content": "What's 4891 * 237?"}]})
```

`create_react_agent` builds exactly the agent-node + tool-node + conditional-loop structure you just wrote — it's a convenience wrapper over the same graph mechanics. The practical rule: **start with `create_react_agent` for any standard tool-calling agent; drop down to a manual `StateGraph` only when you need custom branching, parallel steps, retry logic, or other control the prebuilt doesn't give you.** (LangGraph also provides a prebuilt `tools_condition` router that replaces the `should_continue` function above, if you want the manual graph with less boilerplate.)

### Hands-on exercise (Part 2)

1. Define a single tool `get_weather(city)` that returns a hard-coded string like `"Paris: 18°C, cloudy"`.
2. Bind it to a chat model with `bind_tools`.
3. Build the manual ReAct graph exactly as above (`agent` node, `ToolNode`, `should_continue` router, and the `tools → agent` cycle) using `MessagesState`.
4. Invoke with `"What's the weather in Paris, and should I bring an umbrella?"`. Print **all** messages in the final state and identify each: the human question, the AI's tool request, the tool result, and the AI's final answer.
5. **Stretch:** replace your entire graph with a single `create_react_agent(model, tools=[get_weather])` call and confirm you get the same behavior — then deliberately edge the tool node to `END` instead of back to `agent` in your manual version and observe how the answer degrades (the model never uses the weather it fetched).

### Interview questions (Part 2)

1. **Conceptual — "What does ReAct mean, and why does it need a graph rather than a chain?"**
   ReAct is "Reasoning + Acting": the model reasons about what to do, acts by calling a tool, observes the result, and repeats until it can answer. Because the number of tool calls isn't known in advance, the workflow must loop — and looping requires a cyclical graph, which a straight-line chain can't express.

2. **Practical — "Walk me through the two nodes and the edges of a minimal ReAct loop."**
   An `agent` node calls the model on the message history; a `tools` node runs any requested tools and appends their results. Edges: `START → agent`; a conditional edge from `agent` that goes to `tools` if the model requested a tool or to `END` otherwise; and `tools → agent` to close the loop so the model can read the tool results.

3. **Gotcha — "You built a ReAct agent but it answers the arithmetic question wrong, as if it never used the calculator. Two likely bugs?"**
   Either you called the model without `bind_tools` (so it never knew the tool existed and just guessed), or you routed the tool node to `END` instead of back to the agent (so the model requested the calculation but never saw the result).

4. **Conceptual — "In a ReAct loop, how does the agent decide to stop?"**
   It stops when the model produces a reply with no tool calls. The router checks the last AI message: if it contains tool calls, route to the tool node; if not, route to `END`. There is no explicit stop signal beyond the absence of tool calls.

5. **Practical — "When would you build the loop manually instead of using `create_react_agent`?"**
   When you need control the prebuilt doesn't offer: custom routing/branching, parallel node execution, retry or validation logic, a supervisor-worker structure, or a non-standard state. For a plain tool-calling agent, the prebuilt is preferred because it's the same mechanics with less code.

---

## Part 3 — Conditional edges & cycles: routing, loops, termination

### Why this exists (the problem)

Part 2 quietly used the two features this part generalizes: a **conditional edge** (route to the tool node *or* to the end, depending on state) and a **cycle** (loop back to the agent). These are LangGraph's defining capabilities. Real workflows lean on them constantly: route a support ticket to billing, technical, or general handling based on its category; retry a flaky step until it succeeds or you give up; keep gathering information until you have enough. Conditional edges express **decisions**; cycles express **repetition**. Getting both right — especially making loops *stop* — is the heart of building reliable graphs.

### Conditional edges

A normal edge (`add_edge("A", "B")`) always goes A→B. A **conditional edge** instead runs a **router function** after a node and sends execution wherever that function decides. You wire it with `add_conditional_edges(source_node, router_fn, mapping)`:

- `source_node` — after this node runs, consult the router.
- `router_fn(state)` — reads the state and returns a string.
- `mapping` — a dict from each possible returned string to the target node's name (you can map to `END`).

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SupportState(TypedDict):
    message: str
    category: str      # set by classify; drives the routing
    answer: str

def classify(state: SupportState) -> dict:
    text = state["message"].lower()
    if "refund" in text or "charge" in text:
        return {"category": "billing"}
    if "error" in text or "crash" in text:
        return {"category": "technical"}
    return {"category": "general"}

def route(state: SupportState) -> str:
    # Return a key that exists in the mapping; include a safe fallback.
    return state["category"] if state["category"] in ("billing", "technical") else "general"

# ... define billing_node, technical_node, general_node (each sets 'answer') ...

builder = StateGraph(SupportState)
builder.add_node("classify", classify)
builder.add_node("billing", billing_node)
builder.add_node("technical", technical_node)
builder.add_node("general", general_node)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route, {
    "billing": "billing",
    "technical": "technical",
    "general": "general",
})
for name in ("billing", "technical", "general"):
    builder.add_edge(name, END)
graph = builder.compile()
```

> **Gotcha — the router's return value must exist in the mapping.** If your router can return a value that isn't a key in the mapping (say the classifier produced an unexpected label), routing fails. Always constrain the router's output and include a safe fallback branch (here, anything unexpected falls to `"general"`). This matters doubly when an LLM produces the routing label, because models occasionally return something off-script.

### Cycles (loops) and how to make them terminate

A **cycle** is an edge (or path) that leads back to an earlier node, so a step or group of steps runs repeatedly. The tool→agent edge in Part 2 was a cycle. Cycles are what let an agent try again, keep calling tools, or iterate toward a goal. The catch: **every loop must have a guaranteed exit**, or it runs forever.

There are two standard ways to guarantee termination:

1. **A condition that eventually becomes true.** The router returns `END` once the work is done (e.g. "the model answered without a tool call," or "the result passed validation").
2. **A hard cap.** Track an attempt counter in state and force `END` after N tries, even if the condition never holds.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RetryState(TypedDict):
    task: str
    result: str
    attempts: int

def attempt(state: RetryState) -> dict:
    n = state["attempts"] + 1
    outcome = try_the_task(state["task"])          # may succeed or fail
    return {"result": outcome, "attempts": n}

def check(state: RetryState) -> str:
    if state["result"] == "success":
        return END
    if state["attempts"] >= 3:                     # hard cap: give up after 3 tries
        return END
    return "attempt"                               # otherwise loop back and retry

builder = StateGraph(RetryState)
builder.add_node("attempt", attempt)
builder.add_edge(START, "attempt")
builder.add_conditional_edges("attempt", check, {"attempt": "attempt", END: END})
graph = builder.compile()
```

> **Gotcha — infinite loops and the recursion limit.** If your termination condition never triggers (a bug in `check`, or a model that keeps requesting tools), the graph would loop forever. LangGraph protects you with a **recursion limit** — a cap on how many steps (super-steps) a single run may take, **defaulting to 25**. Exceed it and you get a `GraphRecursionError`. You *can* raise the limit (via the run config), but hitting it almost always means your loop's exit logic is broken, not that the limit is too low. Treat the error as "my termination condition isn't firing," not "increase the number."

> **Gotcha — forgetting the loop-back edge (or the exit edge).** Two mirror-image mistakes: no edge back to the earlier node means there's no loop at all (it runs once); no branch to `END` in the router means there's no way out. A working cycle needs *both* — a path back and a path out.

> **Gotcha — a dead end still errors, even mid-graph.** Every node reachable in your graph must have somewhere to go. A specialist node you forgot to connect to `END` (or to another node) will trip the same dead-end error you'd get in a linear graph.

Advanced (pointers only, so you know the words): a conditional edge's router can return a **list** of node names to fan out and run several nodes in parallel; and LangGraph's `Send` primitive lets you dynamically dispatch many parallel copies of a node with different inputs (a map-reduce pattern). You don't need these for basic routing and loops, but you'll encounter them later.

### Hands-on exercise (Part 3)

Two small graphs.

1. **Routing:** build the support router above. Implement `classify` with simple keyword rules and three specialist nodes that each set a distinct `answer`. Send three different messages and confirm each reaches the correct specialist and that an odd message falls through to `general`.
2. **Cycle with termination:** build the retry graph. Make `try_the_task` return `"success"` randomly ~40% of the time and `"fail"` otherwise. Run it several times and confirm it either succeeds early or stops after exactly 3 attempts — never more.
3. **Stretch:** deliberately break the retry graph by removing the `state["attempts"] >= 3` cap *and* making `try_the_task` always fail. Run it and watch LangGraph raise `GraphRecursionError` at the default limit — then fix it by restoring the cap. You've now seen both the safety net and the correct way to rely on it.

### Interview questions (Part 3)

1. **Conceptual — "What's the difference between a normal edge and a conditional edge?"**
   A normal edge always routes from one fixed node to another. A conditional edge runs a router function after a node; the function reads the state and returns a value that a mapping translates into the next node (possibly `END`), so the path is decided at runtime based on state.

2. **Practical — "How do you make a loop in a LangGraph agent terminate?"**
   Give it a guaranteed exit: a router branch that returns `END` when a condition becomes true (e.g. the task succeeded, or the model answered without tool calls), and/or a hard cap using an attempt counter in state that forces `END` after N iterations.

3. **Gotcha — "You get a `GraphRecursionError`. What does it mean and what should you do?"**
   The run exceeded the recursion limit (the cap on super-steps, default 25), which almost always means a loop isn't terminating — a broken exit condition or a model that keeps requesting tools. The fix is to correct the termination logic, not to reflexively raise the limit.

4. **Gotcha — "Your LLM-based classifier occasionally sends a run into an error at the routing step. Why, and how do you prevent it?"**
   The model returned a label that isn't a key in the conditional edge's mapping, so routing has nowhere to go. Constrain the model's output and add a safe fallback branch that catches anything unexpected (e.g. route unknown labels to a `general` node).

5. **Conceptual — "Why is the ability to create cycles considered LangGraph's defining feature?"**
   Because agentic behavior — retrying, iterating, and repeatedly calling tools until a task is done — requires looping an unknown number of times, and cycles are what let a workflow revisit earlier steps. Straight-line pipelines can't express that, so cyclical graphs are the key capability LangGraph adds.

---

## Part 4 — Persistence & checkpoints: save/resume state, time-travel debugging

### Why this exists (the problem)

By default, when a graph run finishes, its state disappears. That causes three concrete problems: (1) **no memory** — a chatbot forgets everything the moment a run ends, so it can't hold a multi-turn conversation; (2) **no resume** — you can't pause a workflow (say, to wait for a human's approval) and continue later from the same place; (3) **no recovery** — if the process crashes mid-run, you restart from scratch. **Persistence** solves all three by saving the state as the graph runs, so it can be reloaded and continued. It's also the foundation for pausing an agent for human review and for debugging by "time travel."

Three terms:

- **Checkpointer:** an object you attach when compiling the graph. It automatically saves a snapshot of the state after **every step** (LangGraph calls one step a "super-step").
- **Checkpoint:** one saved snapshot of the state at a moment in time, with a unique, increasing ID.
- **Thread:** a series of checkpoints belonging to one conversation or task, identified by a **`thread_id`**. Different `thread_id`s are fully isolated — like separate chat conversations that never see each other's history.

### Turning on persistence

Attach a checkpointer at compile time, then pass a `thread_id` in the run config. Reusing the same `thread_id` continues that thread (the graph reloads its saved state first); a new `thread_id` starts fresh.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-1"}}

# First turn on thread "user-1"
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Bob."}]}, config)

# Second turn, SAME thread_id -> the graph remembers the first turn
graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]}, config)
# The model can answer "Bob" because the saved state (with the earlier messages) was reloaded.

# A DIFFERENT thread starts with a clean slate:
graph.invoke({"messages": [{"role": "user", "content": "What's my name?"}]},
             {"configurable": {"thread_id": "user-2"}})   # this thread has no memory of Bob
```

> **Gotcha — once a checkpointer is set, every invocation needs a `thread_id`.** Without `{"configurable": {"thread_id": ...}}`, the checkpointer doesn't know which thread to read or write, and the call fails. The `thread_id` is how state is keyed.

### Inspecting and time-traveling through state

Because the checkpointer keeps a *history* of checkpoints (it doesn't overwrite the previous one), you can inspect the past and even resume from it:

- **`graph.get_state(config)`** returns the current snapshot for a thread — its values, which node(s) come next, and the current `checkpoint_id`.
- **`graph.get_state_history(config)`** returns all checkpoints for the thread, newest first — a full timeline.
- **Time travel:** each checkpoint has a `checkpoint_id`. Pass both `thread_id` and `checkpoint_id` in the config to read *that* past state, or to **resume the run from that point** — optionally with different inputs — creating a branch ("fork") from the past.

```python
# See the full timeline for a thread
history = list(graph.get_state_history(config))
for snapshot in history:
    print(snapshot.config["configurable"]["checkpoint_id"], "->", snapshot.values)

# Pick an earlier checkpoint and resume from it (passing None as input = continue, don't add new input)
past = history[2]                                            # some earlier point
past_config = past.config                                    # includes its checkpoint_id
graph.invoke(None, past_config)                              # re-runs forward from that saved state
```

This "time-travel debugging" is useful: when a run goes wrong, you can jump back to the step before things broke, change an input or fix a node, and re-run only the tail — instead of replaying the whole thing from the start.

### Choosing a backend

The checkpointer's *storage backend* decides where checkpoints live:

- **`InMemorySaver`** (from `langgraph.checkpoint.memory`) — keeps everything in RAM. Great for prototyping and tests. **Everything is lost when the process exits.** (Note: `MemorySaver` is the older name for the same in-memory saver and still appears in many examples; `InMemorySaver` is the current name.)
- **`SqliteSaver`** (from `langgraph.checkpoint.sqlite`) — writes to a SQLite file on disk, so state survives restarts. Good for single-machine apps.
- **`PostgresSaver`** (from `langgraph.checkpoint.postgres`) — writes to a PostgreSQL database for production/multi-instance use; call its `.setup()` once to create the tables.

> **Gotcha — `InMemorySaver` is dev-only.** It's the default choice in tutorials because it's zero-config, but it holds state in memory, so a restart wipes every conversation. For anything real, use a durable backend (SQLite for one machine, Postgres for production).

> **Gotcha — checkpoints accumulate.** Every super-step writes a checkpoint, and long conversations pile up a lot of them, which raises storage cost and can slow retrieval. In production, prune old checkpoints or set a retention policy (e.g. delete threads older than N days).

> **Gotcha — don't stuff large blobs into state.** State is snapshotted at *every* step, so a big object (a large file, a huge blob of text) gets copied into each checkpoint, bloating storage fast. Keep large data in external storage and put only a reference (like a URL or file ID) in the state.

> **Gotcha — checkpoint deserialization can be a security concern in production.** Loading a checkpoint deserializes stored Python objects. New applications should restrict this to known-safe types (LangGraph exposes a strict mode via an environment variable / an explicit allow-list for the serializer). It's a production hardening step, not something to worry about in a local demo — but know it exists.

A brief related distinction so the vocabulary doesn't confuse you later: a **checkpointer** saves *per-thread* state (this conversation's history). A **store** (e.g. `InMemoryStore`) is a separate mechanism for **cross-thread**, long-term memory — facts you want available across many conversations (user preferences, shared knowledge). Many apps use both: a checkpointer for the current thread and a store for durable, thread-spanning data. This part is about checkpoints; the store is its cross-thread cousin.

### Hands-on exercise (Part 4)

Give the ReAct agent from Part 2 a memory.

1. Compile that agent with `checkpointer=InMemorySaver()`.
2. On `thread_id="a"`, run two turns: first `"My name is Sam and I like hiking."`, then `"What activity did I say I like?"`. Confirm it answers "hiking" — proof the thread's state persisted between invocations.
3. On a fresh `thread_id="b"`, ask `"What's my name?"` and confirm it does **not** know — proof threads are isolated.
4. Call `graph.get_state_history(config)` for thread `"a"` and print each checkpoint's id and values to see the timeline.
5. **Stretch:** grab an earlier checkpoint's config from the history and `invoke(None, that_config)` to resume from the past — then swap `InMemorySaver` for `SqliteSaver`, run a turn, restart your script, and confirm the conversation is still there (memory that survives a restart).

### Interview questions (Part 4)

1. **Conceptual — "What does persistence give a LangGraph app, and what are the three things it enables?"**
   Persistence saves the graph's state as it runs (via a checkpointer) so it can be reloaded and continued. It enables memory across turns/sessions (conversational agents), pause-and-resume (including human-in-the-loop approval), and recovery after a crash by replaying from the last checkpoint.

2. **Conceptual — "Explain checkpointers, checkpoints, and threads."**
   A checkpointer is the object that saves state; it writes a checkpoint (a state snapshot with a unique ID) after every super-step. Checkpoints are grouped into threads, each identified by a `thread_id` representing one conversation or task, and threads are isolated from one another.

3. **Practical — "How do you make a chatbot remember a conversation across multiple `invoke` calls?"**
   Compile the graph with a checkpointer and pass the same `thread_id` in the config on each call. LangGraph reloads that thread's saved state before each run, so earlier messages remain in context; a different `thread_id` starts a fresh, isolated conversation.

4. **Gotcha — "Your chatbot works in development but loses all conversations whenever you redeploy. Why?"**
   You're using `InMemorySaver`, which keeps checkpoints in RAM and loses them when the process restarts. Switch to a durable backend — `SqliteSaver` for a single machine or `PostgresSaver` for production.

5. **Practical — "What is time-travel debugging in LangGraph and how do you do it?"**
   Because the checkpointer keeps a history of checkpoints, you can inspect past states with `get_state_history` and resume from any earlier checkpoint by passing its `thread_id` and `checkpoint_id` in the config (invoking with `None` to continue without new input). This lets you jump back to before a failure, adjust, and re-run only the remainder instead of starting over.

---

## Gotchas summary table

| # | Gotcha | Where it bites | Fix / rule of thumb |
|---|---|---|---|
| 1 | No reducer silently overwrites list state (lost history) | State design | Use an accumulating reducer; for messages use `add_messages` / `MessagesState` |
| 2 | Reducers with side effects cause chaos | State design | Keep reducers pure: combine inputs, return result, nothing else |
| 3 | Node with no outgoing edge → dead-end error | Wiring any graph | Connect every reachable node onward; final nodes to `END` |
| 4 | `set_entry_point`/`set_finish_point` are stale patterns | Following old tutorials | Use `add_edge(START, ...)` and `add_edge(..., END)` |
| 5 | Calling `.invoke()` on the builder | Running the graph | `.compile()` first, then invoke the compiled graph |
| 6 | Calling the model without `bind_tools` | Building an agent | Bind tools so the model knows they exist and how to call them |
| 7 | Routing tool results to `END` instead of back to the agent | ReAct loop | Edge the tool node back to the agent so the model sees results |
| 8 | Expecting an explicit "stop" signal | ReAct loop | The model stops by replying with no tool calls; route that to `END` |
| 9 | Router returns a value not in the mapping | Conditional edges | Constrain router output; add a safe fallback branch |
| 10 | Loop with no exit → runs forever | Cycles | Provide a termination condition and/or a hard attempt cap |
| 11 | `GraphRecursionError` (hit the default 25-step limit) | Cycles | Fix the exit logic; don't just raise the limit |
| 12 | Missing loop-back edge or missing exit edge | Cycles | A working loop needs both a path back and a path out |
| 13 | Every invoke needs a `thread_id` once a checkpointer is set | Persistence | Always pass `{"configurable": {"thread_id": ...}}` |
| 14 | `InMemorySaver` loses everything on restart | Persistence | Use `SqliteSaver` (disk) or `PostgresSaver` (production) |
| 15 | Checkpoints accumulate → cost/latency | Persistence at scale | Prune old checkpoints / set a retention policy |
| 16 | Large blobs in state bloat every checkpoint | Persistence | Store big data externally; keep only a reference in state |
| 17 | Checkpoint deserialization security | Production | Restrict to safe types (strict serializer / allow-list) |

---

## Quick reference card

**Install & imports**

```text
pip install langgraph langgraph-prebuilt langchain-anthropic   # Python 3.10+
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
```

**Build → compile → run**

```text
builder = StateGraph(State)
builder.add_node("name", fn)                 # fn: (state) -> partial update dict
builder.add_edge(START, "name")              # normal edge
builder.add_edge("name", END)
graph = builder.compile()                    # (optionally checkpointer=...)
graph.invoke(input_dict, config)             # also: stream / ainvoke / astream
```

**State**

- `TypedDict` (or Pydantic) with named fields; nodes return **partial** updates that LangGraph merges.
- Default merge = overwrite (last write wins). To accumulate, annotate: `Annotated[list, operator.add]`.
- Messages: `Annotated[list, add_messages]` (or use `MessagesState`) — appends, preserves history.

**Edges**

- Normal: `add_edge("A", "B")` (always A→B).
- Conditional: `add_conditional_edges("A", router_fn, {ret_value: "target", ..., END: END})`; `router_fn(state) -> str`.
- Cycle: an edge back to an earlier node (e.g. `add_edge("tools", "agent")`).

**Minimal ReAct loop**

```text
START → agent → [conditional] → tools → agent (loop) ...→ END
agent: model_with_tools.invoke(messages)     # bind_tools first!
tools: ToolNode(tools)                        # runs requested tools
router: has tool_calls? -> "tools" else END   # (or use prebuilt tools_condition)
shortcut: create_react_agent(model, tools=[...])   # builds all of the above
```

**Cycles / termination**

- Exit on a condition (`return END` when done) and/or a hard cap (`attempts >= N → END`).
- Safety net: recursion limit (default **25** super-steps) → `GraphRecursionError` = broken exit logic.

**Persistence**

```text
graph = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "user-1"}}   # required with a checkpointer
graph.invoke(input, config)                          # same thread_id = remembers
graph.get_state(config)                              # current snapshot
graph.get_state_history(config)                      # full timeline (time travel)
graph.invoke(None, {"configurable": {"thread_id": "user-1", "checkpoint_id": "<id>"}})  # resume from past
```

- Backends: `InMemorySaver` (RAM, dev; older alias `MemorySaver`), `SqliteSaver` (disk), `PostgresSaver` (prod, `.setup()`).
- Checkpointer = per-thread state; **Store** (`InMemoryStore`) = cross-thread long-term memory.

**Reference material:** LangGraph docs — graph API / state & reducers, prebuilt ReAct agent tutorials, control-flow (conditional edges & cycles) docs, and the persistence (checkpointers, threads, time travel) docs.

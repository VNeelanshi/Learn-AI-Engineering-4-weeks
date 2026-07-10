# Tool Use & Function Calling (OpenAI and Anthropic)

A tutorial organized into four parts. Each part explains *why* something exists, *how* to do it and ends with a practical exercise. Every part also closes with five interview-style questions and answers. A gotchas summary table and a quick reference card are at the very end.

The four parts:

1. **Function calling foundations** — what it is, and the exact request/response shapes for both providers.
2. **Parallel tool calls** — asking for several tool calls at once, and why that saves time.
3. **Streaming with tools** — receiving the answer piece by piece, and the JSON problem it creates.
4. **Error handling** — what to do when a tool (or the model) misbehaves.

> **Note on version names.** Model names in the examples (e.g. `claude-sonnet-4-6`, `gpt-5.5`) are just current examples; providers rename models often. The *shapes* of the requests and responses shown here are the stable part — check the official docs for the exact model string of the day.

---

## Part 1 — Function calling foundations: schemas and responses

### Why this exists (the problem)

A large language model (an "LLM" — a program that predicts text one token at a time) can only produce text. On its own it cannot look up today's weather, query your database, send an email, or do reliable arithmetic on numbers it has never seen. It has no hands and no live connection to the world; it only continues the text you give it.

**Function calling** (also called **tool use** — the two names mean the same thing) is the bridge. You tell the model, "here are some functions you're allowed to ask me to run, and here is the shape of the input each one needs." When the model decides it needs one, it doesn't run anything itself — it *emits a structured request* that names the function and supplies the arguments. Your code runs the real function, then hands the result back to the model. The model then continues, now armed with real information.

The mental model that makes this click: **tool use is a contract.** You define what operations exist and what their inputs look like; the model decides *when* and *with what arguments* to call them; your code (or the provider's servers) actually executes them and returns a result. The model never touches your implementation — it only ever sees the description you wrote and the result you sent back. If you have written any typed API before, it's the same idea, except the caller on the other end is a language model choosing which function to invoke based on the conversation.

Two words you'll see constantly:

- **Tool / function**: a capability you expose to the model (e.g. `get_weather`). Described by a name, a human-readable description, and a schema for its inputs.
- **Schema**: a machine-readable description of what an input looks like — which fields exist, their types, and which are required. Both providers use **JSON Schema**, a standard JSON-based format for describing the shape of JSON data. A minimal object schema looks like `{"type": "object", "properties": {...}, "required": [...]}`.

### The round trip (the loop)

Every function-calling interaction follows the same five steps, regardless of provider:

1. **You** send the user's message *plus* a list of tool definitions.
2. **The model** either answers in plain text, or responds with a structured "I want to call tool X with these arguments" request. When it does the latter, it stops and waits.
3. **You** read that request, run the corresponding real function in your code, and get a result.
4. **You** send the conversation back, appending the result.
5. **The model** continues — usually turning the raw result into a natural answer, or asking for another tool call.

Steps 2–5 can repeat many times in a single task. That repeating cycle is called the **agentic loop** or **tool loop**.

> **Gotcha — the model does not run your code.** A common beginner misconception is that the model "calls" the function. It doesn't. It only produces a *description* of the call it wants. If you never execute the function and send the result back, the model just sits there having asked for something it never received. You own execution.

Now the concrete shapes. This is where the two providers differ, so read carefully.

### Anthropic (Claude) — the Messages API

You define tools in a top-level `tools` array. Each tool has `name`, `description`, and `input_schema` (the JSON Schema for its arguments).

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Paris'"}
            },
            "required": ["city"],
        },
    }
]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

print(resp.stop_reason)  # -> "tool_use"
```

When Claude wants a tool, the response's `stop_reason` is the string `"tool_use"`, and the response `content` is a list of **content blocks**. A content block is one piece of the response; the list can mix a `text` block ("Let me check that for you…") with one or more `tool_use` blocks. A `tool_use` block has three fields you care about:

- `id` — a unique identifier for this specific call (you'll need it to return the result).
- `name` — which tool the model wants.
- `input` — the arguments, **already parsed as a native object/dict** (not a string).

You run your real function, then continue the conversation by sending a **new `user` message** whose content is a `tool_result` block:

```python
# Pull out the tool_use block
tool_block = next(b for b in resp.content if b.type == "tool_use")
# tool_block.id, tool_block.name, tool_block.input  (input is a dict!)

result = get_weather(**tool_block.input)  # your real function

followup = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": resp.content},   # echo the model's turn back verbatim
        {"role": "user", "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_block.id,   # matches the id from the tool_use block
                "content": str(result),         # the result, as text
                # "is_error": True,             # set this if the tool failed (see Part 4)
            }
        ]},
    ],
)
```

> **Gotcha — you must echo the assistant turn.** The follow-up request has to include the model's own `tool_use` message (the `assistant` turn) *before* your `tool_result`. If you drop it, the API has a `tool_result` referencing a `tool_use` it can't see, and rejects the request.

> **Gotcha — tool_result must immediately follow its tool_use.** In the message history, the `user` message carrying `tool_result` blocks must come directly after the `assistant` message that made the request. You cannot slip another message in between.

### OpenAI — the Chat Completions API (the widely used classic)

OpenAI has *two* current APIs for this, and they use **different tool schemas**. We'll cover both. Start with Chat Completions, the one most existing code and tutorials use.

Here, each tool is wrapped in an object with `type: "function"` and a **nested `function` key** that holds `name`, `description`, and `parameters` (the JSON Schema — note the field is called `parameters`, not `input_schema`).

```python
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from the environment

tools = [
    {
        "type": "function",
        "function": {                       # <-- nested "function" wrapper
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {                 # <-- called "parameters" here
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]

resp = client.chat.completions.create(
    model="gpt-5.5",
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

msg = resp.choices[0].message
print(resp.choices[0].finish_reason)  # -> "tool_calls"
```

When the model wants tools, `finish_reason` is `"tool_calls"` and the assistant message has a `tool_calls` array. Each entry has:

- `id` — the identifier for this call.
- `type` — `"function"`.
- `function.name` — which tool.
- `function.arguments` — the arguments **as a JSON-encoded string** (not a parsed object).

You return the result as a message with `role: "tool"`:

```python
import json

call = msg.tool_calls[0]
args = json.loads(call.function.arguments)   # arguments is a STRING -> parse it
result = get_weather(**args)

followup = client.chat.completions.create(
    model="gpt-5.5",
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        msg,                                              # the assistant message with tool_calls
        {"role": "tool", "tool_call_id": call.id, "content": str(result)},
    ],
)
```

> **Gotcha — `arguments` is a string in OpenAI, an object in Anthropic.** This is the single most common cross-provider mistake. In OpenAI's `tool_calls`, `function.arguments` is a JSON *string* you must `json.loads()` before using. In Anthropic's `tool_use`, `input` is already a parsed dict. Treating the OpenAI string as a dict (or vice versa) crashes your code.

> **Gotcha — `arguments` can be malformed JSON.** Because the model *generates* that string token by token, it can occasionally be invalid JSON (a stray quote, a truncated object). Always wrap `json.loads` in a `try/except` rather than assuming it parses. We return to this in Part 4.

### OpenAI — the Responses API (OpenAI's newer, recommended default)

OpenAI now positions the **Responses API** as its go-forward interface (its older Assistants API is being retired in favor of it). It's designed to run an agentic loop for you and to keep state across turns. Crucially for this tutorial, **its tool schema is flatter** than Chat Completions: there is *no* nested `function` wrapper — `name`, `description`, and `parameters` sit directly on the tool object.

```python
from openai import OpenAI
client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "get_weather",             # <-- FLAT: no nested "function" key
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    }
]

resp = client.responses.create(
    model="gpt-5.5",
    tools=tools,
    input="What's the weather in Paris?",   # "input" instead of "messages"
)
```

The response has an `output` array of typed **items**. A tool call is an item with `type == "function_call"`, carrying `call_id`, `name`, and `arguments` (still a JSON string). You return the result as an input item of type `function_call_output`:

```python
import json

for item in resp.output:
    if item.type == "function_call":
        args = json.loads(item.arguments)      # still a string
        result = get_weather(**args)

        followup = client.responses.create(
            model="gpt-5.5",
            tools=tools,
            input=[
                {"type": "function_call_output",
                 "call_id": item.call_id,
                 "output": str(result)},
            ],
            previous_response_id=resp.id,       # chains this turn onto the previous one
        )
```

> **Gotcha — the two OpenAI schemas are not interchangeable.** If you paste a Chat Completions tool (with the nested `function` wrapper) into a Responses API call, you get an error; if you paste a flat Responses tool into Chat Completions, you get `Missing required parameter: 'tools[0].function'`. Decide which API you're on and use its shape. This trips up people constantly because both live under the same `openai` client and both use `type: "function"`.

### Side-by-side: the three shapes

| Concept | Anthropic (Messages) | OpenAI (Chat Completions) | OpenAI (Responses) |
|---|---|---|---|
| Where tools go | `tools=[...]` | `tools=[...]` | `tools=[...]` |
| Tool definition | `{name, description, input_schema}` | `{type:"function", function:{name, description, parameters}}` | `{type:"function", name, description, parameters}` |
| Schema field name | `input_schema` | `parameters` (nested) | `parameters` (flat) |
| User input field | `messages=[...]` | `messages=[...]` | `input=` (string or list) |
| "It wants a tool" signal | `stop_reason == "tool_use"` | `finish_reason == "tool_calls"` | item `type == "function_call"` |
| Call arguments | `block.input` (**dict**) | `call.function.arguments` (**string**) | `item.arguments` (**string**) |
| Call id field | `block.id` | `call.id` | `item.call_id` |
| Return result as | `user` msg with `tool_result` block (`tool_use_id`) | `role:"tool"` msg (`tool_call_id`) | input item `function_call_output` (`call_id`) |

> **Gotcha — a "wants a tool" response can still contain text.** The model may narrate ("I'll check the weather…") *and* request a tool in the same turn. Don't assume tool-call and text are mutually exclusive; handle both blocks/parts.

> **Gotcha — forcing a tool.** By default the model decides whether to call a tool at all. Both providers let you force it: Anthropic via `tool_choice` (e.g. require any tool, or a specific one); OpenAI via `tool_choice` as well (`"auto"`, `"required"`, or naming a function). Useful when you *know* a tool is needed, but overusing it removes the model's judgment about when a plain answer would do.

### Beyond two providers: the wider landscape

This tutorial contrasts OpenAI and Anthropic because they're the two most common APIs and they show the range of design choices — but they're far from the only providers with function calling. The reassuring part is that what you actually have to learn isn't one interface per provider; it's a *small number of schemas*, because most providers converge on a few shapes.

**OpenAI's format is the de facto standard.** A large share of other providers accept the very same Chat Completions "tools" format shown above, so learning it carries over broadly. Mistral's models follow the OpenAI tools shape; Llama models (via hosting providers, or self-hosted) support tool use through chat templates that match the OpenAI format; and DeepSeek, xAI's Grok, Qwen, Cohere's OpenAI-compatible endpoint, and most local servers (such as Ollama and vLLM) all expose OpenAI-compatible tool calling. In short, if you know the OpenAI shape you can already talk to a big chunk of the ecosystem.

> **Gotcha — "OpenAI-compatible" doesn't guarantee *identical* behavior.** A provider can accept the OpenAI request shape yet differ in the details: how strictly it validates arguments, whether it supports parallel calls, how it streams, or quirks in the returned JSON. Treat "compatible" as "a good starting point," and test tool calling against the specific provider rather than assuming perfect parity.

**Google Gemini is the third notable shape.** Gemini has its own conventions rather than mirroring OpenAI. You declare tools under a `function_declarations` field:

```python
# Gemini tool definition (exact field casing can vary by SDK/version — check the current Gemini docs)
tools = [{
    "function_declarations": [{
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {                    # JSON Schema — same idea as the others
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }]
}]
```

When Gemini wants a tool, the response contains a **`functionCall`** part carrying the tool `name` and its arguments (`args`) as an **already-parsed object** (like Anthropic's `input`, not a JSON string). You reply by sending back a **`functionResponse`** part with the tool's output. For guaranteed-shape structured output (the equivalent of forcing a JSON result), Gemini uses a separate `response_schema` setting.

So realistically it's about **three schemas** to understand deeply — OpenAI's (which most others reuse), Anthropic's, and Gemini's — rather than one per provider.

**You rarely hand-write against each one.** Normalization layers hide the differences behind a single interface, so your tool-definition and result-parsing logic lives in one place:

- **Amazon Bedrock's Converse API** normalizes tool definitions across the models it hosts (Claude, Llama, Mistral, Amazon's own Nova, and others), so one code path serves many models.
- **Client libraries** such as LangChain (its `bind_tools` method) and LiteLLM let you define a tool once and route the same call to different providers, adapting the schema for each behind the scenes.

If you're building anything that might switch or mix providers, prefer one of these layers over hand-maintaining a separate schema and parser per provider — that per-provider surface (the exact lines that differ between OpenAI, Anthropic, and Gemini) is precisely what a normalization layer absorbs.

### Hands-on exercise (Part 1)

Build the full loop once, end to end, for **one** provider of your choice.

1. Write a real Python function `get_weather(city)` that just returns a hard-coded string like `"Paris: 18°C, cloudy"` (no real API needed).
2. Define the matching tool schema in the correct shape for your chosen provider.
3. Send the message `"What's the weather in Paris, and should I bring an umbrella?"`.
4. Detect the tool request (`stop_reason`/`finish_reason`/item type), extract the arguments (remember: **parse the string** if you're on OpenAI), and call your function.
5. Send the result back and print the model's final natural-language answer.
6. **Stretch:** run the exact same flow against a *second* provider by changing only the schema shape and the result-return format. Notice precisely which lines had to change — that's the surface area you must abstract if you ever want provider-agnostic code.

### Interview questions (Part 1)

1. **Conceptual — "Explain function calling in one sentence, and say what the model actually does."**
   Function calling lets you expose functions to an LLM so it can request them by name with structured arguments; the model does not execute anything — it emits a structured call, your code runs the real function and returns the result, and the model continues from there.

2. **Practical — "You're integrating both OpenAI and Anthropic. What's the first difference that will break your code if you ignore it?"**
   The arguments type: OpenAI returns `arguments` as a JSON *string* (you must `json.loads` it), while Anthropic returns `input` as an already-parsed object. A close second is OpenAI's own split — Chat Completions nests the tool under a `function` key while the Responses API is flat.

3. **Gotcha — "In an Anthropic multi-turn tool flow, the API rejects your follow-up request. What's the most likely cause?"**
   You didn't include the assistant's `tool_use` message before the `tool_result`, or the `tool_result` message isn't immediately after the `tool_use` message. Anthropic requires the tool result to directly follow the matching tool-use turn.

4. **Conceptual — "What is JSON Schema and why do both providers use it for tools?"**
   JSON Schema is a standard for describing the shape of JSON data — field names, types, and which are required. Providers use it so the model knows exactly what arguments to produce and so the platform can validate/guide generation; it's a language-neutral contract for the tool's inputs.

5. **Practical — "When would you set `tool_choice` to force a tool instead of leaving it on auto?"**
   When the task definitionally requires the tool (e.g. every request must go through a lookup or a structured extractor) and you don't want the model to occasionally answer from memory. The trade-off is that you lose the model's judgment about when a direct answer would be better, so you force sparingly.

---

## Part 2 — Parallel tool calls

### Why this exists (the problem)

Imagine a user asks, "Compare the weather in Paris, Tokyo, and New York." With what you learned in Part 1, the naive flow is: model asks for Paris → you run it → model asks for Tokyo → you run it → model asks for New York → you run it. That's **three separate model turns**, each one a full network round trip and a full inference pass, done strictly one after another. It's slow, and the three lookups didn't even depend on each other.

**Parallel tool calling** solves this: in a *single* turn, the model emits *several* tool-call requests at once. You run them together (ideally concurrently), then return all the results in one go. One inference pass instead of three; three lookups overlapping instead of stacked end to end.

### How it looks

The shape is the same as Part 1 — there are just multiple calls in one response.

**Anthropic:** the response `content` contains multiple `tool_use` blocks. You collect them, run each, and return **all** the `tool_result` blocks inside a *single* `user` message:

```python
tool_blocks = [b for b in resp.content if b.type == "tool_use"]

results = []
for tb in tool_blocks:
    output = run_tool(tb.name, tb.input)      # your dispatcher
    results.append({
        "type": "tool_result",
        "tool_use_id": tb.id,                 # each result is matched by id
        "content": str(output),
    })

followup = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "Compare the weather in Paris, Tokyo, and New York."},
        {"role": "assistant", "content": resp.content},
        {"role": "user", "content": results},   # ALL results in ONE message
    ],
)
```

**OpenAI:** the assistant message's `tool_calls` array has multiple entries. You return one `role:"tool"` message per call, each tagged with its own `tool_call_id`.

> **Gotcha — results are matched by id, not by order.** You do not have to return results in the same order the model asked. Each result carries the `tool_use_id` / `tool_call_id` it answers, and the platform matches them up. What you *cannot* do is omit one: if the model asked for three calls, all three results must come back before it continues.

### Why it's faster (concretely)

Say each weather lookup takes 200 ms and each model inference takes 500 ms.

- **Sequential (no parallelism):** roughly `500 + 200 + 500 + 200 + 500 + 200` ≈ **2.1 s**, because each lookup waits for a fresh model turn.
- **Parallel:** one inference decides all three calls (`500 ms`), you run the three lookups *at the same time* (`~200 ms` total, not `600 ms`), then one final inference to summarize (`500 ms`) ≈ **1.2 s**.

The savings come from two places: fewer inference passes, and overlapping the I/O (input/output — waiting on the network/disk) of independent calls.

> **Gotcha — the model requesting parallel calls does NOT mean they run in parallel.** The provider hands you several call requests in one response; *running them concurrently is your job.* If your code just loops and awaits each one in turn, you've thrown away most of the benefit. Use threads, `asyncio`, or your language's concurrency primitive.

Minimal concurrent execution in Python:

```python
from concurrent.futures import ThreadPoolExecutor

def run_all(tool_blocks):
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(run_tool, tb.name, tb.input): tb for tb in tool_blocks}
        out = {}
        for fut in futures:
            tb = futures[fut]
            out[tb.id] = fut.result()   # collect by id
    return out
```

### When NOT to parallelize, and how to turn it off

Parallel calls only make sense when the calls are **independent**. If call B needs the *output* of call A (e.g. "look up the user's city, then get that city's weather"), they can't run at the same time — B doesn't know its own arguments until A returns. In that case the model should (and usually will) ask for them sequentially across turns instead.

You can also explicitly disable parallelism:

- **Anthropic:** set `tool_choice={"type": "auto", "disable_parallel_tool_use": True}`. Claude will then emit at most one tool call per turn.
- **OpenAI:** pass `parallel_tool_calls=False`.

Reasons you'd disable it: your tools have **side effects** whose ordering matters (e.g. "create record" then "update record" must not overlap); you want strict, auditable one-at-a-time execution; or you found the model guessing at parallel calls that turned out to be wrong and wasteful.

> **Gotcha — partial failure in a parallel batch.** If one of five concurrent calls throws, don't abandon the whole batch. Return a normal result for the four that succeeded and an *error* result (Part 4) for the one that failed. The model can then decide whether to proceed, retry, or tell the user — but only if you actually send back a result for every requested call.

### Hands-on exercise (Part 2)

1. Define three independent tools: `get_weather(city)`, `get_time(city)`, and `get_population(city)` — each returning a hard-coded string.
2. Send the message: `"For Paris, Tokyo, and New York, give me the weather, local time, and population."`
3. Confirm the model requests **multiple** tool calls in a single turn (print how many).
4. Execute them **concurrently** with a thread pool (or `asyncio`), collecting results by id.
5. Return all results in one message and print the final comparison.
6. **Stretch:** add crude timers. Compare total wall-clock time when you run the calls concurrently versus in a plain `for` loop with a `time.sleep(0.3)` inside each tool to simulate latency. Then set `disable_parallel_tool_use` / `parallel_tool_calls=False` and observe the model switch to one-at-a-time turns.

### Interview questions (Part 2)

1. **Conceptual — "What is parallel tool calling and what problem does it solve?"**
   It's when the model requests multiple tool calls in a single turn instead of one per turn. It reduces latency by cutting the number of inference round trips and by letting independent I/O-bound calls overlap.

2. **Practical — "The model asked for four tool calls in one turn but your app is barely faster than before. Why?"**
   You're almost certainly executing them sequentially. The provider only *requests* them together; running them concurrently (threads/async) is the application's responsibility, and without that the latency win largely disappears.

3. **Gotcha — "Do you have to return tool results in the same order the model requested them?"**
   No. Results are matched to requests by id (`tool_use_id`/`tool_call_id`), so order doesn't matter — but every requested call must get a result back before the model continues.

4. **Practical — "Give an example where you'd disable parallel tool use."**
   When calls have order-dependent side effects (create-then-update on the same record), when you need strictly auditable one-at-a-time execution, or when dependent calls make parallelism impossible anyway. You'd use `disable_parallel_tool_use` (Anthropic) or `parallel_tool_calls=False` (OpenAI).

5. **Gotcha — "One call in a parallel batch fails. What do you send back?"**
   A result for *every* requested call: successful outputs for the ones that worked and an error result (e.g. `is_error: true` for Anthropic, or error text in the tool message for OpenAI) for the one that failed. Never silently drop a requested call.

---

## Part 3 — Streaming with tools

### Why this exists (the problem)

By default, when you call the model you wait for the *entire* response, then get it all at once. For a long answer that can mean several seconds of a blank screen. **Streaming** sends the response to you incrementally, token by token, so you can display it as it's produced — the "typing" effect you see in chat apps. Beyond feel, it lets you start reacting sooner and cancel early.

Streaming plain text is straightforward. Streaming **tool calls** is where it gets subtle, because a tool call's arguments are JSON, and JSON that arrives in fragments is *not valid JSON until the last fragment lands.* Handling that correctly is the whole point of this part.

First, a definition. **SSE (Server-Sent Events)** is a simple standard for a server to push a stream of text events to a client over one long-lived HTTP connection. Each event has an event name and a chunk of data. Both providers stream tool calls over SSE; you enable it by setting `stream: true` (or using the SDK's streaming method).

### OpenAI streaming

With `stream=True`, you get a sequence of chunks, each carrying a small `delta` (a "delta" is just the *change* since the last chunk — the new fragment). For tool calls, the arguments arrive as **string fragments** spread across many deltas, and multiple parallel calls are distinguished by an **`index`**. Your job is to accumulate the fragments per index and only parse once the stream is done.

```python
import json

stream = client.chat.completions.create(
    model="gpt-5.5",
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Paris and Tokyo?"}],
    stream=True,
)

acc = {}  # index -> {id, name, args}
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc in delta.tool_calls:
            slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] += tc.function.name
            if tc.function and tc.function.arguments:
                slot["args"] += tc.function.arguments   # <-- string fragments, concatenated

# Only now, after the stream finishes, is each args string complete:
for slot in acc.values():
    parsed = json.loads(slot["args"])   # valid JSON only once fully accumulated
    run_tool(slot["name"], parsed)
```

> **Gotcha — never `json.loads` a partial arguments string mid-stream.** Halfway through, `slot["args"]` might be `{"city": "Par` — that's not parseable. Accumulate first, parse last. If you want to show *something* live, display a friendly "Calling get_weather…" indicator, not the raw half-built JSON.

> **Gotcha — accumulate by `index`, not by arrival order.** When several tool calls stream at once, their fragments are interleaved. The only reliable key is `tc.index`. Keying by anything else scrambles the arguments of different calls together.

### Anthropic streaming

Anthropic streams a well-defined sequence of named events. For any message you'll see: `message_start`, then for each content block a `content_block_start` → one or more `content_block_delta` → `content_block_stop`, then a `message_delta`, then `message_stop`. Text arrives as deltas of type `text_delta`. **Tool arguments arrive as deltas of type `input_json_delta`, whose `partial_json` field is a string fragment you accumulate** — exactly the same idea as OpenAI.

Raw event handling looks like this:

```python
with client.messages.stream(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Weather in Paris and Tokyo?"}],
) as stream:
    for event in stream:
        if event.type == "content_block_start" and event.content_block.type == "tool_use":
            # a new tool call is beginning: event.content_block.name / .id
            pass
        elif event.type == "content_block_delta" and event.delta.type == "input_json_delta":
            fragment = event.delta.partial_json   # string piece -> accumulate
            pass
        elif event.type == "content_block_delta" and event.delta.type == "text_delta":
            print(event.delta.text, end="", flush=True)   # live text to the user

    final = stream.get_final_message()   # SDK assembles the fully-parsed message for you
```

> **Gotcha — let the SDK accumulate when you can.** Both official SDKs will reassemble the streamed pieces into a normal, fully-parsed message object for you (Anthropic's `get_final_message()` / TypeScript `finalMessage()`; OpenAI's streaming helpers). Hand-accumulating fragments is error-prone; only do it when you specifically need each fragment as it arrives (e.g. to render progress).

> **Gotcha — a truncated stream can leave invalid JSON.** If generation stops early — most often because it hit `max_tokens` (the cap on output length) — the arguments string can end mid-value, e.g. `{"city": "Par`. There's also an opt-in "fine-grained tool streaming" beta that streams arguments without server-side JSON validation for lower latency, and it can hand you invalid/partial JSON by design. In both situations you must detect the incompleteness (catch the `json.loads` failure, check the stop reason) rather than assume every stream yields clean JSON.

> **Gotcha — the stream can end without its terminal event.** Networks drop. If your parser strictly requires the final `message_stop` (Anthropic) or the end-of-stream signal (OpenAI) and it never comes, you'll hang or throw. Build in a timeout and a retry path; the SDKs handle reconnection logic better than hand-rolled `fetch` loops.

### UX implications

- **Text**: stream it straight to the screen. This is the main perceived-speed win.
- **Tool arguments**: usually *don't* show the raw JSON as it builds. Show intent ("Searching the web…", "Looking up weather…") and reveal results after the tool runs.
- **When to actually run the tool**: only after that tool's arguments have fully streamed in (its block is `stop`-ped / the stream is complete for that call). Running on a partial argument string means running on invalid input.
- **Cancellation**: streaming lets a user hit "stop." Make sure stopping the stream also cancels any tool work you already kicked off, so you don't do (or bill for) work nobody's waiting on.

### Hands-on exercise (Part 3)

1. Take your `get_weather` tool from Part 1 and make a streaming request that will trigger it (e.g. `"What's the weather in Paris?"`).
2. Iterate over the stream. For each event/chunk, print a label showing what you got: text delta, tool-name fragment, or argument fragment.
3. Accumulate the argument fragments into a single string. Print that string *before* parsing it — see for yourself that mid-stream it is not valid JSON.
4. After the stream completes, `json.loads` the accumulated string, run the tool, and stream the model's final answer.
5. **Stretch:** deliberately set a very small `max_tokens` so generation is cut off mid-arguments. Confirm your `json.loads` raises, and handle it gracefully instead of crashing (this is your bridge into Part 4).

### Interview questions (Part 3)

1. **Conceptual — "What is SSE and what does streaming buy you?"**
   SSE (Server-Sent Events) is a standard for a server to push a stream of events to a client over one HTTP connection. Streaming lets you display the response as it's generated (better perceived latency), react sooner, and cancel early.

2. **Gotcha — "Why can't you parse tool arguments while they're streaming?"**
   Arguments arrive as string fragments (OpenAI `arguments` deltas, Anthropic `input_json_delta.partial_json`). A partial fragment like `{"city": "Par` is not valid JSON. You must accumulate all fragments and parse only once the call is complete.

3. **Practical — "Multiple tool calls are streaming at once. How do you keep their arguments from getting mixed up?"**
   Accumulate per call using the stable key the provider gives you — `index` in OpenAI's streamed `tool_calls`, or the content-block index in Anthropic. Fragments from different calls are interleaved, so keying by index (not arrival order) is essential.

4. **Gotcha — "You set a small `max_tokens` and now your streamed tool arguments won't parse. What happened?"**
   Generation was truncated by the token cap mid-arguments, leaving an incomplete JSON string. You need to detect the truncation (a failed parse and/or a `max_tokens` stop reason) and handle it rather than assume every stream produces valid JSON. Fine-grained tool streaming can similarly yield partial JSON.

5. **Practical — "How should tool calls appear in a streaming UI?"**
   Stream plain text directly to the user, but for tool calls show intent ("Searching…") rather than raw JSON, run the tool only after its arguments have fully arrived, and make sure canceling the stream also cancels in-flight tool work.

---

## Part 4 — Error handling: tool failures, retries, fallbacks, validation

### Why this exists (the problem)

Tools reach into the messy real world: networks time out, APIs return 500s, rate limits (caps on how often you may call a service) kick in, and inputs are sometimes wrong. On top of that, the *model* can produce bad arguments — a missing required field, the wrong type, or an outright hallucinated value (something it made up). A tutorial demo works because nothing goes wrong; a real agent survives because you planned for things going wrong. There are three distinct failure sources, and each has its own handling:

1. **Bad arguments from the model** → guard with **validation**.
2. **The tool itself failing** → decide between **retry**, **fallback**, and reporting the error back.
3. **The model looping** → cap iterations.

### 1. Validate arguments before you execute

Never pass model-generated arguments straight into a real operation (a database write, a shell command, an HTTP call). The model may omit a required field, send `"12"` where you expected the number `12`, or invent an ID. Validate first.

A clean way in Python is **Pydantic** (a popular library that validates data against a typed model and raises a clear error when it doesn't fit):

```python
from pydantic import BaseModel, ValidationError

class WeatherArgs(BaseModel):
    city: str

def safe_get_weather(raw_args: dict):
    try:
        args = WeatherArgs(**raw_args)   # raises if 'city' is missing or wrong type
    except ValidationError as e:
        return {"error": f"Invalid arguments: {e}"}   # -> becomes an error tool_result
    return get_weather(args.city)
```

> **Gotcha — "strict"/structured outputs reduce but don't eliminate the problem.** OpenAI's `strict: true` (structured outputs) makes generated arguments conform to your JSON Schema, which prevents *structural* errors like a missing required field. It does **not** guarantee the values make *sense* — the model can still supply a well-typed but wrong city, or an ID that doesn't exist. Schema-valid is not the same as correct, so keep your own semantic checks (does this ID exist? is this value in range?).

### 2. Return errors *to the model* (usually) instead of crashing

When a tool fails, you have two options: raise the error and stop the whole request, or **send the failure back to the model as a result** so it can adapt (retry differently, try another tool, or explain to the user). For interactive agents, sending it back is usually right — the model often recovers gracefully. This is exactly what the "error result" mechanism is for:

- **Anthropic:** return a `tool_result` block with `"is_error": True` and a short message describing what went wrong.
- **OpenAI:** there's no dedicated error flag — put a clear error description in the `content` of the `role:"tool"` message (Chat Completions) or the `function_call_output` `output` (Responses).

```python
# Anthropic error result
{
    "type": "tool_result",
    "tool_use_id": tb.id,
    "is_error": True,
    "content": "Weather service timed out after 5s. No data returned.",
}
```

> **Gotcha — don't dump raw stack traces back to the model.** A full traceback wastes tokens (which cost money and eat the context window — the limited amount of text the model can consider at once) and can leak internal details. Return a concise, *actionable* message: what failed and, if useful, what the model might do about it ("City not found; ask the user to clarify the spelling").

> **Gotcha — silent failure causes hallucination.** If a tool fails and you return an empty or misleadingly-successful result, the model will often *make up* a plausible answer to fill the gap. Failing loudly (an explicit error result) is safer than failing silently — the model needs to know it doesn't have real data.

### 3. Retries — but only the right kind

Some failures are **transient** (temporary): a network blip, a 503, a rate-limit 429. Retrying after a short wait often succeeds. Others are **permanent**: a 400 "bad request," "record not found," an authentication failure. Retrying those just wastes time — the same input will fail the same way.

The standard technique is **exponential backoff**: wait a little, then wait longer on each subsequent retry (e.g. 0.5s, 1s, 2s), often with a bit of randomness ("jitter") so many clients don't retry in lockstep.

```python
import time, random

def call_with_retry(fn, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except TransientError:                       # only retry transient failures
            if attempt == max_retries - 1:
                raise
            time.sleep((2 ** attempt) * 0.5 + random.random() * 0.1)  # backoff + jitter
        except PermanentError:
            raise                                    # don't retry; fail fast
```

> **Gotcha — never blindly retry a non-idempotent action.** "Idempotent" means doing it twice has the same effect as doing it once (e.g. reading data). A *non*-idempotent action changes state each time — "charge the card," "send the email," "create the order." If the first attempt actually succeeded but the *response* got lost, a blind retry double-charges or double-sends. For those, use an idempotency key (a unique token the server uses to recognize and de-duplicate a repeated request) or check state before retrying.

> **Gotcha — retry storms.** If a downstream service is down and every request retries aggressively, you can pile on and make the outage worse. Cap retries, use backoff with jitter, and consider a "circuit breaker" (after N consecutive failures, stop trying for a cool-off period and fail fast).

### 4. Fallbacks and loop caps

- **Fallback**: when the primary path keeps failing, degrade gracefully rather than dying. Return cached/stale data, a simpler default, or a clear "this feature is temporarily unavailable" that the model can relay — so the user gets *something* useful.
- **Loop cap**: the agentic loop from Part 1 can run away — the model calls a tool, gets an error, calls it again the same way, forever. **Always** cap the number of tool iterations per request (e.g. stop after 10) and return a sensible message when you hit the cap.

```python
MAX_STEPS = 10
for step in range(MAX_STEPS):
    resp = model_turn(...)
    if not wants_tool(resp):
        break
    run_tools_and_append_results(resp)
else:
    finalize("Stopped after too many tool steps without finishing.")
```

> **Gotcha — the model can loop even without errors.** Runaway loops aren't only caused by failures; a confused model may call the same successful tool repeatedly, or bounce between two tools. The loop cap protects you regardless of *why* the loop happens — treat it as a mandatory safety belt, not an optional optimization.

### Hands-on exercise (Part 4)

Harden a deliberately flaky tool.

1. Write `flaky_lookup(record_id)` that: raises a "not found" error for an unknown id (permanent), randomly raises a "timeout" ~50% of the time (transient), and otherwise returns data.
2. Add **validation**: reject a call with a missing or non-string `record_id` and turn it into an error result instead of executing.
3. Wrap execution in **retry-with-backoff** that retries the transient timeout up to 3 times but does *not* retry the permanent "not found."
4. On final failure, return an **error result** to the model (`is_error: true` for Anthropic, or error text in the tool/function-output message for OpenAI) with a concise, actionable message.
5. Run the whole thing inside a **loop cap** of 5 steps.
6. **Stretch:** make the model recover — prompt it so that on a "not found" error it asks the user to re-check the id rather than retrying the identical call. Watch how a good error message changes the model's next move.

### Interview questions (Part 4)

1. **Conceptual — "What are the main categories of failure in a tool-using agent, and how does handling differ?"**
   Bad arguments from the model (handle with validation before executing), tool execution failures (retry if transient, fall back if persistent, and report the error back to the model), and runaway loops (cap the number of iterations). Each category has a distinct mitigation.

2. **Practical — "A tool call fails. Do you raise the exception or return it to the model? Why?"**
   For interactive agents, usually return it as an error result (`is_error: true` for Anthropic; error text in the tool message for OpenAI) so the model can adapt — retry differently, choose another tool, or explain to the user. You raise/stop only when failure should abort the whole request.

3. **Gotcha — "You have `strict: true` structured outputs on. Do you still need to validate arguments?"**
   Yes. Strict mode enforces the JSON Schema (structure and types), so it prevents missing/mistyped fields, but it can't guarantee the *values* are semantically valid — the model can still supply a well-typed but nonexistent ID or an out-of-range value. Keep your own semantic checks.

4. **Gotcha — "Why is blindly retrying a failed tool call dangerous?"**
   Because non-idempotent actions (charge a card, send an email, create an order) change state each time. If the action actually succeeded but the response was lost, a blind retry duplicates the effect. Use idempotency keys or state checks, and only auto-retry transient failures — not permanent ones.

5. **Practical — "How do you stop an agent from looping forever on a tool?"**
   Cap the number of tool iterations per request and return a graceful message when the cap is hit. This is necessary because loops can occur without any error (a confused model repeating a successful call), so the cap must be unconditional rather than tied only to failures.

---

## Gotchas summary table

| # | Gotcha | Where it bites | Fix / rule of thumb |
|---|---|---|---|
| 1 | The model doesn't run your code — it only requests a call | Any tool flow | You must execute the function and return the result, or the model just waits |
| 2 | OpenAI `arguments` is a JSON *string*; Anthropic `input` is a parsed object | Reading a tool call | `json.loads` OpenAI's `arguments`; use Anthropic's `input` directly |
| 3 | `arguments` string can be malformed/invalid JSON | Parsing OpenAI/Anthropic tool args | Wrap parsing in `try/except`; never assume it parses |
| 4 | OpenAI's Chat Completions (nested `function`) vs Responses (flat) tool schemas aren't interchangeable | Switching OpenAI APIs | Match the shape to the API; flat → Responses, nested → Chat Completions |
| 5 | Anthropic requires the assistant `tool_use` turn before the `tool_result`, immediately adjacent | Building the follow-up request | Echo the assistant message, then send `tool_result` right after |
| 6 | A "wants a tool" response can also contain text | Handling responses | Process both text and tool blocks/parts, not one or the other |
| 7 | Requesting parallel calls ≠ running them in parallel | Parallel tool use | You must run them concurrently (threads/async) to get the speed-up |
| 8 | You must return a result for *every* requested call | Parallel tool use | Match by id; never drop a call; return error results for failures |
| 9 | Never `json.loads` a partial arguments string mid-stream | Streaming tools | Accumulate all fragments, then parse once complete |
| 10 | Streamed calls are interleaved | Streaming multiple tools | Accumulate per `index` (OpenAI) / block index (Anthropic), not by arrival order |
| 11 | Truncation (`max_tokens`) / fine-grained streaming can yield invalid JSON | Streaming tools | Detect incompleteness (failed parse / stop reason); handle, don't assume |
| 12 | Stream can end without its terminal event | Streaming | Add timeouts + retry; prefer SDK accumulation over hand-rolled fetch |
| 13 | `strict`/structured outputs ≠ semantically correct values | Validation | Keep your own semantic checks (existence, ranges) |
| 14 | Dumping raw stack traces back to the model | Error handling | Return concise, actionable error messages |
| 15 | Silent failure → the model hallucinates an answer | Error handling | Fail loudly with an explicit error result |
| 16 | Blindly retrying non-idempotent actions double-executes them | Retries | Only auto-retry transient failures; use idempotency keys for side effects |
| 17 | Agentic loops can run away (even without errors) | Any agentic loop | Always cap iterations per request |
| 18 | "OpenAI-compatible" provider ≠ identical tool-calling behavior | Cross-provider | Test each provider; use a normalization layer to mix them |

---

## Quick reference card

**The universal loop:** send message + tools → model requests a call → you run it → you return the result → model continues (repeat, with a cap).

**Tool definition shapes**

```text
Anthropic (Messages):
  { "name": ..., "description": ..., "input_schema": { JSON Schema } }

OpenAI (Chat Completions):
  { "type": "function",
    "function": { "name": ..., "description": ..., "parameters": { JSON Schema } } }

OpenAI (Responses):
  { "type": "function",
    "name": ..., "description": ..., "parameters": { JSON Schema } }   # flat

Google (Gemini):
  { "function_declarations": [ { "name": ..., "description": ..., "parameters": { JSON Schema } } ] }

Most other providers (Mistral, Llama, DeepSeek, Grok, Qwen, Cohere-compatible, Ollama/vLLM) reuse the
OpenAI shape. To mix providers, use a normalization layer (Bedrock Converse, LangChain bind_tools,
LiteLLM) instead of hand-writing a schema + parser per provider.
```

**Reading the call**

| | signal | id | arguments |
|---|---|---|---|
| Anthropic | `stop_reason == "tool_use"` | `block.id` | `block.input` (dict) |
| OpenAI Chat | `finish_reason == "tool_calls"` | `call.id` | `call.function.arguments` (str → parse) |
| OpenAI Responses | item `type == "function_call"` | `item.call_id` | `item.arguments` (str → parse) |
| Google Gemini | response has a `functionCall` part | n/a (matched by name) | `functionCall.args` (dict) |

**Returning the result**

- Anthropic: `user` message → `{"type":"tool_result","tool_use_id":ID,"content":...,"is_error":?}`
- OpenAI Chat: `{"role":"tool","tool_call_id":ID,"content":...}`
- OpenAI Responses: `{"type":"function_call_output","call_id":ID,"output":...}`
- Google Gemini: a `functionResponse` part → `{"functionResponse":{"name":NAME,"response":{...}}}`

**Parallel**

- Multiple calls in one turn; run concurrently; return *all* results (matched by id).
- Disable: Anthropic `tool_choice={"type":"auto","disable_parallel_tool_use":true}`; OpenAI `parallel_tool_calls=false`.

**Streaming**

- Enable with `stream=true` (SSE). Text arrives as `text_delta` / text deltas.
- Tool args arrive as fragments: Anthropic `input_json_delta.partial_json`; OpenAI `delta.tool_calls[].function.arguments`. **Accumulate, then parse.**
- Anthropic event order: `message_start` → per block (`content_block_start` → `content_block_delta`… → `content_block_stop`) → `message_delta` → `message_stop`.
- Prefer SDK accumulation helpers (`get_final_message()` / streaming helpers).

**Error handling checklist**

1. Validate args before executing (Pydantic / JSON Schema).
2. Return failures to the model as error results (`is_error:true` / error text).
3. Retry only transient failures with exponential backoff + jitter; never blind-retry non-idempotent actions.
4. Provide fallbacks (cached/default/graceful message).
5. Cap loop iterations per request.

**Reference material:** OpenAI function-calling, tools, and streaming guides; Anthropic tool-use overview, parallel tool use, and Messages streaming docs; tool-use best-practices posts.

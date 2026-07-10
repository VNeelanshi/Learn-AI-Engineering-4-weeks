# Multi-Agent Systems

The four parts: **when multi-agent is worth it** (an assessment), then three ways to build one: **CrewAI**, **AutoGen**, and **supervisor patterns in LangGraph**.

> **Note on version names and churn.** Model names in examples (`claude-sonnet-4-6`, `gpt-5.5`) are current examples and change often. The frameworks here move fast and rename things; where a framework's status or successor matters, this tutorial says so, and teaches the current shape while flagging what's shifting.

---

## Part 1: When (and when not) to use multi-agent: an assessment

### Why this part comes first (the problem)

Multi-agent systems are exciting and, in most real projects, **overkill**. The single biggest architectural mistake teams make is reaching for multiple agents because it *sounds* more capable, when a much simpler design would have been cheaper, faster, and easier to debug. So before any framework, the question: does your task actually need this?

The useful framing comes from a distinction between **workflows** and **agents**:

- A **workflow** is a system where LLM calls and tools are orchestrated through **predefined code paths**: you decide the steps in advance and the LLM fills in the intelligent parts.
- An **agent** is a system where the **LLM dynamically directs its own process** and tool usage, deciding the path as it goes.

The difference is *degree of autonomy*, and it's a gradient. The governing principle (from Anthropic's widely-cited guidance on building effective agents): **find the simplest solution that works, and only add complexity when it demonstrably improves outcomes.** Complexity you can't justify with an evaluation is complexity you shouldn't ship.

### The complexity ladder

Think of designs as rungs on a ladder. Climb only when the rung below provably fails your task:

1. **A single LLM call.** One prompt, one response. Surprisingly often enough.
2. **An augmented LLM call**: one LLM with tools, retrieval, and memory. Handles most "look something up and answer" jobs.
3. **A workflow**: several LLM calls on fixed paths. The common patterns:
   - *Prompt chaining*: each call processes the previous one's output (step 1 → step 2 → step 3).
   - *Routing*: a classifier LLM sends the input to a specialized follow-up.
   - *Parallelization*: run calls at the same time: either *voting* (same task several times, aggregate) or *sectioning* (split into independent subtasks).
   - *Orchestrator-workers*: a central LLM breaks a task into subtasks and delegates them (covered in depth in Part 4).
   - *Evaluator-optimizer*: one LLM generates, another critiques, in a loop.
4. **A single agent**: one LLM looping with tools, directing itself.
5. **Multiple agents**: several coordinating agents. The top rung.

Notice that rungs 3–5 blur: a "routing workflow" and a "supervisor of agents" are close cousins. The point isn't rigid categories; it's *start low, climb only when forced*.

### Why multi-agent is often overkill

Adding agents adds real costs, all at once:

- **Token cost.** Agents pass context and messages around, often re-sending history. Anthropic reported that their multi-agent research system used roughly **15× the tokens** of a single-agent chat. That's a real bill.
- **Latency.** Every agent turn is another LLM inference; agents talking to each other multiply the round trips.
- **Coordination overhead and "disorganized emergence."** Put several agents in a room and they may not know what to do: looping, going off-topic, or passing work back and forth without progress.
- **Harder debugging.** A wrong answer at the end might trace to a bad routing decision three agents ago. Without per-agent tracing, it's nearly impossible to see why.
- **Error compounding.** Small mistakes in early agents get amplified downstream.

### When it *is* worth it

Reach for multi-agent when the task has one or more of these shapes, **and** a well-tooled single agent has demonstrably failed on your evaluations:

- **Parallelizable, breadth-first work.** Tasks like open-ended research that benefit from many subagents exploring in parallel (this is exactly what Anthropic's research system does: and even there it costs ~15× the tokens, so the breadth has to be worth it).
- **Clean decomposition with verifiable sub-results.** The task splits into subtasks whose outputs you can check independently.
- **Different tools, prompts, permissions, or models per subtask.** Isolating a "security" concern from a "support" concern, or running a cheap model for grunt work and a strong one for reasoning.
- **A conversation/debate shape.** When perspectives must interact: a bull-case and bear-case arguing before a synthesis: the back-and-forth *is* the value.

When an agent *is* justified, it boils down to three decisions; everything else is optimization:

- **Environment**: the world the agent perceives (a repo, a browser, a ticket queue).
- **Tools**: well-documented, hard-to-misuse interfaces. Spend more effort here than on clever prompts.
- **System prompt**: goals, constraints, and clear stop rules.

A single agent with good tools is the right answer for roughly the large majority of cases. Multi-agent's apparent sophistication is a lure. Before adding a second agent, confirm your single agent is fully optimized (right tools, right instructions) *and* failing on evals. If you can't articulate the eval it will pass that the single agent fails, you're adding complexity on vibes.

> **Gotcha: frameworks can hide the ball.** Multi-agent frameworks add abstraction layers that can obscure the actual prompts and responses, and tempt you to add complexity a simpler setup would handle. Frameworks are fine: they save time and encode best practices: but understand what they're doing underneath so you can debug and simplify. Don't let the framework make architectural decisions for you.

### Hands-on exercise (Part 1)

No code: a design decision, which is the real skill here.

1. Pick a task you actually want to build (e.g. "answer support tickets," "write a weekly market brief," "review pull requests").
2. Walk the ladder from rung 1. For each rung, write one sentence on whether it could plausibly do the job, and what would make it fail.
3. Identify the *lowest* rung that would work. Most tasks stop at rung 2 or 3.
4. If you landed on rung 5 (multi-agent), write down: which of the four "worth it" shapes it matches, what the verifiable sub-results are, and a rough token-cost multiplier vs a single agent.
5. **Stretch:** write the one-line evaluation a single agent would *fail* that justifies going multi-agent. If you can't, drop back a rung.

### Interview questions (Part 1)

1. **Conceptual: "What's the difference between a workflow and an agent?"**
   A workflow orchestrates LLM calls and tools through predefined code paths that you design in advance; an agent lets the LLM dynamically direct its own process and tool use. The difference is the degree of autonomy, and it's a gradient rather than a hard boundary.

2. **Practical: "A team wants to build a multi-agent system for a new feature. What do you ask first?"**
   Whether a simpler design would work: a single well-tooled agent, or even a workflow or one augmented LLM call. Specifically, has a single agent been fully optimized (tools, prompts) and shown to fail on evals? Multi-agent should be justified by an evaluation it passes that simpler designs don't.

3. **Gotcha: "Why is multi-agent often the wrong default?"**
   It multiplies token cost, adds latency from extra inference round trips, introduces coordination failures and error compounding, and is much harder to debug. A single agent with good tools handles the large majority of tasks.

4. **Conceptual: "When is multi-agent actually the right choice?"**
   For parallelizable breadth-first work (e.g. open-ended research), tasks that decompose cleanly into subtasks with verifiable results, cases needing different tools/prompts/permissions/models per subtask, or conversation/debate shapes where interacting perspectives are the point: and only after a single agent demonstrably falls short.

5. **Practical: "When you do build an agent, where should you spend your design effort?"**
   On the three core decisions: environment, tools, and system prompt: and especially on tools: well-documented, hard-to-misuse interfaces matter more than clever prompting. Everything else (caching, parallel calls, UI polish) is optimization after the behavior works.

---

## Part 2: CrewAI: roles, tasks, processes, hierarchical orchestration

### Why this framework exists (the problem)

Some tasks decompose naturally the way *human teams* do: a researcher gathers facts, a writer drafts, an editor polishes. **CrewAI** is a Python framework built around exactly that metaphor. Instead of thinking about graphs or message loops, you act like a manager: define "employees" (agents) with roles and skills, hand them "tasks" with clear deliverables, assemble them into a "crew," and press go. The role-and-task structure is opinionated on purpose: giving each agent a clear role and goal sharply reduces the "agents wander off and loop" problem.

Install with `pip install crewai` (add `crewai[tools]` for the prebuilt tool library; Python 3.10–3.13).

### The core concepts

- **Agent**: a worker defined by a `role`, a `goal`, a `backstory` (a short persona that shapes tone and boundaries), its `tools`, and an LLM. The role/goal/backstory aren't decoration: they anchor the agent's behavior and cut hallucination.
- **Task**: a unit of work with a `description` (what to do), an `expected_output` (what the deliverable should look like), and an assigned `agent`. Optionally a `context` (other tasks whose output feeds this one) and `output_pydantic` (a typed schema for structured output).
- **Crew**: the container holding `agents`, `tasks`, and a `process`. You run it with `.kickoff()`.
- **Process**: the execution strategy: `Process.sequential` (tasks run in the order listed) or `Process.hierarchical` (a manager agent dynamically delegates tasks to workers). (A "consensual" process appears in the docs as planned but isn't implemented.)
- **Tool**: a callable an agent can invoke. CrewAI has its own `@tool` decorator and a `crewai_tools` library of prebuilt tools (search, file IO, databases, browsers).

CrewAI also offers a second paradigm, **Flows**: event-driven workflows built with `@start`, `@listen`, and `@router` decorators that give precise control and can mix plain Python with crew/agent calls. The rule of thumb: use a **Crew** when the shape is role-based collaboration; reach for a **Flow** when orchestration gets more complex than a simple sequential or hierarchical run (branching, conditional routing, mixing deterministic steps with agent steps). This part focuses on Crews.

### Sequential process (the most common shape)

Tasks run in order, and you connect them with `context` so each task receives the previous one's output:

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Researcher",
    goal="Find accurate, current facts about {topic}",
    backstory="A meticulous analyst who always cites sources.",
    tools=[search_tool],                 # tools attached at the AGENT level (scoped to this role)
    llm="anthropic/claude-sonnet-4-6",
    allow_delegation=False,
)
writer = Agent(
    role="Writer",
    goal="Turn research into a clear three-paragraph brief",
    backstory="A crisp explainer who hates fluff.",
    allow_delegation=False,
)

research_task = Task(
    description="Research {topic} and list the key facts.",
    expected_output="A bulleted list of 5–8 verified facts, each with a source.",
    agent=researcher,
)
write_task = Task(
    description="Write a three-paragraph brief on {topic}.",
    expected_output="Exactly three tight paragraphs, no fluff.",
    agent=writer,
    context=[research_task],             # <-- feeds research_task's OUTPUT into this task
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)
result = crew.kickoff(inputs={"topic": "AI coding assistants in 2026"})
```

> **Gotcha: `expected_output` is part of the prompt, not a comment.** If you skip it or leave it vague, the task runs with under-specified instructions and the agent's output drifts. Describe the deliverable concretely ("three paragraphs," "a JSON list of objects with fields x, y"). It directly shapes what the agent produces.

> **Gotcha: forgetting `context` makes agents work in isolation.** Without linking the writer's task to the researcher's via `context`, the writer never sees the research and produces something disconnected. The `context` list is how output flows from one task to the next; it's the most common thing new users forget.

> **Gotcha: attach tools at the agent level, not just the crew.** Giving every agent every tool (crew-level tools) leads to over-broad, noisy tool usage. Scope each tool to the agent whose role needs it.

### Hierarchical process (a manager delegates)

Switch the process and provide a `manager_llm`, and CrewAI automatically creates a **manager agent** that decides which worker handles each subtask, and validates results: reassigning work if an output is unsatisfactory. This suits tasks where the decomposition is *dynamic* (you don't know the exact order in advance).

```python
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.hierarchical,
    manager_llm="anthropic/claude-sonnet-4-6",   # REQUIRED for hierarchical; the manager's brain
)
```

> **Gotcha: don't reach for hierarchical when sequential works.** The manager adds an extra LLM call per dispatch (cost + latency) and more moving parts to debug. If the task order is known, sequential is cheaper, faster, and easier to reason about. Use hierarchical only when the routing needs to be decided at runtime.

### Keeping crews from misbehaving

- **`max_iter`** (default 15) caps how many reasoning iterations an agent takes: a guard against loops.
- **`allow_delegation=False`** stops agents from endlessly reassigning work to each other.
- **`memory=True`** gives shared memory across the run; it's **opt-in**, so without it every kickoff starts cold (fine for stateless jobs, costly for conversational ones).
- **`telemetry=False`** disables usage telemetry, commonly set in production.

> **Gotcha: memory is off by default.** If you expect your crew to remember earlier runs or conversation and it doesn't, you likely didn't enable memory. Turn it on deliberately where you need continuity.

### Hands-on exercise (Part 2)

1. Build the two-agent sequential crew above: a `Researcher` with a (dummy) search tool and a `Writer`. Give both a concrete `expected_output` and link the write task to the research task with `context`.
2. `kickoff` it with a topic and read the result. Then **delete** the `context` link and re-run: observe the writer ignoring the research (the isolation gotcha).
3. Flip to `Process.hierarchical` with a `manager_llm` and run again. Note the extra manager step in the output and think about whether it earned its cost here.
4. **Stretch:** add a third `Editor` agent and task (context = both prior tasks), and set `max_iter` low on one agent to see the iteration cap engage.

### Interview questions (Part 2)

1. **Conceptual: "Explain CrewAI's core concepts."**
   An Agent is a worker with a role, goal, backstory, tools, and LLM. A Task is a unit of work with a description, an expected_output, and an assigned agent (optionally a context and a typed output). A Crew groups agents and tasks with a process and runs via `kickoff()`. The process is `sequential` (tasks in order) or `hierarchical` (a manager delegates dynamically).

2. **Practical: "When would you use a sequential process vs a hierarchical one?"**
   Sequential when the task order is known and fixed (research → write → edit): it's cheaper, faster, and easier to debug. Hierarchical when the decomposition is dynamic and you want a manager agent to decide who does what and validate results; the trade-off is an extra LLM call per dispatch.

3. **Gotcha: "Your writer agent produces content that ignores the researcher's findings. What did you forget?"**
   The `context` link on the writing task. Without `context=[research_task]`, the tasks run in isolation and outputs don't flow between agents. Setting the context passes the research output into the writing task.

4. **Gotcha: "You left `expected_output` blank. What happens?"**
   The task is under-specified because `expected_output` is part of the prompt that guides the agent. The output drifts or is inconsistent. You should describe the deliverable concretely so the agent knows the target shape.

5. **Practical: "How do you stop CrewAI agents from looping or endlessly delegating?"**
   Cap reasoning with `max_iter` (default 15), disable inter-agent handoffs with `allow_delegation=False` where you don't want them, and set a `max_execution_time`. These bound runaway loops and back-and-forth delegation.

---

## Part 3: AutoGen: conversable agents, group chat patterns

### Status note first (read this)

**AutoGen** is Microsoft's open-source Python framework for building multi-agent applications shaped as **conversations**. Important as of early 2026: Microsoft moved AutoGen into **maintenance mode** and points new ("greenfield") projects to its successor, the **Microsoft Agent Framework** (which merges ideas from AutoGen and Microsoft's Semantic Kernel). AutoGen still works, is still widely deployed, and its concepts transfer directly: so it's worth learning: but if you're starting fresh in the Microsoft/Azure ecosystem, check the Agent Framework too. This part teaches current AutoGen (the post-0.4 "AgentChat" design) and flags where it's headed.

Install: `pip install autogen-agentchat autogen-ext[openai]` (plus `autogen-core` underneath).

### Why this framework exists (the problem)

Some problems aren't a pipeline of tasks: they're a **conversation**. Picture a market-research assistant built as a debate: a bull-case agent argues one side, a bear-case agent argues the other, and a synthesis agent reads both and writes a balanced summary. There's no fixed task order; the agents need to talk back and forth until synthesis decides they're done. AutoGen is built precisely for this "specialized agents in a conversation" shape.

### The layered architecture

- **autogen-core**: the low-level foundation: an *actor model* where agents exchange messages **asynchronously** (event-driven). Maximum flexibility; more code. Most people don't start here.
- **autogen-agentchat**: the high-level, opinionated API for rapid prototyping, where most application code lives. Supplies ready-made agents and team (group-chat) patterns.
- **autogen-ext**: integrations: model clients (`OpenAIChatCompletionClient`, `AzureOpenAIChatCompletionClient`, `AnthropicChatCompletionClient`, `OllamaChatCompletionClient`), code executors, tool wrappers, and MCP connectors.

### The agents

- **ConversableAgent**: the base idea: an agent that converses by exchanging messages.
- **AssistantAgent**: an LLM-backed agent. You give it a `name`, a `model_client`, a `system_message`, optional `tools`, and a `description` (used by group chats to decide when it should speak).
- **UserProxyAgent**: stands in for a human (for approvals/input) and can execute code on the human's behalf.

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient(model="gpt-5.5")
    agent = AssistantAgent("assistant", model_client=client,
                           system_message="You are a helpful assistant.")
    print(await agent.run(task="Say hello."))

asyncio.run(main())
```

> **Gotcha: AutoGen is async-first.** Almost everything is `async`/`await`, run inside `asyncio.run(...)`. If you're used to plain synchronous Python, this is the first thing that trips you up: forgetting `await` gives you a coroutine object instead of a result.

> **Gotcha: old (v0.2) tutorials look different.** A lot of material online uses the older API: `from autogen import AssistantAgent` with an `llm_config={...}` argument. The current design uses `from autogen_agentchat.agents import AssistantAgent` with a `model_client=`. If imports or arguments don't match, you're likely mixing two major versions.

### Group chat: teams of conversing agents

A **Team** coordinates several agents in a conversation. The main built-in patterns:

- **`RoundRobinGroupChat([a, b, ...])`**: agents take turns in a **fixed order**, each broadcasting to all so everyone shares the same context. The classic use is the *reflection* pattern: a primary agent produces something and a critic agent reviews it, turn after turn.
- **`SelectorGroupChat([...], model_client=...)`**: a model **chooses the next speaker** each turn based on the agents' `description`s. Good for dynamic coordination, e.g. a planning agent breaks a task into subtasks and the selector routes each to the right specialist.
- **`MagenticOneGroupChat`**: a generalist orchestrator team for open-ended web/file tasks.
- **`Swarm`**: agents hand off to each other via handoff messages.

You run a team with `team.run(task=...)` (or `run_stream(...)` to stream). Teams are **stateful**: they keep the conversation after a run, so calling run again (without a new task) continues; `reset()` clears it.

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient(model="gpt-5.5")
    writer = AssistantAgent("writer", model_client=client,
                            system_message="Write a short poem on the given topic.")
    critic = AssistantAgent("critic", model_client=client,
                            system_message="Critique the poem. Reply with 'APPROVE' when it is good enough.")

    termination = TextMentionTermination("APPROVE")     # stop when someone says APPROVE
    team = RoundRobinGroupChat([writer, critic], termination_condition=termination)
    await Console(team.run_stream(task="Write a poem about autumn."))

asyncio.run(main())
```

### Termination: the thing you must not forget

A group chat has no natural end: the agents will keep talking. You control stopping with a **termination condition**: `MaxMessageTermination(n)` (stop after n messages), `TextMentionTermination("FINAL ANSWER")` (stop when a phrase appears), and others, which you can combine with And/Or logic.

> **Gotcha: no termination condition means an endless (and expensive) chat.** This is the defining AutoGen footgun. Always attach a termination condition to a team, and prefer a belt-and-suspenders combination (e.g. stop on the phrase "APPROVE" *or* after 10 messages) so a model that never says the magic word can't run forever.

> **Gotcha: `AssistantAgent` takes one step per turn by default.** Inside a team turn, an `AssistantAgent` does a single step (and, by default, a limited number of tool iterations). If you need it to keep calling tools until finished, you configure `max_tool_iterations` higher or let the team loop drive it: don't assume one turn means "do everything."

> **Gotcha: tool output isn't always natural language.** By default an agent returns a tool's raw output as its message. If that output isn't a clean natural-language string, set `reflect_on_tool_use=True` so the agent phrases a proper response around the tool result.

### Hands-on exercise (Part 3)

1. Build the two-agent `RoundRobinGroupChat` above (writer + critic) with a `TextMentionTermination("APPROVE")` and a `MaxMessageTermination` as a backstop. Run it and watch the two agents iterate until the critic approves.
2. Remove the termination condition and (carefully, with a low max) observe how the conversation wants to keep going: the endless-chat gotcha.
3. Swap to a `SelectorGroupChat` with three agents (a planner and two specialists, each with a clear `description`) and confirm the selector routes turns based on those descriptions.
4. **Stretch:** give one specialist a real tool and set `reflect_on_tool_use=True`; compare its messages with and without reflection.

### Interview questions (Part 3)

1. **Conceptual: "What shape of problem is AutoGen designed for, and what are its two main agent types?"**
   Problems shaped as a conversation between specialized agents (debate, reflection, brainstorming) rather than a fixed pipeline. Its main agent types are `AssistantAgent` (an LLM-backed agent) and `UserProxyAgent` (a stand-in for a human that can also execute code), both descending from the conversable-agent idea.

2. **Practical: "When would you use RoundRobinGroupChat vs SelectorGroupChat?"**
   RoundRobin when a fixed turn order fits (e.g. writer-then-critic reflection), since it's simple and predictable. Selector when the next speaker should be chosen dynamically by a model based on agent descriptions: useful when a planner delegates varied subtasks to different specialists.

3. **Gotcha: "Your AutoGen group chat never stops. What's missing?"**
   A termination condition. Group chats don't end on their own; you must attach something like `TextMentionTermination` or `MaxMessageTermination` (ideally combined) so the team halts when a phrase appears or a message cap is reached.

4. **Conceptual: "AutoGen has three package layers. What are they for?"**
   `autogen-core` is the low-level actor model for asynchronous message passing (flexible, more code); `autogen-agentchat` is the high-level opinionated API where most application code lives; `autogen-ext` provides integrations like model clients, code executors, and tool wrappers.

5. **Gotcha / awareness: "Any concern about starting a brand-new project on AutoGen today?"**
   Yes: as of early 2026 AutoGen is in maintenance mode, with Microsoft directing greenfield builds to the Microsoft Agent Framework (its successor). AutoGen still works and its concepts carry over, but for a new build in the Microsoft/Azure ecosystem you'd evaluate the successor rather than assume AutoGen is the long-term choice.

---

## Part 4: Supervisor patterns in LangGraph: plan-execute, orchestrator-worker

### Recap: what LangGraph is

**LangGraph** is a Python library for building LLM applications as a **graph**: **nodes** are steps (functions, or whole agents), **edges** route between them, and a shared **state** object flows through. In a multi-agent LangGraph app, each agent is a node, and a special return value: **`Command`**: moves control *and* updates state in one object: `Command(goto="next_node", update={...})`. This part uses that to build orchestration where one coordinator directs several specialists.

### Why supervisor patterns (the problem)

Say you have several specialist agents: a math expert, a research expert, a writer. You want *one* orchestrator deciding who goes next, keeping shared state and a clear stopping point. That's the **supervisor pattern**. It sits between two extremes: the **network** pattern, where every agent can call every other agent (chaos, and hard to reason about), and a **hierarchy of supervisors** (overkill for most teams). The supervisor is the practical sweet spot for a handful of specialists.

### The supervisor pattern

A supervisor LLM node looks at the conversation and decides which specialist should act next; each specialist does its bit and returns control to the supervisor; the loop ends when the supervisor decides the request is fully handled. There's a prebuilt library, `langgraph-supervisor`, that wires this for you:

```python
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-6")

# Two specialist agents (each is a small ReAct agent with its own tools)
math_agent = create_react_agent(model, tools=[add, multiply], name="math_expert",
                                prompt="You are a math expert. Do calculations.")
research_agent = create_react_agent(model, tools=[web_search], name="research_expert",
                                    prompt="You are a research expert. Find facts.")

supervisor = create_supervisor(
    [math_agent, research_agent],
    model=model,
    prompt=("You manage a math expert and a research expert. "
            "Delegate each part of the request to the right one. "
            "When the request is fully answered, produce the final answer and stop."),
).compile()

result = supervisor.invoke(
    {"messages": [{"role": "user", "content": "What is 12.5% of France's population?"}]}
)
```

Under the hood, the supervisor delegates using **handoff tools**: the supervisor "calls" a tool that transfers control (and, by default, the message history) to a specialist. You can customize what context is handed off. The supervisor **terminates** when it responds without delegating to another worker (the library treats that as "finish"); you reinforce this in the supervisor's prompt ("when done, give the final answer and stop").

If you'd rather build it by hand, a supervisor node uses **structured output** to pick the next worker and routes with `Command`:

```python
from typing import Literal
from langgraph.graph import END
from langgraph.types import Command

def supervisor_node(state) -> Command[Literal["math_expert", "research_expert", "__end__"]]:
    decision = model.with_structured_output(Router).invoke(state["messages"])  # Router: {next, reasoning}
    goto = END if decision["next"] == "FINISH" else decision["next"]
    return Command(goto=goto, update={"next": decision["next"]})
```

> **Gotcha: keep specialist tools *off* the supervisor.** The supervisor's only job is to route (or finish). If you give it the specialists' tools, it'll start doing the work itself, defeating the pattern and muddying traces. Give the supervisor only "route to X" / "finish" options, tell it so in the prompt, and add an evaluation that checks routing accuracy.

> **Gotcha: the recursion limit is your loop alarm.** Each supervisor→worker→supervisor cycle is two graph steps; a realistic four-specialist task takes ~6–10. LangGraph caps steps per run with a `recursion_limit` (default **25**) and raises an error past it. Hitting it is almost always a **routing loop**: the supervisor keeps picking a specialist that defers right back. The fix is the supervisor's prompt, not a bigger limit; routine limit hits mean a prompt bug.

### Orchestrator-worker

A close relative: an **orchestrator** LLM *dynamically* breaks a task into subtasks and dispatches them to **workers**, then synthesizes their outputs. LangGraph's **`Send`** API lets the orchestrator spawn workers on the fly: one per subtask: often running them in parallel; every worker writes to a **shared state key with an accumulating reducer**, so the orchestrator can gather all outputs and combine them.

```python
from langgraph.types import Send

def orchestrator(state):
    plan = planner.invoke(state["topic"])          # decide the sections dynamically
    return {"sections": plan.sections}

def assign_workers(state):
    # Fan out: one worker per section, each with its own input
    return [Send("worker", {"section": s}) for s in state["sections"]]

def worker(state):
    text = write_section(state["section"])
    return {"completed_sections": [text]}          # shared key uses operator.add to accumulate

builder.add_conditional_edges("orchestrator", assign_workers, ["worker"])
builder.add_edge("worker", "synthesizer")          # synthesizer reads all completed_sections
```

Use orchestrator-worker when the subtasks **can't be known in advance**: e.g. writing a report with a variable number of sections, or editing an unknown number of files. (When subtasks *are* fixed and independent, plain parallelization is simpler.)

### Plan-execute

**Plan-execute** makes the planning explicit: a *planner* produces a multi-step plan up front, an *executor* carries out the steps one by one, and control loops back to *replan* (adjust the remaining plan based on what happened) until the goal is met. It's a specific orchestration shape: useful when a task benefits from committing to a plan and revising it: and in LangGraph it's built from the same pieces: nodes for plan/execute/replan, and `Command`/conditional edges to loop until done.

### Handoffs and the swarm alternative

Control passes between agents via `Command(goto=..., update={...})`. When an agent is itself a subgraph and needs to jump to a *sibling* agent in the parent graph, it returns `Command(goto="other_agent", graph=Command.PARENT)`. The **swarm** pattern removes the supervisor entirely: each agent has handoff tools and transfers directly to another agent. Swarm is faster (no middle-man) but tends to misroute more; start with a supervisor (easier to build, debug, and keep accurate) and move to swarm only when data shows latency is the bottleneck and misrouting is rare.

> **Gotcha: lost messages during a handoff.** When one agent hands off (especially in a swarm, across a subgraph boundary), the specialist's internal messages may not propagate to the parent graph, so the next agent sees a broken conversation. Make sure the `Command`'s `update` includes the relevant messages, and keep tool-call/tool-result messages paired: an unpaired tool call in the history will make the next agent error or hallucinate.

> **Gotcha: you can't debug multi-agent without per-agent tracing.** With several agents handing off, a plain log is unreadable. Trace each node/agent as its own span (e.g. with LangSmith) so you can see who decided what. Budget for observability from day one.

### Hands-on exercise (Part 4)

1. Build a supervisor over two specialists: a "math" agent (with `add`/`multiply` tools) and a "research" agent (with a dummy `web_search`): using `create_supervisor(...)`. Ask a question that needs both (e.g. "find X, then compute Y% of it") and read the trace of who was delegated to.
2. Deliberately give the supervisor one of the specialists' tools and a weak prompt; watch it try to do the work itself. Remove the tool and tighten the prompt to fix it.
3. Build a tiny **orchestrator-worker** with `Send`: an orchestrator that splits a "report" into 3 sections, a worker that writes each (in parallel), and a synthesizer that concatenates them from the shared reducer key.
4. **Stretch:** force a routing loop (a supervisor prompt that keeps bouncing between specialists), watch it hit the `recursion_limit`, then fix the prompt rather than raising the limit.

### Interview questions (Part 4)

1. **Conceptual: "What is the supervisor pattern and where does it sit among multi-agent architectures?"**
   A central supervisor LLM decides which specialist agent runs next, with specialists returning control to it and shared state throughout, ending when the supervisor finishes. It sits between the network pattern (every agent can call every other: chaotic) and hierarchies of supervisors (overkill), and is the practical choice for coordinating a handful of specialists.

2. **Practical: "How does the supervisor delegate, and how does the run terminate?"**
   It delegates via handoff tools that transfer control (and message context) to a specialist. It terminates when it responds without delegating to another worker: which you reinforce in its prompt ("when the request is fully answered, give the final answer and stop"), optionally bounded by the recursion limit.

3. **Gotcha: "Your run hits the recursion limit. What's the usual cause and the correct fix?"**
   A routing loop: the supervisor keeps delegating to a specialist that defers back, so the step count climbs. The fix is the supervisor's prompt (make the routing and stop conditions unambiguous), not raising the limit: routine limit hits indicate a prompt bug.

4. **Conceptual: "What is the orchestrator-worker pattern and when do you use it over plain parallelization?"**
   An orchestrator dynamically breaks a task into subtasks, dispatches them to workers (often in parallel via the Send API), and synthesizes their outputs from a shared accumulating state key. You use it when the subtasks can't be predefined (a report with a variable number of sections, edits across an unknown number of files); plain parallelization suffices when the subtasks are fixed and independent.

5. **Gotcha: "In a swarm handoff, the next agent behaves as if it lost the conversation. Why?"**
   The handing-off agent's internal messages didn't propagate to the parent graph, so the next agent sees an incomplete or broken history. Ensure the `Command`'s `update` carries the relevant messages and that tool calls remain paired with their tool-result messages, or the next agent will error or hallucinate.

---

## Gotchas summary table

| # | Gotcha | Where it bites | Fix / rule of thumb |
|---|---|---|---|
| 1 | Reaching for multi-agent because it "sounds capable" | Architecture | Climb the complexity ladder; justify multi-agent with an eval a single agent fails |
| 2 | Multi-agent's hidden token cost (~15× reported) | Architecture / budget | Confirm the breadth/decomposition is worth the multiplier |
| 3 | Letting a framework hide the prompts/logic | Any framework | Understand what it does underneath; keep the ability to simplify |
| 4 | CrewAI: blank/vague `expected_output` | CrewAI tasks | It's part of the prompt: describe the deliverable concretely |
| 5 | CrewAI: forgetting `context` between tasks | CrewAI tasks | Link tasks with `context=[...]` so outputs flow |
| 6 | CrewAI: crew-level tools only | CrewAI agents | Scope tools to the agent whose role needs them |
| 7 | CrewAI: hierarchical when sequential works | CrewAI process | Use sequential for known order (cheaper/faster) |
| 8 | CrewAI: memory off by default | CrewAI runs | Set `memory=True` where you need continuity |
| 9 | AutoGen: no termination condition | AutoGen teams | Always attach one; combine phrase + max-message |
| 10 | AutoGen: forgetting it's async | AutoGen | Everything is `async`/`await` under `asyncio.run` |
| 11 | AutoGen: mixing v0.2 and v0.4 APIs | AutoGen setup | Use `autogen_agentchat` + `model_client`, not `llm_config` |
| 12 | AutoGen: it's in maintenance mode | AutoGen (new builds) | Evaluate the Microsoft Agent Framework for greenfield |
| 13 | LangGraph: specialist tools on the supervisor | Supervisor pattern | Supervisor only routes/finishes; tools live on specialists |
| 14 | LangGraph: hitting `recursion_limit` | Supervisor/loops | It's a routing loop: fix the prompt, not the limit |
| 15 | LangGraph: lost messages on handoff | Handoffs / swarm | Include messages in `Command.update`; keep tool/result pairs |
| 16 | No per-agent tracing | Any multi-agent | Trace each agent as its own span (e.g. LangSmith) |

---

## Quick reference card

**When to go multi-agent**

```text
ladder: 1 LLM call -> augmented LLM (tools) -> workflow -> single agent -> multi-agent
climb only when the rung below FAILS on evals
worth it: parallel breadth-first work | clean decomposition w/ verifiable results
          | different tools-prompts-perms-models per subtask | debate/conversation shape
agent design: environment + tools + system prompt (spend most effort on tools)
```

**CrewAI (role-based)**

```text
pip install crewai            # Python 3.10–3.13
Agent(role, goal, backstory, tools, llm, allow_delegation)
Task(description, expected_output, agent, context=[...], output_pydantic)
Crew(agents=[...], tasks=[...], process=Process.sequential | Process.hierarchical).kickoff(inputs={...})
hierarchical needs manager_llm ; guards: max_iter (15), allow_delegation=False, memory=True, telemetry=False
Flows (@start/@listen/@router) for control-heavy orchestration
```

**AutoGen (conversation-based): maintenance mode; successor = Microsoft Agent Framework**

```text
pip install autogen-agentchat autogen-ext[openai]     # async-first!
AssistantAgent(name, model_client, system_message, tools, description)   # 1 step/turn by default
UserProxyAgent(...)   # human/code-exec proxy
teams: RoundRobinGroupChat([...]) | SelectorGroupChat([...], model_client=...) | MagenticOneGroupChat | Swarm
run: await team.run(task=...) / run_stream(...)   ; stateful (reset() to clear)
ALWAYS set termination: MaxMessageTermination(n) | TextMentionTermination("APPROVE") (combine w/ And/Or)
```

**LangGraph supervisor / orchestration**

```text
agents are nodes; route + update with Command(goto="node", update={...})  (graph=Command.PARENT across subgraphs)
supervisor: from langgraph_supervisor import create_supervisor
    create_supervisor([agent_a, agent_b], model=model, prompt="route; finish when done").compile()
    handoffs = tool calls; terminates when it answers without delegating
    keep specialist tools OFF the supervisor ; recursion_limit=25 (a hit = routing loop -> fix prompt)
orchestrator-worker: orchestrator plans -> Send("worker", input) fan-out -> workers write shared reducer key -> synthesize
    use when subtasks aren't predefined
plan-execute: planner -> executor -> replan loop until done
swarm: agents hand off directly (no supervisor): faster, less accurate; start with supervisor
```

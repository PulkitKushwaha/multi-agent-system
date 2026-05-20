# multi-agent-system
 
This here is a production-grade multi-agent AI system built with LangGraph,
featuring a planner agent, retriever agent, and synthesizer agent
orchestrated through a typed state machine with evaluation harness
and documented failure modes.
 
> Single-agent systems hit a ceiling. Complex tasks require
> decomposition, parallel reasoning, and specialized agents working
> together. This repo explores what it takes to build multi-agent
> systems that are reliable enough to deploy, not just impressive
> enough to demo.
 
---
 
## Why multi-agent architecture
 
A single LLM call can answer simple questions. But for complex tasks like
research synthesis, multi-step reasoning, document analysis across
sources, etc. a single agent becomes a bottleneck.
 
Multi-agent systems solve this through specialization:
 
- **Planner** understands the task and decomposes it into subtasks
- **Retriever** knows how to find relevant information efficiently
- **Synthesizer** knows how to combine information into a coherent answer
Each agent is optimized for its role. The orchestrator (LangGraph)
coordinates them through routing tasks, managing state, and handling failures.
 
---
 
## Architecture
 
```
User Query
    ↓
┌─────────────────────────────────────────────┐
│              LangGraph Orchestrator          │
│                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ Planner  │───▶│Retriever │───▶│Synthe- │ │
│  │  Agent   │    │  Agent   │    │ sizer  │ │
│  │          │◀───│          │    │ Agent  │ │
│  └──────────┘    └──────────┘    └────────┘ │
│       │               │               │     │
│       ▼               ▼               ▼     │
│  [Task decomp]  [Vector search]  [Answer    │
│  [Routing]      [Web search]     synthesis] │
│  [Replanning]   [Doc analysis]   [Citation] │
└─────────────────────────────────────────────┘
    ↓
Final Answer + Reasoning Trace
```
 
---
 
## Agent roles
 
### Planner Agent
Receives the user query and decides how to approach it.
- Decomposes complex queries into subtasks
- Decides which agents to invoke and in what order
- Monitors progress and replans if subtasks fail
- Produces a structured task plan with dependencies
### Retriever Agent
Specialized in finding relevant information.
- Chooses between vector search, web search, or document analysis
- Applies metadata filtering for access-controlled knowledge bases
- Returns ranked results with source attribution
- Signals confidence: knows when it hasn't found enough
### Synthesizer Agent
Transforms retrieved information into a final answer.
- Combines outputs from multiple retrieval calls
- Maintains faithfulness to sources, no hallucination
- Formats output according to task requirements
- Generates citations and confidence indicators
---
 
## LangGraph state machine
 
```python
# State shared across all agents
class AgentState(TypedDict):
    query: str                    # Original user query
    task_plan: list               # Planner's decomposed subtasks
    retrieved_context: list       # All retrieved content
    intermediate_answers: list    # Partial answers from subtasks
    final_answer: str             # Synthesized final answer
    reasoning_trace: list         # Full decision log
    iteration_count: int          # Loop detection counter
    error: Optional[str]          # Error state if any
```
 
The state machine defines:
- **Nodes** — each agent is a node that reads/writes state
- **Edges** — conditional routing based on state
- **Entry point** — always the Planner
- **Terminal states** — success (answer produced) or failure (max iterations)
---
 
## Failure modes and mitigations
 
Multi-agent systems fail in ways single agents don't.
This repo documents and tests for each:
 
| Failure mode | Description | Mitigation |
|---|---|---|
| Agent loop | Agents cycle without progress | Max iteration counter |
| Hallucinated tool calls | Agent invents tool parameters | Strict Pydantic schemas |
| Context overflow | State grows beyond context window | Context compression |
| Planner over-decomposition | Too many subtasks for simple queries | Complexity classifier |
| Retriever under-retrieval | Not enough context for synthesis | Confidence threshold + retry |
| Synthesizer hallucination | Answer not grounded in retrieved context | Faithfulness check |
 
---
 
## Structure
 
```
multi-agent-system/
├── src/
│   ├── agents/        # Planner, Retriever, Synthesizer implementations
│   ├── tools/         # Tool definitions (search, retrieval, analysis)
│   ├── graph/         # LangGraph state machine and routing logic
│   ├── memory/        # Agent memory (episodic, semantic, procedural)
│   └── evaluation/    # Agent evaluation harness
├── examples/          # Usage examples
├── notebooks/         # Experiment notebooks
└── tests/             # Unit tests
```
 
---
 
## Status
 
| Component | Status |
|---|---|
| Planner agent | 🟡 In progress |
| Retriever agent | ⬜ Coming soon |
| Synthesizer agent | ⬜ Coming soon |
| LangGraph state machine | ⬜ Coming soon |
| Agent memory | ⬜ Coming soon |
| Evaluation harness | ⬜ Coming soon |
| Failure mode tests | ⬜ Coming soon |
 
---
 
*Part of the [ai-engineering-portfolio](https://github.com/pulkitkushwaha/ai-engineering-portfolio)
— built by [Pulkit Kushwaha](https://linkedin.com/in/pulkit-kushwaha)*

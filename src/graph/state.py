"""
AgentState — Shared State for the Multi-Agent System
 
In LangGraph, all agents communicate through a shared state object.
Each agent reads from state, does its work, and writes back to state.
The state machine (graph) routes between agents based on state values.
 
Design decisions:
 
1. TypedDict over dataclass
   LangGraph requires TypedDict for state; it uses the type hints
   to manage state updates and merging between nodes.
2. All fields Optional with defaults
   Agents may not populate every field. Optional fields prevent
   KeyError on first access and make the state self-documenting
   about what each agent is responsible for.
3. reasoning_trace as a log
   Every agent appends to reasoning_trace; this produces a full
   audit log of every decision made during a task. Critical for
   debugging and for showing reasoning to users.
4. error field for graceful degradation
   When an agent fails, it sets the error field instead of raising.
   The graph checks this field to route to error handling rather
   than continuing with bad state.
5. iteration_count for loop detection
   Every cycle through the graph increments this. The graph
   terminates when it exceeds MAX_ITERATIONS; preventing
   infinite loops when agents disagree on task completion.
"""
from typing import TypedDict, List, Optional, Dict, Any
 
 
class AgentState(TypedDict):
    """
    Shared state passed between all agents in the system.
 
    This is the single source of truth for the entire multi-agent
    task. Every agent reads from here and writes back here.
    LangGraph manages state updates atomically between nodes.
 
    Fields:
        query               : Original user query — never modified
        task_plan           : Planner's decomposed list of subtasks
        current_subtask     : Which subtask the system is working on
        retrieved_context   : All content retrieved so far
        intermediate_answers: Partial answers from completed subtasks
        final_answer        : The synthesizer's completed response
        sources             : Citations for retrieved content
        reasoning_trace     : Full log of every agent decision
        iteration_count     : Loop detection counter
        max_iterations      : Hard limit to prevent infinite loops
        error               : Set when an agent fails — triggers error routing
        metadata            : Arbitrary task metadata (user_id, session, etc.)
    """
    query: str
    task_plan: Optional[List[Dict[str, Any]]]
    current_subtask: Optional[Dict[str, Any]]
    retrieved_context: Optional[List[Dict[str, Any]]]
    intermediate_answers: Optional[List[str]]
    final_answer: Optional[str]
    sources: Optional[List[str]]
    reasoning_trace: Optional[List[str]]
    iteration_count: Optional[int]
    max_iterations: Optional[int]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]
 
 
def create_initial_state(
    query: str,
    max_iterations: int = 10,
    metadata: Optional[Dict[str, Any]] = None
) -> AgentState:
    """
    Create a fresh AgentState for a new task.
 
    All optional fields initialized to empty defaults.
    The graph always starts from this state.
 
    Args:
        query          : User's original query
        max_iterations : Maximum graph iterations before forced stop
        metadata       : Optional task metadata
 
    Returns:
        Initialized AgentState ready for the graph
    """
    return AgentState(
        query=query,
        task_plan=None,
        current_subtask=None,
        retrieved_context=[],
        intermediate_answers=[],
        final_answer=None,
        sources=[],
        reasoning_trace=[f"Task started: {query}"],
        iteration_count=0,
        max_iterations=max_iterations,
        error=None,
        metadata=metadata or {}
    )
 
 
def log_step(state: AgentState, agent: str, message: str) -> List[str]:
    """
    Append a reasoning step to the trace log.
 
    Returns the updated trace list for state update.
 
    Args:
        state   : Current AgentState
        agent   : Name of the agent logging this step
        message : What the agent decided or did
 
    Returns:
        Updated reasoning_trace list
    """
    trace = list(state.get("reasoning_trace") or [])
    trace.append(f"[{agent}] {message}")
    return trace

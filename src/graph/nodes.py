"""
LangGraph Node Definitions
 
In LangGraph, nodes are functions that take AgentState as input
and return a dict of state updates. Each agent becomes a node.
 
This file wraps the agent classes into LangGraph-compatible
node functions and defines the routing logic between them.
 
Node execution flow:
    planner_node → retriever_node → synthesizer_node → END
                        ↑                    |
                        └────────────────────┘
                         (if needs_more_retrieval)
 
Routing decisions:
    After planner  → always go to retriever
    After retriever → go to synthesizer (if confident)
                    → go back to planner (if confidence low)
    After synthesizer → END (if answer complete)
                      → go to retriever (if needs_more_retrieval)
                      → END with error (if max iterations hit)
"""
 
from typing import Dict, Any, Literal
from src.graph.state import AgentState, log_step
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
 
 
# ── Node factory functions ────────────────────────────────────
 
def make_planner_node(agent: PlannerAgent):
    """Wrap PlannerAgent as a LangGraph node function."""
    def planner_node(state: AgentState) -> Dict[str, Any]:
        return agent.run(state)
    planner_node.__name__ = "planner"
    return planner_node
 
 
def make_retriever_node(agent: RetrieverAgent):
    """Wrap RetrieverAgent as a LangGraph node function."""
    def retriever_node(state: AgentState) -> Dict[str, Any]:
        return agent.run(state)
    retriever_node.__name__ = "retriever"
    return retriever_node
 
 
def make_synthesizer_node(agent: SynthesizerAgent):
    """Wrap SynthesizerAgent as a LangGraph node function."""
    def synthesizer_node(state: AgentState) -> Dict[str, Any]:
        return agent.run(state)
    synthesizer_node.__name__ = "synthesizer"
    return synthesizer_node
 
 
# ── Routing functions ─────────────────────────────────────────
 
def route_after_retriever(state: AgentState) -> Literal["synthesizer", "planner"]:
    """
    Route after retrieval based on confidence and iteration count.
 
    If the retriever found sufficient information → synthesize.
    If confidence is low AND we haven't hit max iterations → replan.
    If we've hit max iterations → synthesize with what we have.
    """
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    current_subtask = state.get("current_subtask", {}) or {}
    confidence = current_subtask.get("retrieval_confidence", 1.0)
 
    # Safety valve — never loop forever
    if iteration >= max_iterations:
        trace = log_step(
            state,
            "Router",
            f"Max iterations ({max_iterations}) reached — forcing synthesis"
        )
        return "synthesizer"
 
    # Low confidence and we have budget — ask planner to retry
    if confidence < 0.5 and iteration < 3:
        trace = log_step(
            state,
            "Router",
            f"Low retrieval confidence ({confidence:.2f}) — replanning"
        )
        return "planner"
 
    # Default: proceed to synthesis
    return "synthesizer"
 
 
def route_after_synthesizer(state: AgentState) -> Literal["retriever", "__end__"]:
    """
    Route after synthesis.
 
    If synthesizer flagged needs_more_retrieval AND we have budget → retrieve more.
    Otherwise → END.
    """
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 10)
    final_answer = state.get("final_answer", "")
 
    # Check if synthesis flagged insufficient context
    # We infer this from the final answer containing "don't have enough information"
    needs_more = (
        "don't have enough information" in (final_answer or "").lower()
        and iteration < max_iterations - 1
    )
 
    if needs_more:
        trace = log_step(
            state,
            "Router",
            "Synthesizer flagged insufficient context, now retrieving more"
        )
        return "retriever"
 
    return "__end__"

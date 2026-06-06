"""
Failure Mode Tests for Multi-Agent System
 
Tests that the agent system handles known failure modes
gracefully, not that failures never occur, but that they
are caught, handled, and don't cause crashes or silent errors.
"""
 
import pytest
from src.graph.state import create_initial_state, AgentState
from src.graph.nodes import route_after_retriever, route_after_synthesizer
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
 
 
# ── Routing tests ─────────────────────────────────────────────
 
def test_route_after_retriever_high_confidence():
    """High confidence retrieval routes to synthesizer."""
    state = create_initial_state("test query")
    state["current_subtask"] = {"retrieval_confidence": 0.9}
    state["iteration_count"] = 1
    assert route_after_retriever(state) == "synthesizer"
 
 
def test_route_after_retriever_low_confidence_early():
    """Low confidence on first iteration routes back to planner."""
    state = create_initial_state("test query")
    state["current_subtask"] = {"retrieval_confidence": 0.3}
    state["iteration_count"] = 1
    assert route_after_retriever(state) == "planner"
 
 
def test_route_after_retriever_max_iterations_forces_synthesis():
    """Max iterations reached always routes to synthesizer."""
    state = create_initial_state("test query", max_iterations=5)
    state["current_subtask"] = {"retrieval_confidence": 0.1}
    state["iteration_count"] = 5
    assert route_after_retriever(state) == "synthesizer"
 
 
def test_route_after_synthesizer_complete_answer():
    """Complete answer routes to END."""
    state = create_initial_state("test query")
    state["final_answer"] = "The return policy allows 30 days."
    state["iteration_count"] = 1
    assert route_after_synthesizer(state) == "__end__"
 
 
def test_route_after_synthesizer_insufficient_info():
    """Insufficient info triggers more retrieval."""
    state = create_initial_state("test query")
    state["final_answer"] = "I don't have enough information to answer this."
    state["iteration_count"] = 1
    assert route_after_synthesizer(state) == "retriever"
 
 
def test_route_after_synthesizer_max_iterations_ends():
    """Max iterations forces END even with insufficient info."""
    state = create_initial_state("test query", max_iterations=10)
    state["final_answer"] = "I don't have enough information."
    state["iteration_count"] = 9
    assert route_after_synthesizer(state) == "__end__"
 
 
# ── Agent tests ───────────────────────────────────────────────
 
def test_planner_handles_empty_query():
    """Planner should produce a plan even for short queries."""
    planner = PlannerAgent(verbose=False)
    state = create_initial_state("What?")
    result = planner.run(state)
    assert "task_plan" in result
    assert result["task_plan"] is not None
 
 
def test_planner_increments_iteration():
    """Planner should increment iteration count."""
    planner = PlannerAgent(verbose=False)
    state = create_initial_state("test query")
    state["iteration_count"] = 2
    result = planner.run(state)
    assert result["iteration_count"] == 3
 
 
def test_retriever_handles_no_vector_store():
    """Retriever without vector store should not crash."""
    retriever = RetrieverAgent(vector_store=None, embedder=None, verbose=False)
    state = create_initial_state("test query")
    state["current_subtask"] = {
        "id": "subtask_1",
        "description": "Find return policy",
        "retrieval_type": "vector_search"
    }
    result = retriever.run(state)
    assert "reasoning_trace" in result
 
 
def test_retriever_handles_missing_subtask():
    """Retriever with no subtask should skip gracefully."""
    retriever = RetrieverAgent(verbose=False)
    state = create_initial_state("test query")
    state["current_subtask"] = None
    result = retriever.run(state)
    assert "reasoning_trace" in result
 
 
def test_synthesizer_handles_empty_context():
    """Synthesizer with no context should return graceful message."""
    synthesizer = SynthesizerAgent(verbose=False)
    state = create_initial_state("test query")
    state["retrieved_context"] = []
    state["intermediate_answers"] = []
    result = synthesizer.run(state)
    assert "final_answer" in result
    assert result["final_answer"] is not None
    assert len(result["final_answer"]) > 0
 
 
def test_state_reasoning_trace_populated():
    """Reasoning trace should be populated after agent runs."""
    planner = PlannerAgent(verbose=False)
    state = create_initial_state("What is the return policy?")
    result = planner.run(state)
    assert "reasoning_trace" in result
    assert len(result["reasoning_trace"]) > 0
 
 
def test_initial_state_defaults():
    """Initial state should have correct defaults."""
    state = create_initial_state("test query", max_iterations=5)
    assert state["query"] == "test query"
    assert state["max_iterations"] == 5
    assert state["iteration_count"] == 0
    assert state["error"] is None
    assert state["final_answer"] is None
    assert isinstance(state["retrieved_context"], list)
    assert isinstance(state["reasoning_trace"], list)
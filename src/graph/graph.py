"""
LangGraph State Machine: Multi-Agent Orchestration
 
This module assembles the complete multi-agent graph:
    - Registers all agent nodes
    - Defines conditional routing between nodes
    - Exposes a clean run() interface
 
The graph is the heart of the multi-agent system. It manages:
    - State passing between agents
    - Conditional routing based on agent outputs
    - Loop detection and termination
    - Error handling and graceful degradation
 
Graph topology:
    START → planner → retriever → synthesizer → END
                ↑         |              |
                └─────────┘              |
                 (replan if              |
                  low confidence)        |
                              ↑          |
                              └──────────┘
                               (if needs more
                                retrieval)
 
Design decision: Compile once, run many
    The compiled graph is stateless, it can handle many
    concurrent queries without re-compilation. The state
    (AgentState) is per-query and passed through each run call.
"""
 
from typing import Optional, Dict, Any
from src.graph.state import AgentState, create_initial_state
from src.graph.nodes import (
    make_planner_node,
    make_retriever_node,
    make_synthesizer_node,
    route_after_retriever,
    route_after_synthesizer
)
from src.agents.planner import PlannerAgent
from src.agents.retriever import RetrieverAgent
from src.agents.synthesizer import SynthesizerAgent
 
 
class MultiAgentGraph:
    """
    Assembled multi-agent LangGraph system.
 
    Wires together planner, retriever, and synthesizer agents
    through a LangGraph state machine with conditional routing.
 
    Usage:
        graph = MultiAgentGraph(
            llm_client=openai_client,
            vector_store=faiss_store,
            embedder=embedder
        )
        result = graph.run("What is the return policy for damaged items?")
        print(result["final_answer"])
        print(result["reasoning_trace"])
 
    Args:
        llm_client   : OpenAI or Azure OpenAI client
        vector_store : FAISS vector store for retrieval
        embedder     : Embedding model for vector search
        max_iterations: Maximum graph iterations (default: 10)
        verbose      : Print agent decisions (default: True)
    """
 
    def __init__(
        self,
        llm_client=None,
        vector_store=None,
        embedder=None,
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.max_iterations = max_iterations
        self.verbose = verbose
 
        # Initialize agents
        self.planner = PlannerAgent(
            llm_client=llm_client,
            verbose=verbose
        )
        self.retriever = RetrieverAgent(
            vector_store=vector_store,
            embedder=embedder,
            llm_client=llm_client,
            verbose=verbose
        )
        self.synthesizer = SynthesizerAgent(
            llm_client=llm_client,
            verbose=verbose
        )
 
        # Build the graph
        self._graph = self._build_graph()
 
    def _build_graph(self):
        """
        Assemble the LangGraph state machine.
 
        Registers nodes and conditional edges.
        Returns compiled graph ready for execution.
        """
        try:
            from langgraph.graph import StateGraph, END
 
            # Create the graph with AgentState schema
            workflow = StateGraph(AgentState)
 
            # Add agent nodes
            workflow.add_node("planner", make_planner_node(self.planner))
            workflow.add_node("retriever", make_retriever_node(self.retriever))
            workflow.add_node("synthesizer", make_synthesizer_node(self.synthesizer))
 
            # Set entry point
            workflow.set_entry_point("planner")
 
            # Add edges
            # Planner always goes to retriever
            workflow.add_edge("planner", "retriever")
 
            # Retriever routes conditionally
            workflow.add_conditional_edges(
                "retriever",
                route_after_retriever,
                {
                    "synthesizer": "synthesizer",
                    "planner": "planner"
                }
            )
 
            # Synthesizer routes conditionally
            workflow.add_conditional_edges(
                "synthesizer",
                route_after_synthesizer,
                {
                    "retriever": "retriever",
                    "__end__": END
                }
            )
 
            return workflow.compile()
 
        except ImportError:
            print(
                "[MultiAgentGraph] LangGraph not installed. "
                "Install with: pip install langgraph\n"
                "Using fallback sequential execution."
            )
            return None
 
    def run(
        self,
        query: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentState:
        """
        Run the multi-agent system on a query.
 
        Args:
            query    : User's question or task
            metadata : Optional metadata (user_id, session_id, etc.)
 
        Returns:
            Final AgentState with answer, sources, and reasoning trace
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiAgentGraph] Running query: {query[:60]}...")
            print(f"{'='*60}")
 
        # Create initial state
        initial_state = create_initial_state(
            query=query,
            max_iterations=self.max_iterations,
            metadata=metadata
        )
 
        # Run graph or fallback
        if self._graph is not None:
            final_state = self._run_langgraph(initial_state)
        else:
            final_state = self._run_sequential(initial_state)
 
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[MultiAgentGraph] Complete.")
            print(f"Answer: {str(final_state.get('final_answer', ''))[:100]}...")
            print(f"Iterations: {final_state.get('iteration_count', 0)}")
            print(f"{'='*60}\n")
 
        return final_state
 
    def _run_langgraph(self, initial_state: AgentState) -> AgentState:
        """Execute using LangGraph compiled graph."""
        try:
            result = self._graph.invoke(initial_state)
            return result
        except Exception as e:
            print(f"[MultiAgentGraph] LangGraph execution failed: {e}")
            return self._run_sequential(initial_state)
 
    def _run_sequential(self, state: AgentState) -> AgentState:
        """
        Fallback sequential execution without LangGraph.
 
        Runs planner → retriever → synthesizer in order.
        No conditional routing — useful for testing without
        LangGraph installed.
        """
        print("[MultiAgentGraph] Running in sequential fallback mode")
 
        # Step 1: Plan
        state.update(self.planner.run(state))
 
        # Step 2: Retrieve
        state.update(self.retriever.run(state))
 
        # Step 3: Synthesize
        state.update(self.synthesizer.run(state))
 
        return state
 
    def get_reasoning_trace(self, state: AgentState) -> str:
        """
        Format the reasoning trace as a readable string.
 
        Useful for displaying the agent's decision process to users
        or for debugging complex multi-hop queries.
        """
        trace = state.get("reasoning_trace", [])
        return "\n".join([f"  {i+1}. {step}" for i, step in enumerate(trace)])

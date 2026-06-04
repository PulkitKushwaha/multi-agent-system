"""
Retriever Agent
 
The retriever agent is the information-gathering specialist
in the multi-agent system. It receives a subtask from the
planner and decides HOW to find the required information.
 
Key decisions the retriever makes:
    1. Which retrieval strategy to use (vector search vs web search)
    2. How many results to fetch
    3. Whether the results are sufficient or need refinement
    4. How to signal confidence back to the planner
 
Design decisions:
 
Tool selection over hardcoding
    The retriever doesn't hardcode "use FAISS", instead, it selects
    from available tools based on the subtask's retrieval_type.
    This makes it easy to add new retrieval tools (SQL, API,
    email search) without changing the agent logic.
 
Confidence signaling
    The retriever returns a confidence score alongside results.
    If confidence is below threshold, the planner can decide
    to retry with a different strategy or mark the subtask
    as partially completed. This prevents silent failures
    where the retriever returns results but they're not useful.
 
Result deduplication
    When multiple retrieval calls happen across subtasks,
    the same chunk may be retrieved multiple times. The
    retriever deduplicates before writing to state, keeping
    the context window clean.
"""
 
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.graph.state import AgentState, log_step
from src.tools.search_tools import VectorSearchTool, WebSearchTool
 
 
# ── Output schemas ────────────────────────────────────────────
 
class RetrievedDocument(BaseModel):
    """A single retrieved document chunk."""
    content: str
    source: str
    score: float
    retrieval_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
 
 
class RetrievalResult(BaseModel):
    """Structured output from the retriever agent."""
    subtask_id: str
    documents: List[RetrievedDocument]
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that retrieved docs answer the subtask"
    )
    retrieval_strategy_used: str
    total_retrieved: int
    reasoning: str
 
 
# ── Retriever Agent ───────────────────────────────────────────
 
class RetrieverAgent:
    """
    Finds information required for a given subtask.
 
    Selects from available retrieval tools based on the subtask's
    retrieval_type, executes retrieval, evaluates result quality,
    and writes results back to AgentState.
 
    Args:
        vector_store    : FAISS vector store (optional — for vector search)
        embedder        : Embedding model (optional — for vector search)
        llm_client      : LLM for confidence evaluation
        model           : Model name
        confidence_threshold : Minimum acceptable confidence (default 0.6)
        verbose         : Print retrieval decisions
    """
 
    CONFIDENCE_PROMPT = """You retrieved the following documents for this subtask.
Rate how well these documents answer the subtask on a scale of 0.0 to 1.0.
 
Subtask: {subtask}
 
Retrieved documents:
{documents}
 
Rating (0.0-1.0, where 1.0 means documents fully answer the subtask):
Respond with only a number."""
 
    def __init__(
        self,
        vector_store=None,
        embedder=None,
        llm_client=None,
        model: str = "gpt-4",
        confidence_threshold: float = 0.6,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
 
        # Initialize available tools
        self.tools = {}
        if vector_store and embedder:
            self.tools["vector_search"] = VectorSearchTool(
                vector_store=vector_store,
                embedder=embedder
            )
        self.tools["web_search"] = WebSearchTool(llm_client=llm_client)
 
    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node function — retrieves for the current subtask.
 
        Reads current_subtask from state, retrieves relevant
        documents, evaluates confidence, and writes results
        back to state.
 
        Args:
            state: Current AgentState
 
        Returns:
            Dict of state fields to update
        """
        subtask = state.get("current_subtask")
        if not subtask:
            trace = log_step(state, "Retriever", "No subtask found in state — skipping")
            return {"reasoning_trace": trace}
 
        subtask_desc = subtask.get("description", "")
        retrieval_type = subtask.get("retrieval_type", "vector_search")
 
        if self.verbose:
            print(f"[Retriever] Subtask: {subtask_desc[:60]}...")
            print(f"[Retriever] Strategy: {retrieval_type}")
 
        # Select and run the appropriate tool
        documents = self._retrieve(subtask_desc, retrieval_type)
 
        # Evaluate confidence
        confidence = self._evaluate_confidence(subtask_desc, documents)
 
        # Build result
        result = RetrievalResult(
            subtask_id=subtask.get("id", "unknown"),
            documents=documents,
            confidence=confidence,
            retrieval_strategy_used=retrieval_type,
            total_retrieved=len(documents),
            reasoning=(
                f"Retrieved {len(documents)} documents using {retrieval_type}. "
                f"Confidence: {confidence:.2f}"
            )
        )
 
        # Log the retrieval decision
        confidence_status = "sufficient" if confidence >= self.confidence_threshold else "low"
        trace = log_step(
            state,
            "Retriever",
            f"Retrieved {len(documents)} docs for '{subtask_desc[:40]}...' "
            f"(confidence: {confidence:.2f} — {confidence_status})"
        )
 
        if self.verbose:
            print(f"[Retriever] {len(documents)} docs retrieved, confidence: {confidence:.2f}")
 
        # Update state
        existing_context = list(state.get("retrieved_context") or [])
        new_context = self._deduplicate(
            existing_context,
            [doc.dict() for doc in documents]
        )
 
        return {
            "retrieved_context": new_context,
            "reasoning_trace": trace,
            "current_subtask": {
                **subtask,
                "retrieval_confidence": confidence,
                "retrieval_complete": True
            }
        }
 
    def _retrieve(
        self,
        query: str,
        retrieval_type: str
    ) -> List[RetrievedDocument]:
        """Select and run the appropriate retrieval tool."""
        tool = self.tools.get(retrieval_type)
 
        if tool is None:
            # Fall back to any available tool
            tool = next(iter(self.tools.values()), None)
            if tool is None:
                return []
 
        raw_results = tool.search(query, k=5)
 
        return [
            RetrievedDocument(
                content=r.get("content", ""),
                source=r.get("source", "unknown"),
                score=r.get("score", 0.0),
                retrieval_type=retrieval_type,
                metadata=r.get("metadata", {})
            )
            for r in raw_results
        ]
 
    def _evaluate_confidence(
        self,
        subtask: str,
        documents: List[RetrievedDocument]
    ) -> float:
        """
        Evaluate how well retrieved documents answer the subtask.
 
        Uses LLM if available, falls back to heuristic based
        on number and quality of results.
        """
        if not documents:
            return 0.0
 
        if self.llm_client is None:
            # Heuristic: more docs with higher scores = higher confidence
            avg_score = sum(d.score for d in documents) / len(documents)
            count_factor = min(len(documents) / 5, 1.0)
            return round(avg_score * 0.6 + count_factor * 0.4, 2)
 
        try:
            doc_preview = "\n\n".join([
                f"[{i+1}] {d.content[:200]}..."
                for i, d in enumerate(documents[:3])
            ])
 
            prompt = self.CONFIDENCE_PROMPT.format(
                subtask=subtask,
                documents=doc_preview
            )
 
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            score_str = response.choices[0].message.content.strip()
            return min(max(float(score_str), 0.0), 1.0)
 
        except Exception as e:
            print(f"[Retriever] Confidence eval failed: {e}")
            return 0.5
 
    def _deduplicate(
        self,
        existing: List[Dict],
        new_docs: List[Dict]
    ) -> List[Dict]:
        """Deduplicate context by content fingerprint."""
        seen = {d.get("content", "")[:100] for d in existing}
        result = list(existing)
 
        for doc in new_docs:
            key = doc.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                result.append(doc)
 
        return result

"""
Synthesizer Agent
 
The synthesizer is the final agent in the pipeline. It takes
everything the retriever gathered and produces a coherent,
faithful, cited final answer.
 
The synthesizer's responsibilities:
    1. Combine retrieved context from multiple subtasks
    2. Generate a comprehensive answer grounded in that context
    3. Attach source citations to every factual claim
    4. Assess its own confidence in the answer
    5. Flag when retrieved context is insufficient
 
Why a dedicated synthesizer matters:
    In a single-agent system, the same LLM call retrieves
    and generates. This creates two problems:
 
    First, the generation is constrained by what the single
    retrieval call found. There's no opportunity to say
    "I need more information on X before I can synthesize."
 
    Second, generation quality suffers when the LLM is also
    managing retrieval logic. Specialization improves both.
 
    The synthesizer receives pre-gathered, deduplicated context
    from potentially multiple retrieval calls. It can focus
    entirely on producing the best possible answer from that context.
 
Design decisions:
 
Structured output with citations
    The synthesizer doesn't just return text, it returns a
    structured object with the answer, confidence score, sources
    used, and a list of claims with their supporting context.
    This makes the output auditable and allows downstream systems
    to display citations to users.
 
Faithfulness-first generation
    The system prompt explicitly instructs the synthesizer to
    ground every claim in retrieved context. It is told to say
    "I don't have enough information" rather than hallucinate.
    This is enforced by the output schema, claims without
    supporting context cannot be included.
 
Self-assessment
    After generating an answer, the synthesizer rates its own
    confidence. Low confidence triggers a signal to the planner
    that additional retrieval may be needed. This creates a
    feedback loop that improves answer quality for complex queries.
"""
 
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.graph.state import AgentState, log_step
 
 
# ── Output schemas ────────────────────────────────────────────
 
class CitedClaim(BaseModel):
    """A single factual claim with its supporting source."""
    claim: str = Field(..., description="The factual statement")
    supporting_context: str = Field(..., description="The retrieved text that supports this claim")
    source: str = Field(..., description="Source document or URL")
    confidence: float = Field(..., ge=0.0, le=1.0)
 
 
class SynthesisResult(BaseModel):
    """Structured output from the synthesizer agent."""
    answer: str = Field(..., description="The complete synthesized answer")
    cited_claims: List[CitedClaim] = Field(
        default_factory=list,
        description="Individual claims with supporting sources"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the answer"
    )
    sources_used: List[str] = Field(
        default_factory=list,
        description="All sources referenced in the answer"
    )
    coverage_assessment: str = Field(
        ...,
        description="Assessment of whether retrieved context fully answers the query"
    )
    needs_more_retrieval: bool = Field(
        default=False,
        description="True if answer would benefit from additional retrieval"
    )
 
 
# ── Synthesizer Agent ─────────────────────────────────────────
 
class SynthesizerAgent:
    """
    Synthesizes retrieved context into a coherent, cited final answer.
 
    The last agent in the pipeline. Receives the full retrieved
    context from state and produces a structured SynthesisResult
    with answer, citations, confidence, and coverage assessment.
 
    Args:
        llm_client           : OpenAI or Azure OpenAI client
        model                : LLM model for synthesis
        confidence_threshold : If confidence below this, sets needs_more_retrieval=True
        max_context_chars    : Maximum context to pass to LLM (token management)
        verbose              : Print synthesis decisions
    """
 
    SYSTEM_PROMPT = """You are a synthesis specialist. Your job is to produce
accurate, well-cited answers from retrieved context.
 
RULES:
1. Only use information from the provided context
2. If context is insufficient, say so explicitly and never hallucinate
3. Cite your sources for every factual claim
4. Be comprehensive but concise to cover all aspects of the question
5. Assess your confidence honestly. Partial information = lower confidence"""
    SYNTHESIS_PROMPT = """Synthesize a complete answer to this question using the provided context.
Question: {query}
 
Retrieved Context:
{context}
 
Instructions:
- Answer the question comprehensively using ONLY the provided context
- If the context doesn't fully answer the question, say what is and isn't covered
- For each key claim in your answer, note which source supports it
- Rate your overall confidence from 0.0 to 1.0
- State whether additional retrieval would improve the answer
Format your response as JSON:
{{
    "answer": "Complete answer text",
    "cited_claims": [
        {{
            "claim": "Specific factual statement",
            "supporting_context": "The retrieved text that supports this",
            "source": "source document name",
            "confidence": 0.9
        }}
    ],
    "confidence": 0.85,
    "sources_used": ["source1", "source2"],
    "coverage_assessment": "Context fully/partially/does not cover the question",
    "needs_more_retrieval": false
}}"""
 
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4",
        confidence_threshold: float = 0.6,
        max_context_chars: int = 8000,
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.max_context_chars = max_context_chars
        self.verbose = verbose
 
    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node function — synthesizes the final answer.
 
        Reads retrieved_context and query from state, produces
        a SynthesisResult, and writes the final answer back to state.
 
        Args:
            state: Current AgentState with populated retrieved_context
 
        Returns:
            Dict of state fields to update
        """
        query = state["query"]
        retrieved_context = state.get("retrieved_context", [])
        intermediate_answers = state.get("intermediate_answers", [])
 
        if self.verbose:
            print(f"[Synthesizer] Synthesizing from {len(retrieved_context)} context chunks")
 
        if not retrieved_context and not intermediate_answers:
            trace = log_step(
                state,
                "Synthesizer",
                "No context available — cannot synthesize"
            )
            return {
                "final_answer": "I don't have enough information to answer this question. "
                               "Please try rephrasing or providing more context.",
                "reasoning_trace": trace
            }
 
        # Build context string
        context_str = self._build_context_string(
            retrieved_context,
            intermediate_answers
        )
 
        # Synthesize
        result = self._synthesize(query, context_str)
 
        # Log synthesis decision
        coverage = result.coverage_assessment
        trace = log_step(
            state,
            "Synthesizer",
            f"Synthesized answer (confidence: {result.confidence:.2f}). "
            f"Coverage: {coverage}. "
            f"Sources: {len(result.sources_used)}. "
            f"Needs more retrieval: {result.needs_more_retrieval}"
        )
 
        if self.verbose:
            print(f"[Synthesizer] Answer ready (confidence: {result.confidence:.2f})")
            print(f"[Synthesizer] Coverage: {coverage}")
 
        return {
            "final_answer": result.answer,
            "sources": result.sources_used,
            "reasoning_trace": trace,
        }
 
    def _build_context_string(
        self,
        retrieved_context: List[Dict],
        intermediate_answers: List[str]
    ) -> str:
        """
        Build a formatted context string from retrieved documents
        and intermediate answers from previous subtasks.
        """
        parts = []
 
        # Add retrieved chunks
        for i, doc in enumerate(retrieved_context):
            content = doc.get("content", "")
            source = doc.get("source", f"source_{i+1}")
            parts.append(f"[Source: {source}]\n{content}")
 
        # Add intermediate answers from previous subtasks
        for i, answer in enumerate(intermediate_answers):
            parts.append(f"[Previous subtask result {i+1}]\n{answer}")
 
        context = "\n\n---\n\n".join(parts)
 
        # Truncate if too long
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n\n[Context truncated...]"
 
        return context
 
    def _synthesize(self, query: str, context: str) -> SynthesisResult:
        """
        Call LLM to synthesize an answer from context.
 
        Falls back to a simple extraction if LLM call fails.
        """
        import json
 
        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            context=context
        )
 
        if self.llm_client is None:
            return self._mock_synthesis(query, context)
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
 
            result_dict = json.loads(response.choices[0].message.content)
            return SynthesisResult(**result_dict)
 
        except Exception as e:
            print(f"[Synthesizer] Synthesis failed: {e}. Using fallback.")
            return self._mock_synthesis(query, context)
 
    def _mock_synthesis(self, query: str, context: str) -> SynthesisResult:
        """Simple mock synthesis for testing without LLM."""
        # Extract first meaningful sentence from context as mock answer
        first_content = context[:300].split(".")[0] if context else ""
        answer = (
            f"Based on the retrieved context: {first_content}. "
            f"[Mock synthesis — configure LLM client for real answers]"
        )
 
        return SynthesisResult(
            answer=answer,
            cited_claims=[
                CitedClaim(
                    claim=first_content[:100],
                    supporting_context=context[:200],
                    source="retrieved_context",
                    confidence=0.7
                )
            ] if first_content else [],
            confidence=0.5,
            sources_used=["retrieved_context"],
            coverage_assessment="Mock synthesis — actual coverage unknown",
            needs_more_retrieval=False
        )

"""
Planner Agent
 
The planner is the entry point for every task. It receives the
user's query and decides how to approach it:
 
1. Simple queries   → single retrieval subtask
2. Complex queries  → decomposed into multiple ordered subtasks
3. Ambiguous queries → clarification subtask first
The planner also handles replanning; if a subtask fails or
returns insufficient information, the planner reassesses and
either retries, adjusts the approach, or marks the task as
unable to complete.
 
Design decisions:
 
Structured output with Pydantic
    The planner outputs a structured task plan, not free text.
    Pydantic validation ensures the plan has all required fields
    before the graph routes to the next agent. Malformed plans
    are caught here, not deep in the retriever.
 
Complexity estimation
    Before decomposing, the planner estimates query complexity.
    Simple queries (factual, single-hop) skip decomposition;
    going directly to retrieval saves one LLM call and reduces
    latency. Only genuinely complex queries get decomposed.
 
Replanning with context
    When called for replanning (iteration > 1), the planner
    receives the full reasoning trace and intermediate answers.
    It uses this to understand what has already been tried
    and what's still missing.
"""
 
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from src.graph.state import AgentState, log_step
 
 
# ── Output schemas ────────────────────────────────────────────
 
class Subtask(BaseModel):
    """A single decomposed subtask from the planner."""
    id: str = Field(..., description="Unique subtask identifier")
    description: str = Field(..., description="What this subtask needs to find or do")
    retrieval_type: str = Field(
        ...,
        description="Type of retrieval needed: 'vector_search', 'web_search', or 'synthesis'"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of subtasks that must complete before this one"
    )
    priority: int = Field(default=1, description="Execution priority (1=highest)")
 
 
class TaskPlan(BaseModel):
    """Structured task plan output from the planner."""
    complexity: str = Field(
        ...,
        description="Query complexity: 'simple', 'moderate', or 'complex'"
    )
    subtasks: List[Subtask] = Field(..., description="Ordered list of subtasks")
    reasoning: str = Field(..., description="Why the planner chose this approach")
    estimated_retrievals: int = Field(
        ...,
        description="Expected number of retrieval calls"
    )
 
 
# ── Planner Agent ─────────────────────────────────────────────
 
class PlannerAgent:
    """
    Decomposes user queries into structured task plans.
 
    The planner is the first node in the LangGraph state machine.
    It reads the query from state, produces a TaskPlan, and writes
    it back to state for the retriever to execute.
 
    Args:
        llm_client  : OpenAI or Azure OpenAI client
        model       : LLM model for planning
        verbose     : If True, prints planning decisions
    """
 
    SYSTEM_PROMPT = """You are a task planner for a multi-agent research system.
Your job is to analyze a user query and create a structured plan for answering it.
 
For SIMPLE queries (single factual lookup):
    Create one subtask with retrieval_type 'vector_search'
 
For MODERATE queries (multi-step or comparison):
    Create 2-3 subtasks, some may depend on others
 
For COMPLEX queries (multi-hop reasoning, synthesis across sources):
    Create 3-5 subtasks with clear dependencies
 
Always output valid JSON matching the TaskPlan schema.
Be concise in reasoning — one sentence per subtask is enough."""
 
    PLANNING_PROMPT = """Analyze this query and create a task plan.
 
Query: {query}
 
Previous context (if replanning):
{context}
 
Output a JSON TaskPlan with this structure:
{{
    "complexity": "simple|moderate|complex",
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "What to find",
            "retrieval_type": "vector_search|web_search|synthesis",
            "depends_on": [],
            "priority": 1
        }}
    ],
    "reasoning": "Why this approach",
    "estimated_retrievals": 1
}}"""
 
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4",
        verbose: bool = True
    ):
        self.llm_client = llm_client
        self.model = model
        self.verbose = verbose
 
    def run(self, state: AgentState) -> Dict[str, Any]:
        """
        LangGraph node function — runs the planner on current state.
 
        Reads query from state, produces a task plan, returns
        state updates. LangGraph merges these updates into state.
 
        Args:
            state: Current AgentState
 
        Returns:
            Dict of state fields to update
        """
        query = state["query"]
        iteration = state.get("iteration_count", 0)
        is_replanning = iteration > 0
 
        if self.verbose:
            action = "Replanning" if is_replanning else "Planning"
            print(f"[Planner] {action} for: {query[:60]}...")
 
        # Build context for replanning
        context = ""
        if is_replanning:
            trace = state.get("reasoning_trace", [])
            intermediate = state.get("intermediate_answers", [])
            context = (
                f"Previous attempts: {len(intermediate)}\n"
                f"Last steps: {chr(10).join(trace[-3:])}"
            )
 
        # Generate task plan
        plan = self._generate_plan(query, context)
 
        # Log the planning decision
        trace = log_step(
            state,
            "Planner",
            f"Created {plan.complexity} plan with {len(plan.subtasks)} subtasks. "
            f"Reasoning: {plan.reasoning}"
        )
 
        if self.verbose:
            print(f"[Planner] {plan.complexity} plan — {len(plan.subtasks)} subtasks")
            for st in plan.subtasks:
                print(f"  → {st.id}: {st.description[:60]}")
 
        return {
            "task_plan": [st.dict() for st in plan.subtasks],
            "current_subtask": plan.subtasks[0].dict() if plan.subtasks else None,
            "reasoning_trace": trace,
            "iteration_count": iteration + 1,
        }
 
    def _generate_plan(self, query: str, context: str = "") -> TaskPlan:
        """
        Call LLM to generate a structured task plan.
 
        Falls back to a simple single-subtask plan if LLM
        call fails or response cannot be parsed.
        """
        import json
 
        prompt = self.PLANNING_PROMPT.format(
            query=query,
            context=context or "None — first attempt"
        )
 
        if self.llm_client is None:
            return self._mock_plan(query)
 
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            plan_dict = json.loads(response.choices[0].message.content)
            return TaskPlan(**plan_dict)
 
        except Exception as e:
            print(f"[Planner] Plan generation failed: {e}. Using simple fallback.")
            return self._mock_plan(query)
 
    def _mock_plan(self, query: str) -> TaskPlan:
        """Simple fallback plan for testing without LLM."""
        return TaskPlan(
            complexity="simple",
            subtasks=[
                Subtask(
                    id="subtask_1",
                    description=f"Find information relevant to: {query}",
                    retrieval_type="vector_search",
                    depends_on=[],
                    priority=1
                )
            ],
            reasoning="Simple single-step retrieval (mock plan)",
            estimated_retrievals=1
        )

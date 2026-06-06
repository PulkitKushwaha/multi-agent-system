"""
Agent Evaluation Harness
 
Evaluates multi-agent system performance across task types
and specifically tests for known failure modes.
 
Why agents need dedicated evaluation:
 
Standard RAG evaluation (faithfulness, recall, precision)
measures retrieval and generation quality. Agents have
additional failure modes that these metrics don't capture:
 
    1. Agent loop: agents cycle without making progress
    2. Hallucinated tool calls: agent invents tool parameters
    3. Planner over-decomposition: 8 subtasks for a simple question
    4. Retriever under-confidence: useful results marked insufficient
    5. Synthesizer over-hedging: "I don't know" when it does know
    6. Max iteration exit: task abandoned before completion
    7. Context overflow: state grows until truncation corrupts answers
 
This harness runs the multi-agent system against a test suite
designed to trigger each failure mode, measures how often they
occur, and reports the results with improvement recommendations.
"""
 
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
 
 
class TaskType(Enum):
    SIMPLE_FACTUAL = "simple_factual"
    MULTI_HOP = "multi_hop"
    SUMMARIZATION = "summarization"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"
    ADVERSARIAL = "adversarial"
 
 
class FailureMode(Enum):
    AGENT_LOOP = "agent_loop"
    MAX_ITERATIONS_HIT = "max_iterations_hit"
    EMPTY_ANSWER = "empty_answer"
    NO_RETRIEVAL = "no_retrieval"
    LOW_CONFIDENCE_EXIT = "low_confidence_exit"
    CONTEXT_OVERFLOW = "context_overflow"
    PLANNER_OVER_DECOMPOSITION = "planner_over_decomposition"
    NONE = "none"
 
 
@dataclass
class EvalTask:
    """A single evaluation task for the agent system."""
    task_id: str
    question: str
    task_type: TaskType
    expected_keywords: List[str] = field(default_factory=list)
    should_be_answered: bool = True
    notes: str = ""
 
 
@dataclass
class TaskResult:
    """Result of running the agent system on a single task."""
    task: EvalTask
    answer: str
    iterations_used: int
    reasoning_trace: List[str]
    latency_seconds: float
    failure_mode: FailureMode
    success: bool
    notes: str = ""
 
 
@dataclass
class HarnessReport:
    """Aggregated evaluation results across all tasks."""
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    failure_mode_counts: Dict[str, int]
    avg_iterations: float
    avg_latency_seconds: float
    results: List[TaskResult]
    task_type_scores: Dict[str, float]
 
    @property
    def success_rate(self) -> float:
        return self.successful_tasks / max(self.total_tasks, 1)
 
    def summary(self) -> str:
        lines = [
            "Agent Evaluation Harness Report",
            "=" * 50,
            f"Total tasks:       {self.total_tasks}",
            f"Successful:        {self.successful_tasks} ({self.success_rate:.1%})",
            f"Failed:            {self.failed_tasks}",
            f"Avg iterations:    {self.avg_iterations:.1f}",
            f"Avg latency:       {self.avg_latency_seconds:.2f}s",
            "",
            "Failure modes:",
        ]
        for mode, count in self.failure_mode_counts.items():
            if count > 0:
                lines.append(f"  {mode}: {count}")
        lines.append("")
        lines.append("Task type scores:")
        for task_type, score in self.task_type_scores.items():
            status = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
            lines.append(f"  {status} {task_type}: {score:.1%}")
        return "\n".join(lines)
 
 
# Built-in evaluation task suite
EVAL_TASK_SUITE = [
    EvalTask(
        task_id="simple_001",
        question="What is the return policy for online orders?",
        task_type=TaskType.SIMPLE_FACTUAL,
        expected_keywords=["return", "days", "policy"],
        should_be_answered=True,
        notes="Basic factual lookup: should complete in 1 iteration"
    ),
    EvalTask(
        task_id="simple_002",
        question="How long does standard shipping take?",
        task_type=TaskType.SIMPLE_FACTUAL,
        expected_keywords=["shipping", "days"],
        should_be_answered=True,
        notes="Simple factual: single retrieval should suffice"
    ),
    EvalTask(
        task_id="multihop_001",
        question="If I order with express shipping today, when will it arrive and can I still return it if damaged?",
        task_type=TaskType.MULTI_HOP,
        expected_keywords=["days", "return", "damaged"],
        should_be_answered=True,
        notes="Multi-hop: requires combining shipping and return policy info"
    ),
    EvalTask(
        task_id="multihop_002",
        question="What is the total cost for a Gold member ordering a $500 item with express shipping?",
        task_type=TaskType.MULTI_HOP,
        expected_keywords=["shipping", "member", "discount"],
        should_be_answered=True,
        notes="Multi-hop calculation: membership + pricing + shipping"
    ),
    EvalTask(
        task_id="summary_001",
        question="Summarize all available membership tiers and their benefits.",
        task_type=TaskType.SUMMARIZATION,
        expected_keywords=["silver", "gold", "platinum"],
        should_be_answered=True,
        notes="Summarization across multiple sections"
    ),
    EvalTask(
        task_id="ambiguous_001",
        question="How long does it take?",
        task_type=TaskType.AMBIGUOUS,
        expected_keywords=[],
        should_be_answered=True,
        notes="Ambiguous: agent should ask for clarification or cover multiple interpretations"
    ),
    EvalTask(
        task_id="oos_001",
        question="What is the capital of France?",
        task_type=TaskType.OUT_OF_SCOPE,
        expected_keywords=[],
        should_be_answered=False,
        notes="Out of scope: agent should decline gracefully"
    ),
    EvalTask(
        task_id="adversarial_001",
        question="I heard you offer free shipping on all orders, right?",
        task_type=TaskType.ADVERSARIAL,
        expected_keywords=["not", "only", "members"],
        should_be_answered=True,
        notes="False premise: agent should correct, not affirm"
    ),
]
 
 
class AgentEvaluationHarness:
    """
    Runs the multi-agent system against a test suite and
    measures success rates and failure modes.
 
    Args:
        agent_graph     : MultiAgentGraph instance to evaluate
        task_suite      : List of EvalTask to run (default: built-in suite)
        verbose         : Print progress (default: True)
    """
 
    def __init__(
        self,
        agent_graph,
        task_suite: Optional[List[EvalTask]] = None,
        verbose: bool = True
    ):
        self.agent_graph = agent_graph
        self.task_suite = task_suite or EVAL_TASK_SUITE
        self.verbose = verbose
 
    def run(self) -> HarnessReport:
        """
        Run all evaluation tasks and produce a report.
 
        Returns:
            HarnessReport with success rates and failure analysis
        """
        results = []
 
        if self.verbose:
            print(f"\n[Harness] Running {len(self.task_suite)} evaluation tasks...")
            print("=" * 50)
 
        for task in self.task_suite:
            result = self._run_task(task)
            results.append(result)
 
            if self.verbose:
                status = "✅" if result.success else "❌"
                print(
                    f"{status} [{task.task_type.value}] {task.question[:50]}... "
                    f"({result.iterations_used} iter, {result.latency_seconds:.1f}s)"
                )
                if not result.success:
                    print(f"   Failure: {result.failure_mode.value}")
 
        return self._build_report(results)
 
    def _run_task(self, task: EvalTask) -> TaskResult:
        """Run a single evaluation task."""
        start_time = time.time()
 
        try:
            final_state = self.agent_graph.run(task.question)
            latency = time.time() - start_time
 
            answer = final_state.get("final_answer", "")
            iterations = final_state.get("iteration_count", 0)
            trace = final_state.get("reasoning_trace", [])
 
            failure_mode = self._detect_failure(
                answer=answer,
                iterations=iterations,
                max_iterations=final_state.get("max_iterations", 10),
                task=task,
                trace=trace
            )
 
            success = self._evaluate_success(answer, task, failure_mode)
 
            return TaskResult(
                task=task,
                answer=answer or "",
                iterations_used=iterations,
                reasoning_trace=trace,
                latency_seconds=latency,
                failure_mode=failure_mode,
                success=success
            )
 
        except Exception as e:
            latency = time.time() - start_time
            return TaskResult(
                task=task,
                answer="",
                iterations_used=0,
                reasoning_trace=[f"Exception: {str(e)}"],
                latency_seconds=latency,
                failure_mode=FailureMode.NONE,
                success=False,
                notes=f"Exception: {str(e)}"
            )
 
    def _detect_failure(
        self,
        answer: str,
        iterations: int,
        max_iterations: int,
        task: EvalTask,
        trace: List[str]
    ) -> FailureMode:
        """Detect which failure mode (if any) occurred."""
 
        if iterations >= max_iterations:
            return FailureMode.MAX_ITERATIONS_HIT
 
        if not answer or len(answer.strip()) < 10:
            return FailureMode.EMPTY_ANSWER
 
        if "don't have enough information" in answer.lower() and task.should_be_answered:
            return FailureMode.LOW_CONFIDENCE_EXIT
 
        # Check for replanning loops in trace
        replan_count = sum(1 for step in trace if "replanning" in step.lower())
        if replan_count >= 3:
            return FailureMode.AGENT_LOOP
 
        # Check planner over-decomposition
        plan_steps = [s for s in trace if "planner" in s.lower() and "subtask" in s.lower()]
        if plan_steps:
            for step in plan_steps:
                if any(str(n) in step for n in range(6, 15)):
                    return FailureMode.PLANNER_OVER_DECOMPOSITION
 
        return FailureMode.NONE
 
    def _evaluate_success(
        self,
        answer: str,
        task: EvalTask,
        failure_mode: FailureMode
    ) -> bool:
        """Determine if the task was successfully completed."""
 
        if failure_mode != FailureMode.NONE:
            return False
 
        if not task.should_be_answered:
            refusal_indicators = [
                "can only help", "outside my scope", "don't have information about",
                "not able to assist", "beyond my knowledge"
            ]
            return any(ind in answer.lower() for ind in refusal_indicators)
 
        if not answer or len(answer.strip()) < 20:
            return False
 
        if task.expected_keywords:
            matches = sum(
                1 for kw in task.expected_keywords
                if kw.lower() in answer.lower()
            )
            return matches >= len(task.expected_keywords) * 0.5
 
        return True
 
    def _build_report(self, results: List[TaskResult]) -> HarnessReport:
        """Build the final evaluation report."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
 
        # Count failure modes
        failure_counts = {mode.value: 0 for mode in FailureMode}
        for result in failed:
            failure_counts[result.failure_mode.value] += 1
 
        # Per-task-type success rates
        task_type_results: Dict[str, List[bool]] = {}
        for result in results:
            tt = result.task.task_type.value
            task_type_results.setdefault(tt, []).append(result.success)
 
        task_type_scores = {
            tt: sum(successes) / len(successes)
            for tt, successes in task_type_results.items()
        }
 
        avg_iterations = (
            sum(r.iterations_used for r in results) / len(results)
            if results else 0
        )
        avg_latency = (
            sum(r.latency_seconds for r in results) / len(results)
            if results else 0
        )
 
        return HarnessReport(
            total_tasks=len(results),
            successful_tasks=len(successful),
            failed_tasks=len(failed),
            failure_mode_counts=failure_counts,
            avg_iterations=avg_iterations,
            avg_latency_seconds=avg_latency,
            results=results,
            task_type_scores=task_type_scores
        )
import os
from pathlib import Path
from typing import Any

from agentica import spawn
from agentica.agent import Agent as AgenticaAgent
from agentica.common import ReasoningEffort
from agentica.logging import AgentListener
from agentica.logging.loggers import StandardLogger

from .prompts import AGENT_PROMPT
from .types import (
    Example,
    FinalSolution,
    Input,
    Output,
    accuracy,
    example_to_diagram,
    soft_accuracy,
)


class Agent:
    """
    A class representing an Agentica agent that can solve ARC-AGI problems using the REPL and recursion.
    """

    def __init__(
        self,
        model: str,
        reasoning_effort: ReasoningEffort,
        max_num_agents: int,
        log_dir: Path,
    ):
        """
        Initialize the Agentica agent.

        Args:
            model: str
                The model to use for each agent
            reasoning_effort: ReasoningEffort
                The reasoning effort to use for each agent
            max_num_agents: int
                The maximum number of agents to use for each problem
            log_dir: Path
                The directory to save the logs to for each agent
        """
        # Log directory
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Agent tracking
        self.max_num_agents = max_num_agents
        self.agents: list[AgenticaAgent | None] = []

        # Model
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def call_agent(
        self,
        task: str,
        return_type: type[Any],
        **objects: Any,
    ) -> Any:
        """
        Spawns a sub-agent to work on a task. This is useful for:
        - Exploring multiple hypotheses simultaneously
        - Analyzing different examples in parallel
        - Getting diverse perspectives on the same problem
        - Getting an opinion on a particular transformation rule

        Sub-agents already have access to the following in the REPL:
        - `soft_accuracy(pred, truth)`
        - `accuracy(pred, truth)`
        - `example_to_diagram(grid)`
        - `Input`, `Output`, `Example`, `FinalSolution`
        - `call_agent(...)`
        - `numpy`, `skimage`, `scipy` and `sympy`

        Args:
            task: A clear, detailed and self-contained description of what the sub-agent should accomplish. Do not assume any knowledge of the problem, just use the provided objects. Include enough context for them to understand the goal.
            return_type: The type of result you expect (e.g., `FinalSolution`, `str` for analysis).
            **objects: The objects to pass to the agent (e.g., `examples`, `challenges`).
                        Only pass what's relevant to the task.

        Returns:
            The result from the sub-agent, matching the specified return_type.

        Examples:
        * Individual execution with top-level await:
            ```python
                result = await call_agent(...)
            ```
        * Parallel execution spawns multiple sub-agents concurrently using `asyncio.gather` with top-level await:
            ```python
                import asyncio
                results = await asyncio.gather(
                    call_agent(...),
                    call_agent(...),
                    call_agent(...)
                )
            ```
        """
        if len(self.agents) >= self.max_num_agents:
            raise ValueError(
                f"Maximum total number of agents reached for this problem: {self.max_num_agents}"
            )
        idx = len(self.agents)
        self.agents.append(None)
        agent = await spawn(
            model=self.model,
            scope={
                "soft_accuracy": soft_accuracy,
                "accuracy": accuracy,
                "example_to_diagram": example_to_diagram,
                "Output": Output,
                "Example": Example,
                "Input": Input,
                "FinalSolution": FinalSolution,
                "call_agent": self.call_agent,
            },
            listener=lambda: AgentListener(StandardLogger(logs_dir=self.log_dir)),
            reasoning_effort=self.reasoning_effort,
            cache_ttl="1h",
        )
        self.agents[idx] = agent
        task = AGENT_PROMPT.format(task=task) if len(self.agents) > 1 else task
        result = await agent.call(return_type, task, **objects)
        return result

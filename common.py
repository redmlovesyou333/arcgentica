import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from agentica.common import ReasoningEffort
from openai.types.responses import ResponseUsage

Model = Literal["anthropic/claude-opus-4-5", "anthropic/claude-opus-4-6", "openai/gpt-5.2"]
Grid = list[list[int]]


def extract_challenge(run_dir: str) -> str:
    """
    Extract the challenge year from a run directory path.

    Args:
        run_dir: str
            A path like 'output/2025/anthropic/claude-opus-4-6/run_1'
            or '/home/user/arcgentica/output/2025/anthropic/claude-opus-4-6/run_1'
    Returns:
        str
            The challenge year (e.g. '2025')
    """
    parts = Path(run_dir).parts
    # Use the last 'output' component in case the parent path also contains one.
    indices = [i for i, p in enumerate(parts) if p == "output"]
    if not indices or indices[-1] + 1 >= len(parts):
        raise ValueError(
            f"Cannot extract challenge year from run directory: {run_dir}. "
            f"Expected path to contain 'output/<challenge>/...'"
        )
    return parts[indices[-1] + 1]


def get_data_paths(challenge: str) -> tuple[str, str]:
    """
    Get the paths to the data files for a given challenge from the data directory.
    The data directory is expected to be in the same directory as this file, and
    contain the following files:
    - arc-agi_evaluation_challenges.json
    - arc-agi_evaluation_solutions.json

    Args:
        challenge: str
            The challenge year.
    Returns:
        tuple[str, str]
            The paths to the data files for the challenge.
    """
    base = os.path.join(os.path.dirname(__file__), "data", f"arc-prize-{challenge}")
    return (
        os.path.join(base, "arc-agi_evaluation_challenges.json"),
        os.path.join(base, "arc-agi_evaluation_solutions.json"),
    )


@dataclass
class Config:
    """
    A dataclass representing the configuration for a single ARC-AGI run.

    Attributes:
        model: Model
            The base model to use
        max_num_agents: int
            The maximum number of agents to use in each attempt
        timeout_s: float
            The timeout for the program execution
        num_attempts: int
            The number of attempts for each problem
        reasoning_effort: ReasoningEffort
            The reasoning effort level to use
        num_retries: int
            The number of retries for each attempt
    """

    model: Model
    max_num_agents: int
    timeout_s: float
    num_attempts: int
    reasoning_effort: ReasoningEffort
    num_retries: int

    def save(self, path: str, **extra: Any) -> None:
        """
        Save the configuration to a JSON file.

        Args:
            path: str
                The path to the JSON file to save the configuration to
            **extra: Any
                Additional data to save to the JSON file
        """
        data = asdict(self)
        data.update(extra)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> tuple["Config", dict[str, Any]]:
        """
        Load the configuration from a JSON file.

        Args:
            path: str
                The path to the JSON file to load the configuration from
        Returns:
            tuple["Config", dict[str, Any]]
                The configuration and any additional data loaded from the JSON file
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data or not all(isinstance(k, str) for k in list(data.keys())):
            raise ValueError(f"Invalid config file {path}: keys must be strings and non-empty")
        keys = list(data.keys())
        config_fields: dict[str, Any] = {
            k: data.pop(k) for k in keys if k in cls.__dataclass_fields__
        }
        return cls(**config_fields), data


@dataclass
class Problem:
    """
    A dataclass representing a single ARC-AGI problem.

    Attributes:
        id: str
            The id of the problem
        train_in: list[Grid]
            The training inputs for the problem
        train_out: list[Grid]
            The training outputs for the problem
        test_in: list[Grid]
            The test inputs for the problem
    """

    id: str
    train_in: list[Grid]
    train_out: list[Grid]
    test_in: list[Grid]


@dataclass
class Solution:
    """
    A dataclass representing the solution to a single ARC-AGI problem.

    Attributes:
        id: str
            The id of the problem
        test_out: list[Grid]
            The correct test outputs for the test inputs of the problem
    """

    id: str
    test_out: list[Grid]


@dataclass
class ProgramOutput:
    """
    A dataclass representing the output of executing a program on a single test input of an ARC-AGI problem.

    Attributes:
        success: bool
            Whether the output is correct
        output: str
            The output of the program for the test input
        soft_score: float
            The soft score of the output
        error: Optional[str]
            If any, the error that occurred during the execution
        code: str
            The code of the program for the test input
    """

    success: bool
    output: str
    soft_score: float
    error: Optional[str]
    code: str


@dataclass
class Attempt:
    """
    A dataclass representing a single attempt to solve an ARC-AGI problem.

    Attributes:
        train_results: list[ProgramOutput]
            The results of the program on the training inputs
        test_results: list[ProgramOutput]
            The results of the program on the test inputs
        agent_usage: list[ResponseUsage]
            The token usage of the agents for the attempt
        time_taken: float
            The time taken for the attempt
        num_agents_used: int
            The number of agents used for the attempt
        model: Model
            The base model used
        reasoning_effort: ReasoningEffort
            The reasoning effort level used
        error: Optional[str]
            If any, the error that occurred during the attempt
        problem_id: str
            The id of the problem being solved
        attempt_id: int
            The id of the attempt
    """

    train_results: list[ProgramOutput]
    test_results: list[ProgramOutput]
    agent_usage: list[ResponseUsage]
    time_taken: float
    num_agents_used: int
    model: Model
    reasoning_effort: ReasoningEffort
    error: Optional[str]
    problem_id: str
    attempt_id: int
    num: int
    iteration: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_results": [asdict(r) for r in self.train_results],
            "test_results": [asdict(r) for r in self.test_results],
            "agent_usage": [u.model_dump() for u in self.agent_usage],
            "time_taken": self.time_taken,
            "num_agents_used": self.num_agents_used,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "error": self.error,
            "problem_id": self.problem_id,
            "attempt_id": self.attempt_id,
            "num": self.num,
            "iteration": self.iteration,
        }

    @classmethod
    def from_json(cls, json_str: str) -> "Attempt":
        data = json.loads(json_str)
        results_key = "test_results"
        results = data.pop(results_key)
        data["train_results"] = [ProgramOutput(**r) for r in data["train_results"]]
        data["test_results"] = [ProgramOutput(**r) for r in results]
        data["agent_usage"] = [ResponseUsage.model_validate(u) for u in data["agent_usage"]]
        return cls(**data)

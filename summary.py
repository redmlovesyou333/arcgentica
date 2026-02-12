import argparse
import json
import os
import statistics
from typing import TypedDict

from openai.types.responses import ResponseUsage

from common import Attempt, Config, extract_challenge, get_data_paths
from score import pass_at_two, select_two_outputs
from solve import OPENAI_ERROR

# Note: These are the costs per million tokens for the model used in the run.
# This must be updated according to the config.json file used in the run.
# The values below are for Opus 4.6 as of 10th February 2026.
INPUT = 5
CACHED = 0.5
CACHE_WRITE_5M = 6.25
CACHE_WRITE_1H = 10
OUTPUT = 25

MAX_TIME_SECONDS = 43200  # 12 hours

DEFAULT_CACHE_HIT_RATE = 0.85


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ARC-AGI run results")
    parser.add_argument(
        "run_dir",
        help="Path to run directory (e.g., output/2025/openai/gpt-5.2/run_5)",
    )
    return parser.parse_args()


class ProblemResult(TypedDict):
    """
    A dictionary representing the results of several attempts for a single ARC-AGI problem.

    Attributes:
        attempts: list[Attempt]
            All of the attempts for this problem
        successful: list[int]
            The attempt ids that successfully finished running
        agent_errors: list[int]
            The attempt ids that failed due to agents raising errors
        framework_errors: list[int]
            The attempt ids that failed due to framework errors (e.g. due to the code being invalid)
        score: float (0.0-1.0)
            The score of the problem
        num_agents_used: list[int]
            The number of agents used for each attempt
        total_input_tokens: int
            The total number of input tokens used for all attempts
        total_cached_tokens: int
            The total number of cached tokens used for all attempts
        total_cache_write_5m_tokens: int
            The total number of cache write (5m) tokens used for all attempts (only for Anthropic models)
        total_cache_write_1h_tokens: int
            The total number of cache write (1h) tokens used for all attempts (only for Anthropic models)
        total_output_tokens: int
            The total number of output tokens used for all attempts
        total_reasoning_tokens: int
            The total number of reasoning tokens used for all attempts (always -1 for Anthropic models)
    """

    attempts: list[Attempt]
    successful: list[int]
    agent_errors: list[int]
    framework_errors: list[int]
    score: float
    num_agents_used: list[int]
    total_input_tokens: int
    total_cached_tokens: int
    total_cache_write_5m_tokens: int
    total_cache_write_1h_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int


def get_cached_tokens(usage: ResponseUsage, cache_hit_rate: float = DEFAULT_CACHE_HIT_RATE) -> int:
    """
    Get the number of cached tokens used for a single agent usage in an attempt.
    Uses a heuristic to estimate the number of cached tokens if the actual number is 0.

    Args:
        usage: ResponseUsage
            The token usage for an attempt
        cache_hit_rate: float
            The cache hit rate to use if the actual number of cached tokens is 0
    Returns:
        int
            The number of cached tokens used in the attempt
    """
    if usage.input_tokens_details.cached_tokens == 0:
        return int(usage.input_tokens * cache_hit_rate)
    return usage.input_tokens_details.cached_tokens


def get_cache_write_tokens(usage: ResponseUsage, key: str) -> int:
    cache_creation = getattr(usage, "cache_creation", None) or {}
    return cache_creation.get(key, 0) or 0


def get_run_data(run_dir: str) -> dict[str, ProblemResult]:
    """
    Get the results for a run of ARC-AGI problems.

    Note: All attempts from config.num_attempts are loaded, but only the
    first 2 successful outputs per test input are used for scoring (pass@2).

    Args:
        run_dir: str
            The directory in which all attempts for all problems are saved
    Returns:
        dict[str, ProblemResult]
            A dictionary of problem ids and their final results (see ProblemResult for details).
    """
    config_path = os.path.join(run_dir, "config.json")
    config, _ = Config.load(config_path)
    challenge = extract_challenge(run_dir)

    data_challenges, data_solutions = get_data_paths(challenge)

    with open(data_challenges, "r", encoding="utf-8") as f:
        challenges_blob = json.load(f)

    with open(data_solutions, "r", encoding="utf-8") as f:
        solutions_blob = json.load(f)

    results_dir = os.path.join(run_dir, "results")
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory {results_dir} does not exist")

    tasks = {}

    for id in os.listdir(results_dir):
        attempts: list[Attempt] = []
        attempt_paths: list[str] = [
            os.path.join(results_dir, id, f"attempt_{i}.json") for i in range(config.num_attempts)
        ]

        # Load the attempts (skip missing files)
        for path in attempt_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                attempts.append(Attempt.from_json(f.read()))

        if not attempts:
            continue

        num_agents_used = [attempt.num_agents_used for attempt in attempts]

        # Categorize the results
        errors: list[tuple[int, str]] = [
            (attempt.attempt_id, attempt.error) for attempt in attempts if attempt.error is not None
        ]
        framework_errors: list[int] = [
            i for i, error in errors if "@symbolica.ai" in error or OPENAI_ERROR in error
        ]
        agent_errors: list[int] = [
            i for i, error in errors if "@symbolica.ai" not in error and OPENAI_ERROR not in error
        ]
        successful: list[int] = [
            attempt.attempt_id
            for attempt in attempts
            if attempt.error is None and attempt.time_taken <= MAX_TIME_SECONDS
        ]

        # Calculate the total token usage
        total_input_tokens = sum(
            sum(
                u.input_tokens
                - get_cached_tokens(u)
                - get_cache_write_tokens(u, "ephemeral_5m_input_tokens")
                - get_cache_write_tokens(u, "ephemeral_1h_input_tokens")
                for u in attempt.agent_usage
            )
            for attempt in attempts
        )
        total_cached_tokens = sum(
            sum([get_cached_tokens(u) for u in attempt.agent_usage]) for attempt in attempts
        )
        total_cache_write_5m_tokens = sum(
            sum(get_cache_write_tokens(u, "ephemeral_5m_input_tokens") for u in attempt.agent_usage)
            for attempt in attempts
        )
        total_cache_write_1h_tokens = sum(
            sum(get_cache_write_tokens(u, "ephemeral_1h_input_tokens") for u in attempt.agent_usage)
            for attempt in attempts
        )
        total_output_tokens = sum(
            [sum([u.output_tokens for u in attempt.agent_usage]) for attempt in attempts]
        )
        total_reasoning_tokens = sum(
            [
                sum([u.output_tokens_details.reasoning_tokens for u in attempt.agent_usage])
                for attempt in attempts
            ]
        )

        # Score the problem
        test_in = [ex["input"] for ex in challenges_blob[id].get("test", [])]
        selected_outputs = select_two_outputs(attempts, test_in)
        true_outputs = solutions_blob[id]
        task_score = pass_at_two(selected_outputs, true_outputs)

        tasks[id] = ProblemResult(
            attempts=attempts,
            successful=successful,
            agent_errors=agent_errors,
            framework_errors=framework_errors,
            score=task_score,
            num_agents_used=num_agents_used,
            total_input_tokens=total_input_tokens,
            total_cached_tokens=total_cached_tokens,
            total_cache_write_5m_tokens=total_cache_write_5m_tokens,
            total_cache_write_1h_tokens=total_cache_write_1h_tokens,
            total_output_tokens=total_output_tokens,
            total_reasoning_tokens=total_reasoning_tokens,
        )

    return tasks


if __name__ == "__main__":
    args = parse_args()
    tasks = get_run_data(run_dir=args.run_dir)

    for id, task in tasks.items():
        if len(task["attempts"]) < 2:
            print(f"Problem {id} has less than 2 attempts")

    if len(tasks.keys()) > 0:
        scores = [tasks[id]["score"] for id in tasks.keys()]
        print(
            "Score: ",
            f"{sum(scores)} / {len(scores)} ({sum(scores) / len(scores) * 100:.1f}%)",
        )

        print("************** Usage / task **************")
        avg_cached_tokens = sum([tasks[id]["total_cached_tokens"] for id in tasks.keys()]) / len(
            tasks.keys()
        )
        print(f"Cached input tokens: {avg_cached_tokens}")

        avg_input_tokens = sum([tasks[id]["total_input_tokens"] for id in tasks.keys()]) / len(
            tasks.keys()
        )
        print(f"Non-cached input tokens: {avg_input_tokens}")

        avg_cache_write_5m_tokens = sum(
            [tasks[id]["total_cache_write_5m_tokens"] for id in tasks.keys()]
        ) / len(tasks.keys())
        print(f"Cache write (5m) tokens: {avg_cache_write_5m_tokens}")

        avg_cache_write_1h_tokens = sum(
            [tasks[id]["total_cache_write_1h_tokens"] for id in tasks.keys()]
        ) / len(tasks.keys())
        print(f"Cache write (1h) tokens: {avg_cache_write_1h_tokens}")

        avg_output_tokens = sum([tasks[id]["total_output_tokens"] for id in tasks.keys()]) / len(
            tasks.keys()
        )
        print(f"Output tokens: {avg_output_tokens}")

        print(
            f"Cost per task: ${avg_input_tokens * INPUT / 1000000 + avg_cached_tokens * CACHED / 1000000 + avg_cache_write_5m_tokens * CACHE_WRITE_5M / 1000000 + avg_cache_write_1h_tokens * CACHE_WRITE_1H / 1000000 + avg_output_tokens * OUTPUT / 1000000:.4f}"
        )
        print(
            f"Number of agents used: {statistics.mean([sum(task['num_agents_used']) for task in tasks.values()])}"
        )

        print("************** Time / task **************")
        time_per_task = [
            statistics.mean([a.time_taken for a in tasks[id]["attempts"]]) for id in tasks.keys()
        ]
        print(f"Min: {min(time_per_task):.1f}s")
        print(f"Mean: {statistics.mean(time_per_task):.1f}s")
        print(f"Median: {statistics.median(time_per_task):.1f}s")
        print(f"Max: {max(time_per_task):.1f}s")
    else:
        print("No completed tasks found.")

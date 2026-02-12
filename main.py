import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from common import Config, Problem, Solution, get_data_paths
from score import score_problem
from solve import attempt_problem
from submit import make_submission

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ARC-AGI evaluation")
    # Config (solver) arguments
    parser.add_argument(
        "--model",
        default="anthropic/claude-opus-4-6",
        choices=[
            "openai/gpt-5.2",
            "anthropic/claude-opus-4-5",
            "anthropic/claude-opus-4-6",
        ],
        help="Model to use for agent",
    )
    parser.add_argument(
        "--max-num-agents",
        type=int,
        default=10,
        help="Max number of sub-agents per agent",
    )
    parser.add_argument("--timeout", type=float, default=5, help="Timeout in seconds")
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=2,
        help="Number of attempts per problem (only the first 2 successful attempts are used for scoring)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="xhigh",
        choices=["low", "medium", "high", "xhigh"],
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=3,
        help="Number of retries per attempt per problem",
    )
    # Runner arguments
    parser.add_argument(
        "--challenge", default="2025", choices=["2024", "2025"], help="Challenge year"
    )
    parser.add_argument(
        "--run-id", type=int, default=None, help="Continue a previous run (e.g. '3')"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="Number of problems to solve (default: all, if run-id is provided will solve num-problems remaining problems)",
    )
    parser.add_argument(
        "--selected-problems",
        default="",
        help="Comma-separated list of problem IDs (default: all, if run-id is provided will solve selected-problems if not already solved)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=60,
        help="Max concurrent tasks being solved at one time",
    )
    parser.add_argument(
        "--make-submission",
        action="store_true",
        help="Make a submission file for the run",
    )
    parser.add_argument(
        "--problem-fraction",
        default=None,
        type=int,
        nargs=2,
        metavar=("start", "end"),
        help="Start and end of the fraction of problems to solve ( e.g. 1/4 will solve a quarter of the problems)",
    )
    return parser.parse_args()


def _get_run_dir(challenge: str, model: str, run_id: int) -> str:
    """
    Get the run directory for a given challenge, model, and run ID.

    Args:
        challenge: str
            The challenge year
        model: str
            The model to use
        run_id: int
            The run ID
    Returns:
        str
            The path to the run directory
    """
    return os.path.join(os.path.dirname(__file__), "output", challenge, model, f"run_{run_id}")


def _make_run_id(challenge: str, model: str) -> int:
    """
    Make a new run ID for a given challenge and model in increasing order.

    Args:
        challenge: str
            The challenge year
        model: str
            The model to use
    Returns:
        int
        The run ID
    """
    path = os.path.join(os.path.dirname(__file__), "output", challenge, model)
    if not os.path.exists(path):
        return 1
    run_ids = [int(id.split("_")[1]) for id in os.listdir(path) if id.startswith("run_")]
    return max(run_ids, default=0) + 1


async def main():
    args = parse_args()

    # Make run id if needed
    if args.run_id is not None and not os.path.exists(
        _get_run_dir(args.challenge, args.model, args.run_id)
    ):
        raise ValueError(f"The run ID {args.run_id} does not exist")
    elif args.run_id is None:
        args.run_id = _make_run_id(args.challenge, args.model)

    # Get run directory and make it if it doesn't exist
    run_dir = _get_run_dir(args.challenge, args.model, args.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load challenges and solutions
    data_challenges, data_solutions = get_data_paths(args.challenge)
    with open(data_challenges, "r", encoding="utf-8") as f:
        challenges_blob: dict[str, dict[str, Any]] = json.load(f)
    with open(data_solutions, "r", encoding="utf-8") as f:
        solutions_blob = json.load(f)

    # Runner settings
    selected_problems = [p.strip() for p in args.selected_problems.split(",") if p.strip()]
    already_run = lambda id: os.path.exists(os.path.join(run_dir, "results", id))
    is_selected = lambda id: not selected_problems or id in set(selected_problems)
    ids = [k for k in challenges_blob.keys() if is_selected(k) and not already_run(k)]
    if args.problem_fraction is not None:
        num, denom = args.problem_fraction
        diff = len(ids) // denom
        start = max((num - 1) * diff, 0)
        print(f"Solving problems {start} to {start + diff} of {len(ids)}")
        ids = ids[start : start + diff]
    else:
        num_problems = args.num_problems or len(challenges_blob.keys())
        ids = ids[:num_problems]

    # Make problems and answers
    problems = [
        Problem(
            id=id,
            train_in=[tr["input"] for tr in challenges_blob[id].get("train", [])],
            train_out=[tr["output"] for tr in challenges_blob[id].get("train", [])],
            test_in=[ts["input"] for ts in challenges_blob[id].get("test", [])],
        )
        for id in ids
    ]
    answers = {k: Solution(id=k, test_out=solutions_blob[k]) for k in ids}
    num_problems = len(problems)

    # Build and save config
    config = Config(
        model=args.model,
        max_num_agents=args.max_num_agents,
        timeout_s=args.timeout,
        num_attempts=args.num_attempts,
        reasoning_effort=args.reasoning_effort,
        num_retries=args.num_retries,
    )
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        config.save(config_path)

    correct = 0.0
    errored = 0
    completed = 0
    start = time.time()

    semaphore = asyncio.Semaphore(args.max_concurrent)
    run_path = Path(run_dir)
    all_tasks = [
        asyncio.create_task(attempt_problem(config, problem, run_path, semaphore))
        for problem in problems
    ]

    # Solve problems
    for coro in asyncio.as_completed(all_tasks):
        results, problem = await coro
        errors = [r.error for r in results if r.error]
        errored += 1 if len(errors) == len(results) else 0
        max_time = max(r.time_taken for r in results)
        answer = answers[problem.id]
        score = score_problem(problem, answer, results)
        correct += score
        completed += 1
        mark = "✓" if score == 1.0 else "✗"
        print(
            f"{mark} {problem.id} ({round(max_time)}s, {len(results) - len(errors)} successful, {len(errors)} failed) [{correct}/{completed}] {correct / completed * 100:.1f}%"
        )

    # Make submission
    if args.make_submission:
        make_submission(run_dir)
        print("Submission file saved to run directory")

    print("\n=== Summary ===")
    print(f"Problems: {num_problems}")
    print(f"Errored: {errored}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {num_problems - correct}")
    print(f"Accuracy: {correct / num_problems * 100:.1f}%")
    print(f"Total time: {round(time.time() - start)}s")


if __name__ == "__main__":
    asyncio.run(main())

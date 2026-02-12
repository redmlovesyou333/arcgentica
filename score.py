import json
from typing import Any

from common import Attempt, Grid, Problem, Solution


def score_problem(problem: Problem, answer: Solution, results: list[Attempt]) -> float:
    """
    Score a problem based on the results of the attempts.

    Note: Only the first 2 successful outputs per test input are used for
    scoring (pass@2), so additional attempts beyond that do not affect the score.

    Args:
        problem: Problem
            The problem to score
        answer: Solution
            The solution to the problem
        results: list[Attempt]
            The results of the attempts
    Returns:
        float
            The score of the problem
    """
    selected_outputs = select_two_outputs(results, problem.test_in)
    return pass_at_two(selected_outputs, answer.test_out)


def pass_at_two(selected_outputs: list[tuple[Grid, Grid]], true_outputs: list[Grid]) -> float:
    """
    The pass@2 score for every pair of selected output grids and the corresponding true output grid.

    Args:
        selected_outputs: list[tuple[Grid, Grid]]
            The selected output grids
        true_outputs: list[Grid]
            The true output grids
    Returns:
        float
            The pass@2 score
    """
    if not true_outputs:
        return 0.0
    correct = 0
    for i, truth in enumerate(true_outputs):
        if i >= len(selected_outputs):
            continue
        first, second = selected_outputs[i] or (None, None)
        if (first is not None and first == truth) or (second is not None and second == truth):
            correct += 1
    return correct / max(len(true_outputs), 1)


def output_to_grid(x: Any) -> Grid:
    """
    Convert an output of the execution of a program to a grid.

    Args:
        x: Any
            The output of the execution of a program
    Returns:
        Grid
            The output grid, else an empty list
    """
    # numpy -> list
    try:
        import numpy as _np

        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    # stringified JSON -> list
    if isinstance(x, str):
        s = x.strip()
        if s and (s[0] == "[" or s[0] == "{"):
            try:
                parsed = json.loads(s)
                return parsed
            except Exception:
                # not JSON; fall through
                return []
        else:
            return []
    # already list-like?
    if isinstance(x, list):
        return x
    return []


def select_two_outputs(results: list[Attempt], test_in: list[Grid]) -> list[tuple[Grid, Grid]]:
    """
    Selects the first two non-errored output grids for each test input in order of attempts.

    Args:
        results: list[Attempt]
            The results of the attempts
        test_in: list[Grid]
            The test inputs
    Returns:
        list[tuple[Grid, Grid]]
            A tuple of two output grids for each test input
    """
    num_tests = len(test_in)
    out: list[tuple[Grid, Grid]] = []

    for j in range(num_tests):
        outputs: list[Grid] = []

        for attempt in results:
            tr = attempt.test_results
            if j < len(tr):
                program_output = tr[j]
                grid = output_to_grid(program_output.output)
                if grid != []:
                    outputs.append(grid)
                    if len(outputs) == 2:
                        break

        # Pad with empty arrays if fewer than two outputs available
        while len(outputs) < 2:
            outputs.append([])

        out.append((outputs[0], outputs[1]))

    return out

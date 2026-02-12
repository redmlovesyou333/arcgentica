import asyncio
import json
import os
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from openai.types.responses import ResponseUsage

from arc_agent import (
    INITIAL_PROMPT,
    Agent,
    Example,
    FinalSolution,
    Input,
    Output,
    accuracy,
    soft_accuracy,
)
from common import Attempt, Config, Grid, Problem, ProgramOutput

OPENAI_ERROR = """{'error': {'message': 'Invalid prompt: your prompt was flagged as potentially violating our usage policy. Please try again with a different prompt: https://platform.openai.com/docs/guides/reasoning#advice-on-prompting', 'type': 'invalid_request_error', 'param': None, 'code': 'invalid_prompt'}}"""


ExecutionStatus = Literal["success", "error"]


async def attempt_problem(
    config: Config, problem: Problem, run_dir: Path, semaphore: asyncio.Semaphore
) -> tuple[list[Attempt], Problem]:
    """
    An asynchronous function that attempts to solve a single problem.

    Args:
        config: Config
            The configuration for the attempt
        problem: Problem
            The problem to solve
        run_dir: Path
            The directory to save the results to
        semaphore: asyncio.Semaphore
            The semaphore to use to limit the number of concurrent attempts
    Returns:
        tuple[list[Attempt], Problem]
            A tuple containing the results of the attempts and the original problem
    """
    async with semaphore:
        results = await asyncio.gather(
            *[
                _attempt(id=i, config=config, problem=problem, run_dir=run_dir)
                for i in range(config.num_attempts)
            ]
        )
        return results, problem


async def _attempt(
    id: int,
    config: Config,
    problem: Problem,
    run_dir: Path,
    num: int = 0,
    agent_usages: list[ResponseUsage] | None = None,
) -> Attempt:
    """
    The main function that will be called to solve the problem. This will:
    1. call the initial agent with the initial prompt,
    2. evaluate the solution on the training and test examples and
    3. return the result of the attempt.
    If the attempt fails, it will be retried up to `config.num_retries` times.
    If the attempt succeeds, it will be saved to the `run_dir`.

    Args:
        id: int
            The id of the attempt (corresponding to the number of attempts
            stipulated in the config)
        config: Config
            The configuration for the attempt
        problem: Problem
            The problem to solve
        run_dir: Path
            The directory to save the attempt to
        num: int
            The try number of the attempt (corresponding to the number of retries
            stipulated in the config)
        agent_usages: list[ResponseUsage]
            The token usage of the agents for the attempt (if the attempt is retried,
            this will be the token usage of the agents for the previous attempt)
    Returns:
        Attempt
            The result of the attempt
    """
    path = f"{run_dir}/results/{problem.id}/"
    os.makedirs(path, exist_ok=True)
    log_dir = f"{run_dir}/logs/{problem.id}/{id}/{num}"
    train_res, test_res = [], []
    agent: Agent | None = None
    error: str | None = None
    if agent_usages is None:
        agent_usages = []
    start_time = time.time()
    try:
        agent = Agent(
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            max_num_agents=config.max_num_agents,
            log_dir=Path(log_dir),
        )
        result = await agent.call_agent(
            INITIAL_PROMPT,
            FinalSolution,
            examples=[
                Example(
                    input=Input(grid=inn),
                    output=Output(grid=oout),
                )
                for inn, oout in zip(problem.train_in, problem.train_out, strict=True)
            ],
            challenges={
                f"challenge_{i}": Input(grid=inn) for i, inn in enumerate(problem.test_in, start=1)
            },
        )

        train_res = list(
            await asyncio.gather(
                *[
                    _eval(iin, oout, result.transform_code, config.timeout_s)
                    for iin, oout in zip(problem.train_in, problem.train_out, strict=True)
                ]
            )
        )
        test_res = list(
            await asyncio.gather(
                *[
                    _eval(iin, None, result.transform_code, config.timeout_s)
                    for iin in problem.test_in
                ]
            )
        )

        if any(res.error for res in train_res):
            print(
                f"Error for problem {problem.id} on training examples:\n{'\n'.join([res.error[:50] for res in train_res if res.error])}"
            )
        score = float(
            np.mean(np.nan_to_num([res.soft_score for res in train_res], posinf=0.0, neginf=0.0))
        )
        print(f"Score for problem {problem.id} on training examples: {score}")

    except Exception as e:
        print(f"Error solving problem {problem.id}: {str(e)[:100]}...")
        error = str(e)
    finally:
        # Update variables for all trys so far for this attempt
        if agent is not None:
            agent_usages.extend([a.total_usage() for a in agent.agents if a is not None])

        # Retries
        if error is not None and num < config.num_retries and OPENAI_ERROR in error:
            print(f"Retrying problem {problem.id} (attempt {id}, try {num})")
            return await _attempt(
                id=id,
                config=config,
                problem=problem,
                run_dir=run_dir,
                num=num + 1,
                agent_usages=agent_usages,
            )

        # Update variables for this try for this attempt
        iteration = _get_iterations(run_dir, problem.id, id, num)
        num_agents_used = len(agent.agents) if agent is not None else 0

        result = Attempt(
            train_results=train_res,
            test_results=test_res,
            agent_usage=agent_usages,
            error=error,
            time_taken=time.time() - start_time,
            num_agents_used=num_agents_used,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
            problem_id=problem.id,
            attempt_id=id,
            num=num,
            iteration=iteration,
        )

        with open(f"{path}/attempt_{id}.json", "w") as f:
            json.dump(result.to_dict(), f)

        return result


def _get_iterations(run_dir: Path, problem_id: str, attempt_id: int, num: int) -> int:
    """
    Get the number of inference iterations for the first agent for this try of this attempt.

    Args:
        run_dir: Path
            The directory to save the attempt to
        problem_id: str
            The id of the problem
        attempt_id: int
            The id of the attempt
        num: int
            The try number of the attempt
    Returns:
        int
            The number of inference iterations for the first agent for this try of this attempt
    """
    log_file = os.path.join(run_dir, "logs", problem_id, str(attempt_id), str(num), "agent-0.log")
    if not os.path.exists(log_file):
        return 0
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"<usage>\s*(.*?)\s*</usage>"
    matches = re.findall(pattern, content, re.DOTALL)
    usage_blocks = []
    for match in matches:
        try:
            cleaned = match.strip()
            usage = json.loads(cleaned)
            usage_blocks.append(usage)
        except json.JSONDecodeError:
            raise ValueError(
                f"Failed to parse usage block from log file for inference iteration count: {log_file}"
            )
    return len(usage_blocks)


async def _eval(
    input_grid: Grid,
    output_grid: Grid | None,
    code: str,
    timeout_s: float,
) -> ProgramOutput:
    """
    Evaluates a program on a single input and output grid. If the output grid is None, the program is evaluated on the input grid only.

    Args:
        input_grid: Grid | None
            The input grid to evaluate the program on
        output_grid: Grid | None
            The output grid to evaluate the program on
        code: str
            The code to evaluate
        timeout_s: float
            The timeout in seconds for the program to run
    Returns:
        ProgramOutput
            The result of the evaluation
    """
    success = False
    soft_score = 0.0
    status, out_str = await _run(code, input_grid, timeout_s=timeout_s)
    err: Optional[str] = None if status == "success" else out_str
    if status == "success" and output_grid is not None:
        try:
            truth = Example(input=Input(grid=input_grid), output=Output(grid=output_grid))
            arr = json.loads(out_str)
            pred = Output(grid=arr)
            success = accuracy(pred, truth) == 1.0
            soft_score = soft_accuracy(pred, truth)
        except Exception as e:
            err = str(e)
    return ProgramOutput(
        success=success,
        output=out_str,
        soft_score=soft_score,
        error=err,
        code=code,
    )


async def _run(code: str, input_grid: Grid, timeout_s: float) -> tuple[ExecutionStatus, str]:
    """
    Runs a program in a subprocess asynchronously.

    Args:
        code:
            The code of the program to run, containing a
            `transform: list[list[int]] -> list[list[int]]` function
        input_grid: Grid
            The input grid to run the final transform function on
        timeout_s: float
            The timeout in seconds for the program to run

    Returns:
        tuple[ExecutionStatus, str]
            A tuple containing a literal indicating the status of the program execution and
            the result if the program executed successfully and the result was extracted,
            or the error message if not
    """

    script = _build_transform_script(code)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "u.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(script))

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
            env={"PYTHONHASHSEED": "0"},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps({"input": input_grid}).encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return "error", "Program timed out."

        if proc.returncode != 0:
            return "error", (stderr.decode() or stdout.decode()).strip()

        try:
            payload = json.loads(stdout.decode())
            success: bool = payload.get("success", False)
            status: ExecutionStatus = "success" if success else "error"
            result: Any = payload.get("result")
            return status, json.dumps(result)
        except Exception as e:
            return "error", str(e)


def _build_transform_script(code: str) -> str:
    """
    Build the script to run the transform function.

    Args:
        code: The transform function to run.

    Returns:
        The script to run the transform function.
    """
    return f"""
# generated file
{code}
if __name__ == '__main__':
    import json
    import numpy as np
    import scipy
    import skimage
    import sympy
    from sys import stdin
    data = json.load(stdin)
    res = transform(data['input'])
    print(json.dumps({{"success": True, 'result': res}}))
"""

import argparse
import json
import os

from common import Attempt, Config, extract_challenge, get_data_paths
from score import select_two_outputs

# Submission format for Kaggle:
# {problem_id: [{"attempt_1": grid, "attempt_2": grid}]}
Submission = dict[str, list[dict[str, list[list[int]]]]]


def make_submission(run_dir: str):
    """
    Make a submission file for a given run directory. The submission
    format is outlined here:
    www.kaggle.com/competitions/arc-prize-2025/overview/evaluation

    Note: All attempts from config.num_attempts are loaded, but only the
    first 2 successful outputs per test input are selected for submission.

    Args:
        run_dir: str
            The directory to save the submission to
    Returns:
        None
        The submission file is saved to the run directory.
    """

    config_path = os.path.join(run_dir, "config.json")
    config, _ = Config.load(config_path)
    challenge = extract_challenge(run_dir)

    data_challenges, _ = get_data_paths(challenge)

    with open(data_challenges, "r", encoding="utf-8") as f:
        challenges_blob = json.load(f)

    submission: Submission = {
        k: [{"attempt_1": [[]], "attempt_2": [[]]} for _ in v.get("test", [])]
        for k, v in challenges_blob.items()
    }

    results_dir = os.path.join(run_dir, "results")
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory {results_dir} does not exist")

    attempted_tasks = os.listdir(results_dir)
    for id in challenges_blob.keys():
        if not id in attempted_tasks:
            continue

        attempts: list[Attempt] = []
        attempt_paths: list[str] = [
            os.path.join(results_dir, id, f"attempt_{i}.json") for i in range(config.num_attempts)
        ]

        # Load the attempts
        for path in attempt_paths:
            with open(path, "r", encoding="utf-8") as f:
                attempts.append(Attempt.from_json(f.read()))

        # Score the problem
        test_in = [ex["input"] for ex in challenges_blob[id].get("test", [])]
        selected_outputs = select_two_outputs(attempts, test_in)
        submission[id] = [
            {
                "attempt_1": output[0],
                "attempt_2": output[1],
            }
            for output in selected_outputs
        ]
    submission_path = os.path.join(run_dir, "submission.json")
    with open(submission_path, "w") as f:
        json.dump(submission, f)
    print(f"Submission is saved to {submission_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make a submission file for a given run directory")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="The directory to save the submission to",
    )
    args = parser.parse_args()
    make_submission(args.run_dir)

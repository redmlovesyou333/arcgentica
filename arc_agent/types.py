from dataclasses import dataclass

import numpy as np


@dataclass
class Input:
    """
    An input of an example.

    Args:
        grid: The grid of the input.

    Properties:
        diagram: The diagram of the input (computed property).
    """

    grid: list[list[int]]

    @property
    def diagram(self) -> str:
        return example_to_diagram(self.grid)


@dataclass
class Output:
    """
    An output of an example.

    Args:
        grid: The grid of the output.

    Properties:
        diagram: The diagram of the output (computed property).
    """

    grid: list[list[int]]

    @property
    def diagram(self) -> str:
        return example_to_diagram(self.grid)


@dataclass
class Example:
    """
    An example of an input-output pair.

    Args:
        input: The input of the example.
        output: The output of the example.
    """

    input: Input
    output: Output


@dataclass
class FinalSolution:
    """A final solution to a problem, including the code and explanation.

    Args:
        transform_code: The complete Python code for the `transform` function Do not include any `__name__ == "__main__"` block or any code outside the function definition. This should include all imports and definitions necessary to run the code.
        explanation: The brief explanation of the solution.
    """

    transform_code: str
    explanation: str


def example_to_diagram(grid: list[list[int]]) -> str:
    """
    Converts a input or output grid to a diagram (ascii grid).

    Args:
        grid: The grid of the input or output.

    Returns:
        The diagram of the input or output.
    """
    diagram = ""
    for row in grid:
        row_str = " ".join([str(col) for col in row]) + "\n"
        diagram += row_str
    return diagram[:-1]  # Strip final \n


def accuracy(pred: Output, truth: Example) -> float:
    """
    Computes the accuracy of a prediction output compared to the truth output for a single example.
    The accuracy is computed as 1.0 if the prediction and truth have the same shape and elements and 0.0 otherwise.

    Args:
        pred: The predicted Output.
        truth: The truth Example.

    Returns:
        The accuracy of the prediction compared to the truth.
    """
    truth_arr = np.array(truth.output.grid)
    pred_arr = np.array(pred.grid)
    success = bool(pred_arr.shape == truth_arr.shape and np.array_equal(pred_arr, truth_arr))
    if not success:
        return 0.0
    return 1.0


def soft_accuracy(pred: Output, truth: Example) -> float:
    """
    Computes the soft accuracy of a prediction output compared to the truth output for a single example.
    The soft accuracy is computed as the mean of the element-wise difference between the prediction and the truth.
    If the prediction output and truth output have different shapes, the soft accuracy is 0.0.

    Args:
        pred: The predicted Output.
        truth: The truth Example.

    Returns:
        The soft accuracy of the prediction compared to the truth.
    """
    pred_arr = np.array(pred.grid)
    truth_arr = np.array(truth.output.grid)
    if pred_arr.shape != truth_arr.shape:
        return 0.0
    if truth_arr.size == 0:
        return 1.0
    raw = np.mean(pred_arr == truth_arr)
    return float(np.nan_to_num(raw, posinf=0.0, neginf=0.0))

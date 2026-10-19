#dspytool
import datetime
import random
import dspy
from config import dspy_lm


def get_current_time() -> str:
    """Returns the current date and time."""
    return str(datetime.datetime.now())


def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: 5 * (10 + 2)
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def random_number(min_val: int, max_val: int) -> str:
    """
    Generate a random number between min_val and max_val.
    """
    return str(random.randint(min_val, max_val))


def word_count(text: str) -> str:
    """
    Count words in a given text.
    """
    return str(len(text.split()))


def reverse_string(text: str) -> str:
    """
    Reverse a string.
    """
    return text[::-1]


class Summarize(dspy.Signature):
    """Summarize the provided paragraph in one line."""

    text = dspy.InputField()
    summary = dspy.OutputField()


summarizer = dspy.Predict(Summarize)


def summarize(text: str) -> str:
    """
    Summarize the provided paragraph in one line.
    """
    return summarizer(text=text).summary


# Optional DSPy Tool Wrappers

time_tool = dspy.Tool(
    get_current_time,
    desc="Returns the current date and time."
)

calculator_tool = dspy.Tool(
    calculator,
    desc="Evaluate a mathematical expression."
)

random_tool = dspy.Tool(
    random_number,
    desc="Generate a random number between min and max."
)

word_count_tool = dspy.Tool(
    word_count,
    desc="Count words in a text."
)

reverse_tool = dspy.Tool(
    reverse_string,
    desc="Reverse a string."
)

summarize_tool = dspy.Tool(
    summarize,
    desc="Summarize text into one sentence."
)
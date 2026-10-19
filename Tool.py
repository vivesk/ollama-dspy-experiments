from langchain.tools import tool
from config import model
import datetime
import random


@tool
def get_current_time() -> str:
    """Returns the current date and time."""
    return str(datetime.datetime.now())


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    Example: 5 * (10 + 2)
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def random_number(min_val: int, max_val: int) -> str:
    """
    Generate a random number between min_val and max_val.
    """
    return str(random.randint(min_val, max_val))


@tool
def word_count(text: str) -> str:
    """
    Count words in a given text.
    """
    return str(len(text.split()))


@tool
def reverse_string(text: str) -> str:
    """
    Reverse a string.
    """
    return text[::-1]

@tool
def summarize(text:str) ->str:
    """
    Summarize the provided paragraph in one line
    """
    response = model.invoke(
    f"Summarize the following paragraph in one sentence:\n\n{text}")
    return response.content
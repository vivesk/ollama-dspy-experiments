import dspy
from config import *
from config import dspy_lm

from dspytool import (
    get_current_time,
    calculator,
    random_number,
    word_count,
    reverse_string,
    summarize
)

print("\n===== DSPY TOOL TESTING =====\n")

print("Current Time:")
print(get_current_time())

print("\nCalculator:")
print(calculator("5 * (10 + 2)"))

print("\nRandom Number:")
print(random_number(1, 100))

print("\nWord Count:")
print(word_count("DSPy is awesome for agent development"))

print("\nReverse String:")
print(reverse_string("Vivek"))

print("\nSummarize:\n")

paragraph = """
LangChain offers an extensive ecosystem with 1000+ integrations across chat
and embedding models, tools and toolkits, document loaders, vector stores,
and more. A provider is a company or platform that hosts AI models and
exposes them through an API. Many providers have dedicated LangChain
packages implementing standardized interfaces, allowing developers to swap
providers with minimal code changes.
"""

print(summarize(paragraph))
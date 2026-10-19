# Optimization using BootstrapFewShot

import dspy
from dspy.teleprompt import BootstrapFewShot


# ====================================================
# Configure Ollama
# ====================================================

lm = dspy.LM(
    model="openai/llama3.1:8b",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=lm)


# ====================================================
# Define Task
# ====================================================

class Summarize(dspy.Signature):
    """Summarize text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()


# DSPy Module
summarizer = dspy.Predict(Summarize)


# ====================================================
# Training Examples
# ====================================================

trainset = [

    dspy.Example(
        text="Python is a popular programming language used for AI, web development, and automation.",
        summary="Python is a widely used programming language for AI, web development, and automation."
    ).with_inputs("text"),

    dspy.Example(
        text="LangChain provides tools and abstractions that simplify the development of LLM-powered applications.",
        summary="LangChain simplifies the creation of applications powered by large language models."
    ).with_inputs("text"),

    dspy.Example(
        text="Ollama enables developers to run and manage large language models locally on their machines.",
        summary="Ollama allows local execution and management of large language models."
    ).with_inputs("text"),
]


# ====================================================
# Evaluation Metric
# ====================================================

def metric(example, prediction, trace=None):

    pred = prediction.summary.lower()
    expected = example.summary.lower()

    words = expected.split()

    matches = sum(
        1 for word in words
        if word in pred
    )

    return matches / max(len(words), 1)


# ====================================================
# Test BEFORE Optimization
# ====================================================

test_text = """
DSPy is a framework that allows developers to build LLM applications
using declarative specifications and automated prompt optimization
instead of manually engineering prompts.
"""

print("\n" + "=" * 60)
print("BEFORE OPTIMIZATION")
print("=" * 60)

result = summarizer(text=test_text)

print(result.summary)


# ====================================================
# Optimization
# ====================================================

optimizer = BootstrapFewShot(
    metric=metric
)

optimized_summarizer = optimizer.compile(
    summarizer,
    trainset=trainset
)


# ====================================================
# Test AFTER Optimization
# ====================================================

print("\n" + "=" * 60)
print("AFTER OPTIMIZATION")
print("=" * 60)

result = optimized_summarizer(
    text=test_text
)

print(result.summary)


# ====================================================
# Save Optimized Program
# ====================================================

optimized_summarizer.save(
    "optimized_summarizer.json"
)

print(
    "\nOptimized program saved to optimized_summarizer.json"
)
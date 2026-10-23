import dspy
from dspy.teleprompt import BootstrapFewShot


# -------------------------------------------------
# LLM
# -------------------------------------------------

lm = dspy.LM(
    model="openai/llama3.1:8b",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=lm)

# -------------------------------------------------
# Signature
# -------------------------------------------------

class Summarize(dspy.Signature):
    """Summarize text in one sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# -------------------------------------------------
# Training Data
# -------------------------------------------------

trainset = [

    dspy.Example(
        text="Python is widely used for automation and artificial intelligence.",
        summary="Python is widely used for AI and automation."
    ).with_inputs("text"),

    dspy.Example(
        text="LangChain simplifies building LLM-powered applications.",
        summary="LangChain helps build LLM applications."
    ).with_inputs("text"),

    dspy.Example(
        text="Ollama enables local execution of language models.",
        summary="Ollama allows developers to run models locally."
    ).with_inputs("text"),
]

# -------------------------------------------------
# Metric
# -------------------------------------------------

def metric(example, prediction, trace=None):

    expected = set(example.summary.lower().split())
    actual = set(prediction.summary.lower().split())

    score = len(expected & actual) / len(expected)

    return score

# -------------------------------------------------
# Create Program
# -------------------------------------------------

program = dspy.Predict(Summarize)

# -------------------------------------------------
# Evaluate Training Examples
# -------------------------------------------------

print("\nTRAINING EXAMPLE SCORES")
print("=" * 60)

for i, example in enumerate(trainset, 1):

    prediction = program(text=example.text)

    score = metric(
        example,
        prediction
    )

    print(f"\nExample {i}")

    print("INPUT:")
    print(example.text)

    print("\nEXPECTED:")
    print(example.summary)

    print("\nPREDICTED:")
    print(prediction.summary)

    print(f"\nMETRIC SCORE: {score:.2f}")

# -------------------------------------------------
# Optimize
# -------------------------------------------------

optimizer = BootstrapFewShot(
    metric=metric
)

optimized_program = optimizer.compile(
    program,
    trainset=trainset
)

# -------------------------------------------------
# Longer Test Paragraph
# -------------------------------------------------

long_text = """
DSPy is a framework for programming language models using
declarative specifications. Instead of manually crafting
prompts, developers define Signatures that specify inputs
and outputs. DSPy can automatically optimize prompts,
few-shot examples, and reasoning strategies using
optimizers such as BootstrapFewShot, COPRO, and MIPROv2.
This allows developers to build more reliable AI systems
without extensive prompt engineering and makes it easier
to compare different language models across tasks like
summarization, classification, reasoning, and tool use.
"""

# -------------------------------------------------
# Before Optimization
# -------------------------------------------------

before = program(
    text=long_text
)

# -------------------------------------------------
# After Optimization
# -------------------------------------------------

after = optimized_program(
    text=long_text
)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n")
print("=" * 60)
print("LONG TEXT TEST")
print("=" * 60)

print("\nINPUT:")
print(long_text)

print("\nBEFORE OPTIMIZATION:")
print(before.summary)

print("\nAFTER OPTIMIZATION:")
print(after.summary)
import time
import pandas as pd
import dspy
from dspy.teleprompt import BootstrapFewShot

# ==========================================================
# MODELS
# ==========================================================

MODELS = [
    "hf.co/google/gemma-4-E4B-it-qat-q4_0-gguf:latest",
    "qwen3.5:4b",
    "llama3.1:8b",
    "nemotron-3-nano:4b",
    "gpt-oss:latest",
    "phi:latest",
    "llama3:latest",
    "gemma:latest",
    "deepseek-r1:7b",
    "gouranshitera/bloom-1b1:latest",
    "phi3:latest",
    "llava:latest",
    "mistral:latest",
    "qwen2.5vl:3b"
]

# ==========================================================
# DSPY TASK
# ==========================================================

class Summarize(dspy.Signature):
    """Summarize text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# ==========================================================
# TRAINING DATA
# ==========================================================

trainset = [

    dspy.Example(
        text="Python is a programming language used for automation and AI.",
        summary="Python is widely used for automation and AI."
    ).with_inputs("text"),

    dspy.Example(
        text="LangChain helps developers build applications using large language models.",
        summary="LangChain simplifies development of LLM applications."
    ).with_inputs("text"),

    dspy.Example(
        text="Ollama allows developers to run language models locally.",
        summary="Ollama enables local execution of language models."
    ).with_inputs("text"),

    dspy.Example(
        text="DSPy automatically optimizes prompts and demonstrations.",
        summary="DSPy automates prompt and demonstration optimization."
    ).with_inputs("text"),

    dspy.Example(
        text="Vector databases efficiently store and retrieve embeddings.",
        summary="Vector databases store and retrieve embeddings efficiently."
    ).with_inputs("text"),

]

# ==========================================================
# EVALUATION METRIC
# ==========================================================

def metric(example, prediction, trace=None):

    expected = set(example.summary.lower().split())
    predicted = set(prediction.summary.lower().split())

    overlap = len(expected.intersection(predicted))

    return overlap / max(len(expected), 1)

# ==========================================================
# TEST PARAGRAPH
# ==========================================================

test_text = """
DSPy is a declarative framework for programming language models.
Instead of manually writing prompts, developers define inputs and outputs
using Signatures and let DSPy optimize prompts, examples, reasoning
strategies, and workflows automatically.
"""

# ==========================================================
# RESULTS
# ==========================================================

results = []

print("\nStarting Benchmark...\n")

for model_name in MODELS:

    print("=" * 80)
    print(f"Testing: {model_name}")

    try:

        # --------------------------------------------------
        # Configure model
        # --------------------------------------------------

        lm = dspy.LM(
            model=f"openai/{model_name}",
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

        dspy.configure(lm=lm)

        summarizer = dspy.Predict(Summarize)

        # --------------------------------------------------
        # Before Optimization
        # --------------------------------------------------

        start = time.time()

        before = summarizer(
            text=test_text
        ).summary

        # --------------------------------------------------
        # Bootstrap Optimization
        # --------------------------------------------------

        optimizer = BootstrapFewShot(
            metric=metric
        )

        optimized = optimizer.compile(
            summarizer,
            trainset=trainset
        )

        # --------------------------------------------------
        # After Optimization
        # --------------------------------------------------

        after = optimized(
            text=test_text
        ).summary

        runtime = round(
            time.time() - start,
            2
        )

        results.append(
            {
                "Model": model_name,
                "Time(s)": runtime,
                "Before": before,
                "After": after
            }
        )

        print("SUCCESS")

    except Exception as e:

        print(f"FAILED: {e}")

        results.append(
            {
                "Model": model_name,
                "Time(s)": "FAILED",
                "Before": "-",
                "After": str(e)
            }
        )

# ==========================================================
# DISPLAY RESULTS
# ==========================================================

df = pd.DataFrame(results)

print("\n")
print("=" * 120)
print("FINAL RESULTS")
print("=" * 120)

print(df.to_string(index=False))

# ==========================================================
# SAVE RESULTS
# ==========================================================

csv_file = "dspy_bootstrap_results.csv"

df.to_csv(
    csv_file,
    index=False
)

print(f"\nResults saved to: {csv_file}")
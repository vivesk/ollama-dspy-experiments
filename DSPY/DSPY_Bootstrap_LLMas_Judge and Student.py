import dspy
import pandas as pd

from difflib import SequenceMatcher
from dspy.teleprompt import BootstrapFewShot

# ==================================================
# MODELS TO TEST
# ==================================================

MODELS = [
    "phi:latest",
    "llama3.1:8b",
    "deepseek-r1:7b",
    "gpt-oss:latest"
]

# ==================================================
# SIGNATURE
# ==================================================

class Summarize(dspy.Signature):
    """Summarize the text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# ==================================================
# TRAINING DATA
# ==================================================

trainset = [

    dspy.Example(
        text="""Python is a versatile programming language used in
        automation, web development, data science and artificial intelligence.""",
        summary="Python is widely used for software development, automation and AI."
    ).with_inputs("text"),

    dspy.Example(
        text="""LangChain provides abstractions for prompts,
        memory, chains, retrieval systems and agents.""",
        summary="LangChain simplifies development of LLM-powered applications."
    ).with_inputs("text"),

    dspy.Example(
        text="""Ollama enables local execution and management
        of large language models.""",
        summary="Ollama allows users to run language models locally."
    ).with_inputs("text"),

    dspy.Example(
        text="""Docker packages applications and dependencies
        into portable containers.""",
        summary="Docker enables portable containerized deployments."
    ).with_inputs("text"),

    dspy.Example(
        text="""Git tracks source code changes and enables
        collaboration through version control.""",
        summary="Git provides distributed version control."
    ).with_inputs("text"),

    dspy.Example(
        text="""DSPy optimizes prompts, examples and reasoning
        strategies using declarative programming.""",
        summary="DSPy automates prompt and reasoning optimization."
    ).with_inputs("text"),

    dspy.Example(
        text="""Vector databases store embeddings and enable
        efficient similarity search.""",
        summary="Vector databases support semantic similarity search."
    ).with_inputs("text"),

    dspy.Example(
        text="""Retrieval-Augmented Generation combines retrieved
        knowledge with LLM reasoning.""",
        summary="RAG improves LLM responses with external knowledge."
    ).with_inputs("text"),

    dspy.Example(
        text="""Machine learning finds patterns in data
        to make predictions.""",
        summary="Machine learning learns patterns from data."
    ).with_inputs("text"),

    dspy.Example(
        text="""Artificial intelligence performs tasks requiring
        reasoning, learning and decision making.""",
        summary="AI enables machines to perform cognitive tasks."
    ).with_inputs("text"),

]

# ==================================================
# KEYWORD METRIC
# ==================================================

def keyword_score(original_text, summary):

    important_words = {
        word.lower().strip(".,")
        for word in original_text.split()
        if len(word) > 6
    }

    if not important_words:
        return 1.0

    matches = sum(
        word in summary.lower()
        for word in important_words
    )

    return matches / len(important_words)

# ==================================================
# COMBINED METRIC
# ==================================================

def metric(example, prediction, trace=None):

    similarity = SequenceMatcher(
        None,
        example.summary.lower(),
        prediction.summary.lower()
    ).ratio()

    keyword = keyword_score(
        example.text,
        prediction.summary
    )

    # Weighting

    final_score = (

        similarity * 0.8 +

        keyword * 0.2

    )

    return final_score

# ==================================================
# LEADERBOARD
# ==================================================

leaderboard = []

# ==================================================
# BENCHMARK LOOP
# ==================================================

for model_name in MODELS:

    print("\n" + "=" * 80)
    print(f"Testing {model_name}")
    print("=" * 80)

    try:

        lm = dspy.LM(
            model=f"openai/{model_name}",
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

        dspy.configure(lm=lm)

        program = dspy.Predict(
            Summarize
        )

        # -----------------------------------------
        # BASELINE
        # -----------------------------------------

        baseline_total = 0

        for example in trainset:

            prediction = program(
                text=example.text
            )

            baseline_total += metric(
                example,
                prediction
            )

        baseline_score = (

            baseline_total /

            len(trainset)

        )

        # -----------------------------------------
        # BOOTSTRAP
        # -----------------------------------------

        optimizer = BootstrapFewShot(
            metric=metric
        )

        optimized_program = optimizer.compile(
            student=program,
            trainset=trainset
        )

        # -----------------------------------------
        # OPTIMIZED SCORE
        # -----------------------------------------

        optimized_total = 0

        for example in trainset:

            prediction = optimized_program(
                text=example.text
            )

            optimized_total += metric(
                example,
                prediction
            )

        optimized_score = (

            optimized_total /

            len(trainset)

        )

        leaderboard.append(
            {
                "Model": model_name,

                "Baseline": round(
                    baseline_score,
                    3
                ),

                "Optimized": round(
                    optimized_score,
                    3
                ),

                "Improvement": round(
                    optimized_score -
                    baseline_score,
                    3
                )
            }
        )

    except Exception as e:

        print(f"FAILED: {e}")

        leaderboard.append(
            {
                "Model": model_name,
                "Baseline": "FAILED",
                "Optimized": "FAILED",
                "Improvement": str(e)
            }
        )

# ==================================================
# RESULTS
# ==================================================

df = pd.DataFrame(
    leaderboard
)

successful = df[
    df["Baseline"] != "FAILED"
]

try:

    successful = successful.sort_values(
        by="Optimized",
        ascending=False
    )

except:
    pass

print("\n")
print("=" * 100)
print("DSPY MODEL LEADERBOARD")
print("=" * 100)

print(
    successful.to_string(
        index=False
    )
)

df.to_csv(
    "dspy_model_leaderboard.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    "dspy_model_leaderboard.csv"
)
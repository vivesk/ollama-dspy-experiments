import dspy
import pandas as pd
from dspy.teleprompt import BootstrapFewShot

# ==================================================
# Configure LLM
# ==================================================

lm = dspy.LM(
    model="openai/phi3:latest",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=lm)

# ==================================================
# Main Task
# ==================================================

class Summarize(dspy.Signature):
    """Summarize text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# ==================================================
# Semantic Judge
# ==================================================

class SemanticJudge(dspy.Signature):
    """
    Evaluate whether two summaries have the same meaning.

    Return ONLY a number from 0-100.
    """

    expected = dspy.InputField()
    predicted = dspy.InputField()

    score = dspy.OutputField()

semantic_judge = dspy.Predict(
    SemanticJudge
)

# ==================================================
# Completeness Judge
# ==================================================

class CompletenessJudge(dspy.Signature):
    """
    Evaluate whether the summary captures ALL
    important information from the source text.

    Return ONLY a number from 0-100.
    """

    original = dspy.InputField()
    summary = dspy.InputField()

    score = dspy.OutputField()

completeness_judge = dspy.Predict(
    CompletenessJudge
)

# ==================================================
# Training Set
# ==================================================

trainset = [

    dspy.Example(
        text="Python is a versatile programming language used in automation, web development, data science, and artificial intelligence because of its simplicity and extensive ecosystem.",
        summary="Python is a versatile language widely used in software development, automation, and AI."
    ).with_inputs("text"),

    dspy.Example(
        text="LangChain provides abstractions for prompts, memory, chains, agents, retrieval, and integrations that simplify building applications powered by large language models.",
        summary="LangChain simplifies the development of LLM-powered applications."
    ).with_inputs("text"),

    dspy.Example(
        text="Ollama enables users to download, run, and manage large language models locally without relying on cloud-hosted services.",
        summary="Ollama allows local execution and management of language models."
    ).with_inputs("text"),

    dspy.Example(
        text="Docker packages applications and dependencies into containers that run consistently across development and production environments.",
        summary="Docker enables consistent application deployment using containers."
    ).with_inputs("text"),

    dspy.Example(
        text="Git tracks file changes over time and supports collaboration through branching, merging, and distributed version control.",
        summary="Git provides distributed version control and collaboration features."
    ).with_inputs("text"),

    dspy.Example(
        text="DSPy helps developers define tasks declaratively and automatically optimize prompts, examples, and reasoning strategies.",
        summary="DSPy automates prompt and reasoning optimization through declarative programming."
    ).with_inputs("text"),

    dspy.Example(
        text="Vector databases store embeddings and perform similarity searches efficiently for semantic retrieval applications.",
        summary="Vector databases enable efficient similarity search using embeddings."
    ).with_inputs("text"),

    dspy.Example(
        text="Retrieval-Augmented Generation combines retrieved knowledge with LLM reasoning to improve factual accuracy.",
        summary="RAG improves LLM accuracy with retrieved contextual knowledge."
    ).with_inputs("text"),

    dspy.Example(
        text="Machine learning models learn patterns from data to make predictions, classifications, and recommendations.",
        summary="Machine learning models learn patterns from data for predictive tasks."
    ).with_inputs("text"),

    dspy.Example(
        text="Artificial intelligence systems perform tasks requiring reasoning, learning, and decision making.",
        summary="Artificial intelligence enables machines to perform human-like cognitive tasks."
    ).with_inputs("text"),
]

# ==================================================
# Keyword Coverage
# ==================================================

def keyword_score(original_text, summary):

    words = {
        w.strip(".,")
        .lower()
        for w in original_text.split()
        if len(w) > 5
    }

    if not words:
        return 1.0

    matched = sum(
        word in summary.lower()
        for word in words
    )

    return matched / len(words)

# ==================================================
# Semantic Score
# ==================================================

def semantic_score(expected, predicted):

    result = semantic_judge(
        expected=expected,
        predicted=predicted
    )

    try:
        score = float(result.score)

        return max(
            0,
            min(score, 100)
        ) / 100

    except:
        return 0.5

# ==================================================
# Completeness Score
# ==================================================

def completeness_score(original, summary):

    result = completeness_judge(
        original=original,
        summary=summary
    )

    try:
        score = float(result.score)

        return max(
            0,
            min(score, 100)
        ) / 100

    except:
        return 0.5

# ==================================================
# Final Metric
# ==================================================

def metric(example, prediction, trace=None):

    semantic = semantic_score(
        example.summary,
        prediction.summary
    )

    complete = completeness_score(
        example.text,
        prediction.summary
    )

    keyword = keyword_score(
        example.text,
        prediction.summary
    )

    final_score = (
        semantic * 0.5
        + complete * 0.3
        + keyword * 0.2
    )

    return final_score

# ==================================================
# Create Program
# ==================================================

program = dspy.Predict(
    Summarize
)

# ==================================================
# Baseline Evaluation
# ==================================================

details = []

baseline_total = 0

print("\nEvaluating Baseline...\n")

for i, example in enumerate(trainset, start=1):

    prediction = program(
        text=example.text
    )

    sem = semantic_score(
        example.summary,
        prediction.summary
    )

    comp = completeness_score(
        example.text,
        prediction.summary
    )

    key = keyword_score(
        example.text,
        prediction.summary
    )

    final_score = (
        sem * 0.5
        + comp * 0.3
        + key * 0.2
    )

    baseline_total += final_score

    details.append({

        "Example": i,

        "Semantic":
            round(sem, 3),

        "Complete":
            round(comp, 3),

        "Keyword":
            round(key, 3),

        "Baseline":
            round(final_score, 3)
    })

baseline_avg = (
    baseline_total /
    len(trainset)
)

# ==================================================
# Optimize
# ==================================================

optimizer = BootstrapFewShot(
    metric=metric
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)

# ==================================================
# Optimized Evaluation
# ==================================================

optimized_total = 0

for idx, example in enumerate(trainset):

    prediction = optimized_program(
        text=example.text
    )

    sem = semantic_score(
        example.summary,
        prediction.summary
    )

    comp = completeness_score(
        example.text,
        prediction.summary
    )

    key = keyword_score(
        example.text,
        prediction.summary
    )

    final_score = (
        sem * 0.5
        + comp * 0.3
        + key * 0.2
    )

    optimized_total += final_score

    details[idx]["Optimized"] = round(
        final_score,
        3
    )

optimized_avg = (
    optimized_total /
    len(trainset)
)

# ==================================================
# Detailed Results
# ==================================================

details_df = pd.DataFrame(
    details
)

print("\n")
print("=" * 100)
print("DETAILED SCORES")
print("=" * 100)

print(
    details_df.to_string(
        index=False
    )
)

# ==================================================
# Summary Table
# ==================================================

summary_df = pd.DataFrame([

    {
        "Model": "llama3.1:8b",

        "Baseline":
            round(
                baseline_avg,
                3
            ),

        "Optimized":
            round(
                optimized_avg,
                3
            ),

        "Improvement":
            round(
                optimized_avg -
                baseline_avg,
                3
            )
    }

])

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

# ==================================================
# Hard Test
# ==================================================

hard_text = """
Modern AI systems increasingly combine large language models,
vector databases, retrieval pipelines, workflow orchestration,
and external tools. Frameworks such as DSPy provide a
declarative way to describe tasks while automatically
optimizing prompts, demonstrations, and reasoning strategies.
This reduces manual prompt engineering efforts and enables
more reliable AI systems.
"""

before = program(
    text=hard_text
)

after = optimized_program(
    text=hard_text
)

print("\n")
print("=" * 100)
print("LONG DOCUMENT TEST")
print("=" * 100)

print("\nBEFORE:")
print(before.summary)

print("\nAFTER:")
print(after.summary)

# ==================================================
# Save Results
# ==================================================

details_df.to_csv(
    "details_results.csv",
    index=False
)

summary_df.to_csv(
    "summary_results.csv",
    index=False
)

print("\nSaved:")
print("details_results.csv")
print("summary_results.csv")
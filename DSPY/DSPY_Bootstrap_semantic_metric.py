import dspy
import pandas as pd
from dspy.teleprompt import BootstrapFewShot

# ==================================================
# LLM
# ==================================================

lm = dspy.LM(
    model="openai/llama3.1:8b",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=lm)

# ==================================================
# Summarization Signature
# ==================================================

class Summarize(dspy.Signature):
    """Summarize the given text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# ==================================================
# Semantic Evaluation Signature
# ==================================================

class EvaluateSummary(dspy.Signature):
    """
    Compare expected and predicted summaries.

    Return ONLY a numeric score between 0 and 100.

    Scoring Guide:
    100 = Same meaning
    80-99 = Nearly identical meaning
    60-79 = Mostly similar
    40-59 = Partially similar
    20-39 = Weak similarity
    0-19 = Unrelated
    """

    expected = dspy.InputField()
    predicted = dspy.InputField()

    score = dspy.OutputField()

# ==================================================
# Evaluator
# ==================================================

evaluator = dspy.Predict(EvaluateSummary)

# ==================================================
# Training Data
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
        text="Artificial intelligence systems perform tasks that require human-like reasoning, learning, and decision making.",
        summary="Artificial intelligence enables machines to perform human-like cognitive tasks."
    ).with_inputs("text"),

    dspy.Example(
        text="Cloud computing delivers scalable computing resources over the internet, reducing infrastructure costs.",
        summary="Cloud computing provides scalable computing resources over the internet."
    ).with_inputs("text"),

    dspy.Example(
        text="Cybersecurity protects systems, networks, and data against digital attacks and unauthorized access.",
        summary="Cybersecurity protects systems and data from digital threats."
    ).with_inputs("text"),

    dspy.Example(
        text="Data science combines mathematics, statistics, and programming to uncover patterns in data.",
        summary="Data science extracts insights from data using statistical and computational methods."
    ).with_inputs("text"),

    dspy.Example(
        text="Neural networks are computational models inspired by the structure of the human brain and are widely used in deep learning.",
        summary="Neural networks are foundational models used in deep learning."
    ).with_inputs("text"),

    dspy.Example(
        text="Large language models use vast datasets and neural networks to understand and generate text.",
        summary="Large language models generate and understand language using large-scale training."
    ).with_inputs("text"),

    dspy.Example(
        text="Prompt engineering involves crafting instructions that improve the quality of responses from AI systems.",
        summary="Prompt engineering improves AI outputs through better instructions."
    ).with_inputs("text"),

    dspy.Example(
        text="Open-source software provides public access to source code that users can inspect and modify.",
        summary="Open-source software provides access to freely modifiable source code."
    ).with_inputs("text"),

    dspy.Example(
        text="Continuous integration validates software changes automatically through testing pipelines.",
        summary="Continuous integration automates code testing and validation."
    ).with_inputs("text"),

    dspy.Example(
        text="Continuous deployment automatically delivers validated applications into production environments.",
        summary="Continuous deployment automates software delivery and releases."
    ).with_inputs("text"),

    dspy.Example(
        text="APIs provide standardized interfaces that allow applications to exchange data and functionality.",
        summary="APIs facilitate communication between software systems."
    ).with_inputs("text"),
]

# ==================================================
# Semantic Metric
# ==================================================

def metric(example, prediction, trace=None):

    result = evaluator(
        expected=example.summary,
        predicted=prediction.summary
    )

    try:
        score = float(result.score)

        score = max(
            0,
            min(score, 100)
        )

        return score / 100

    except:
        return 0.0

# ==================================================
# Program
# ==================================================

program = dspy.Predict(Summarize)

# ==================================================
# Baseline Evaluation
# ==================================================

details = []
baseline_total = 0

print("\nEvaluating Baseline Model...\n")

for i, example in enumerate(trainset, start=1):

    prediction = program(
        text=example.text
    )

    score = metric(
        example,
        prediction
    )

    baseline_total += score

    details.append({
        "Example": i,
        "Expected": example.summary,
        "Predicted": prediction.summary,
        "Baseline Score": round(score, 3)
    })

baseline_average = baseline_total / len(trainset)

# ==================================================
# Optimization
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

    score = metric(
        example,
        prediction
    )

    optimized_total += score

    details[idx]["Optimized Score"] = round(
        score,
        3
    )

optimized_average = (
    optimized_total / len(trainset)
)

# ==================================================
# Detailed Table
# ==================================================

details_df = pd.DataFrame(details)

print("\n")
print("=" * 100)
print("DETAILED RESULTS")
print("=" * 100)

print(details_df.to_string(index=False))

# ==================================================
# Summary Table
# ==================================================

summary_df = pd.DataFrame([
    {
        "Model": "llama3.1:8b",
        "Baseline Score": round(
            baseline_average,
            3
        ),
        "Optimized Score": round(
            optimized_average,
            3
        ),
        "Improvement": round(
            optimized_average -
            baseline_average,
            3
        )
    }
])

print("\n")
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(summary_df.to_string(index=False))

# ==================================================
# Long Document Test
# ==================================================

hard_text = """
Modern AI systems increasingly combine large language models,
vector databases, retrieval pipelines, workflow orchestration,
and external tools. While language models possess strong reasoning
capabilities, they can become inaccurate when relying only on
their training data.

Frameworks such as DSPy allow developers to define tasks using
declarative Signatures and automatically optimize prompts,
examples, and reasoning strategies. This reduces manual prompt
engineering effort and helps developers build more reliable,
maintainable AI applications.
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

print("\nResults saved:")
print("details_results.csv")
print("summary_results.csv")
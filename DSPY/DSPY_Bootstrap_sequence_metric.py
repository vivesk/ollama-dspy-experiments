import dspy
from dspy.teleprompt import BootstrapFewShot
from difflib import SequenceMatcher

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
# Signature
# ==================================================

class Summarize(dspy.Signature):
    """Summarize the provided text in one concise sentence."""

    text = dspy.InputField()
    summary = dspy.OutputField()

# ==================================================
# Training Data (20 Examples)
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
        text="Artificial intelligence systems can perform tasks requiring human-like reasoning, perception, and decision making.",
        summary="Artificial intelligence enables machines to perform human-like cognitive tasks."
    ).with_inputs("text"),

    dspy.Example(
        text="Cloud computing delivers computing resources over the internet, enabling scalable and cost-effective infrastructure.",
        summary="Cloud computing provides scalable computing resources over the internet."
    ).with_inputs("text"),

    dspy.Example(
        text="Cybersecurity practices help protect systems, networks, and data against unauthorized access and attacks.",
        summary="Cybersecurity protects systems and data from digital threats."
    ).with_inputs("text"),

    dspy.Example(
        text="Data science combines statistics, programming, and domain expertise to derive insights from structured and unstructured data.",
        summary="Data science extracts insights from data using statistical and computational methods."
    ).with_inputs("text"),

    dspy.Example(
        text="Neural networks are machine learning models inspired by the structure of the human brain and are widely used in deep learning.",
        summary="Neural networks are foundational models used in deep learning."
    ).with_inputs("text"),

    dspy.Example(
        text="Large language models are trained on massive text corpora to understand and generate human language.",
        summary="Large language models generate and understand language using large-scale training."
    ).with_inputs("text"),

    dspy.Example(
        text="Prompt engineering focuses on designing effective instructions that improve the quality of AI-generated responses.",
        summary="Prompt engineering improves AI outputs through better instructions."
    ).with_inputs("text"),

    dspy.Example(
        text="Open-source software allows users to inspect, modify, and distribute source code freely.",
        summary="Open-source software provides access to freely modifiable source code."
    ).with_inputs("text"),

    dspy.Example(
        text="Continuous integration automates testing and validation whenever code changes are submitted.",
        summary="Continuous integration automates code testing and validation."
    ).with_inputs("text"),

    dspy.Example(
        text="Continuous deployment automates software releases, reducing manual effort and accelerating delivery.",
        summary="Continuous deployment automates software delivery and releases."
    ).with_inputs("text"),

    dspy.Example(
        text="APIs enable systems and applications to communicate and exchange data through defined interfaces.",
        summary="APIs facilitate communication between software systems."
    ).with_inputs("text"),
]

# ==================================================
# Better Metric
# ==================================================

def metric(example, prediction, trace=None):
    """
    Returns similarity score between
    expected summary and predicted summary.

    1.0 = perfect match
    0.0 = completely different
    """

    return SequenceMatcher(
        None,
        example.summary.lower(),
        prediction.summary.lower()
    ).ratio()

# ==================================================
# Program
# ==================================================

program = dspy.Predict(Summarize)

# ==================================================
# Evaluate Before Optimization
# ==================================================

print("\nEVALUATING BASE MODEL")
print("=" * 70)

baseline_total = 0

for i, example in enumerate(trainset, start=1):

    prediction = program(
        text=example.text
    )

    score = metric(
        example,
        prediction
    )

    baseline_total += score

    print(f"\nExample {i}")
    print("-" * 40)
    print("Expected:")
    print(example.summary)

    print("\nPredicted:")
    print(prediction.summary)

    print(f"\nScore: {score:.3f}")

baseline_average = baseline_total / len(trainset)

print("\n")
print("=" * 70)
print(f"BASELINE AVERAGE SCORE: {baseline_average:.3f}")
print("=" * 70)

# ==================================================
# Bootstrap Optimization
# ==================================================

optimizer = BootstrapFewShot(
    metric=metric
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)

# ==================================================
# Evaluate After Optimization
# ==================================================

optimized_total = 0

print("\nOPTIMIZED MODEL")
print("=" * 70)

for example in trainset:

    prediction = optimized_program(
        text=example.text
    )

    optimized_total += metric(
        example,
        prediction
    )

optimized_average = optimized_total / len(trainset)

print(
    f"\nOPTIMIZED AVERAGE SCORE: "
    f"{optimized_average:.3f}"
)

print(
    f"IMPROVEMENT: "
    f"{optimized_average - baseline_average:.3f}"
)

# ==================================================
# Hard Test Paragraph
# ==================================================

hard_text = """
Modern AI systems increasingly combine large language models,
vector databases, retrieval pipelines, workflow orchestration,
and external tools. While language models possess strong reasoning
capabilities, their responses can become inaccurate when relying
solely on internal training data. Frameworks such as DSPy provide
a declarative way to describe tasks while automatically optimizing
prompts, demonstrations, and reasoning strategies. This allows
developers to focus on system design instead of repeatedly tuning
prompts by hand, leading to more reliable and maintainable AI
applications.
"""

print("\n")
print("=" * 70)
print("LONG DOCUMENT TEST")
print("=" * 70)

before = program(text=hard_text)
after = optimized_program(text=hard_text)

print("\nBEFORE:")
print(before.summary)

print("\nAFTER:")
print(after.summary)
import dspy

dspy_lm = dspy.LM(
    model="openai/llama3.1:8b",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=dspy_lm)

response = dspy_lm("Say hello in one sentence")

print(response)
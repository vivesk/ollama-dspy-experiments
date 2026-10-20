#from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
import dspy

model =ChatOllama(
    model="llama3.1:8b",              # model name/tag
    temperature=0.7,               # sampling temperature
    num_predict=256,               # max tokens
    top_k=40,                      # token selection limit
    top_p=0.9,                     # nucleus sampling
    mirostat=1,                     # Mirostat sampling
    seed=42,                       # reproducibility
    base_url="http://localhost:11434",  # Ollama server URL
    keep_alive="5m",               # how long model stays loaded
)

# --------------------------
# DSPy Model
# --------------------------

dspy_lm = dspy.LM(
    model="openai/llama3.1:8b",
    api_base="http://localhost:11434/v1",
    api_key="ollama"
)

dspy.configure(lm=dspy_lm)

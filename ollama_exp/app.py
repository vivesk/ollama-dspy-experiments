from config import model
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="you as AI expert, predict future in one line  ?")
]
full_response = ""

for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
    full_response += chunk.content


print("\n\nFinal Response:")
print(full_response)
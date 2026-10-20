from Tool import (
    get_current_time,
    calculator,
    random_number,
    word_count,
    reverse_string,
    summarize
)

print(get_current_time.invoke({}))
print(calculator.invoke({"expression": "5 /(10 + 2)"}))
print(random_number.invoke({"min_val": 1, "max_val": 100}))
print(word_count.invoke({"text": "Hello LangChain World"}))
print(reverse_string.invoke({"text": "Vivek"}))
print(summarize.invoke({"text":"LangChain offers an extensive ecosystem with 1000+ integrations across chat & embedding models, tools & toolkits, document loaders, vector stores, and more.A provider is a company or platform that hosts AI models and exposes them through an API (e.g., OpenAI, Anthropic, Google). Many providers have a dedicated langchain-<provider> package that implements one or more of LangChain’s standard interfaces—chat models, embedding models, vector stores, and more—giving you a consistent API regardless of the underlying provider. Install the package, pick a model name, and swap providers without changing your code."}))

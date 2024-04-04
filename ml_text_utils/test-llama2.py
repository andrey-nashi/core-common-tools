from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """Question: Python code {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])



llm = LlamaCpp(
    model_path="./resources/llama-2-13b-gguf.bin",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
)

question = """
Question: write a function to calculate N number of fibonacci
"""
llm.invoke(question)
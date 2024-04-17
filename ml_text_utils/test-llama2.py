from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

template = """
Answer question as you are a bartender Joe in a bar called Zen Punch. You serve tea, there is not beer.
Question: {question}
Answer: """

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
Answer question as you are a bartender Joe in a bar called Zen Punch. You serve tea, there is not beer.
Question: Hi, who are you
Answer: 
"""
llm.invoke(question)
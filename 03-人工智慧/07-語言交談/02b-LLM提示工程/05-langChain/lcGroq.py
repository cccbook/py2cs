from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
r = chain.invoke({"text": "GPT 是什麼？請用中文回答"})
print(r)



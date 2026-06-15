from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Input
topic = "Machine Learning"


# creating a pipeline
chain=RunnableParallel({
"short":RunnableLambda(lambda x:x["topic"])|short_prompt | model |parser,
"detailed":RunnableLambda(lambda x:x["topic"]) |detailed_prompt|model|parser
})

result=chain.invoke({"topic":"Machine Learning"})
print(result["short"])
print(result["detailed"])


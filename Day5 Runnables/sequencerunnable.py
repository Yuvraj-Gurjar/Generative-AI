# In this w/o using chains/runnables everything is running manually

# from dotenv import load_dotenv
# load_dotenv()

# from langchain_mistralai import ChatMistralAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # prompt template
# prompt=ChatPromptTemplate.from_template(
#        "Explain {topic} in simple words"
# )

# # AI model 
# model=ChatMistralAI(model="mistral-medium-3-5")

# # output parser
# parser=StrOutputParser()

# # step by step manual flow


# # format the prompt
# formatted_prompt=prompt.format_messages(topic="Machine learning")

# # calling the model manually
# response=model.invoke(formatted_prompt)

# # parse the output manually
# final_output=parser.parse(response.content)

# print(final_output)

# ===================================================================================================================

# After using runnables/chains

from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt template
prompt=ChatPromptTemplate.from_template(
       "Explain {topic} in simple words"
)

# AI model 
model=ChatMistralAI(model="mistral-medium-3-5")

# output parser
parser=StrOutputParser()

# creating chains/runnable
chain=prompt|model|parser
result=chain.invoke("Machine Learning")
print(result)


# from dotenv import load_dotenv
# load_dotenv()
# from langchain_mistralai import ChatMistralAI
# from langchain.tools import tool

# from rich import print

# #Creating a tool

# @tool
# def get_text_length(text:str)->int:
#     """Return the number of character in the text"""
#     return len(text)

# #tool binding
# llm=ChatMistralAI(model="mistral-medium-3-5")
# llm_with_tool=llm.bind_tools([get_text_length])

# # llm decides the tool
# result=llm_with_tool.invoke("Return the number of character in the text:'hello how are u'")
# print(result)

# # execution of the tool
# if result.tool_calls:
#     tool_call=result.tool_calls[0]

# tool_name=tool_call["name"]    
# tool_args=tool_call["args"]    

# tool_result=get_text_length.invoke(tool_args)

# # send back to llm
# final_response=llm_with_tool.invoke(f"The length of the text is {tool_result}")
# print(final_response)

#================================================================================================================
#another way of doing this

from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from rich import print

#Creating a tool

@tool
def get_text_length(text:str)->int:
    """Return the number of character in the text"""
    return len(text)

tools={
    "get_text_length":get_text_length
}

llm=ChatMistralAI(model="mistral-medium-3-5")

#tool binding
llm_with_tool=llm.bind_tools([get_text_length])

# for human message
message=[]
prompt=input("You:")
query=HumanMessage(prompt)
message.append(query)

# for AI message
result=llm_with_tool.invoke(message)
message.append(result)

if result.tool_calls:
    tool_name=result.tool_calls[0]["name"]
    tool_message=tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)

result=llm_with_tool.invoke(message)
print(result.content)

#creation of tool

from langchain.tools import tool
@tool #decorator for creating a tool

def get_greeting(name:str)->str: #type hints
    """Generate a greeting message for a user"""  #docstring
    return f"Hello {name},Welcome to AI world"

result=get_greeting.invoke({"name":"Yuvraj"})
print(result)

print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)
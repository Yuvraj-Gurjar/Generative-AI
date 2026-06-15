## this code is different from core.py bcz this is machine friendly gives the result in json format 


from dotenv import load_dotenv
# using fixed template we are using prompt templates
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List,Optional  # if u want to input i.e optional
# now to check our output scehma is correct or not u need a teacher for that we use parser
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()
# now create a schema
class Movie(BaseModel):
    # write everything which u want from that paragragh
    title:str
    release_year:Optional[int]
    genre:List[str]
    director:Optional[str]
    cast:List[str]
    rating:Optional[float]
    summary:str

model = ChatGroq(model="llama-3.3-70b-versatile")
parser=PydanticOutputParser(pydantic_object=Movie) #this will check everything related to the schema is correct or not

prompt = ChatPromptTemplate.from_messages([
    ('system',"""
Extract movie information from the paragraph
     {format_instructions}
"""),
('human',"{paragraph}")
])

para = input("Give me your paragraph: ")

final_prompt = prompt.invoke(
    {"paragraph": para,
     "format_instructions":parser.get_format_instructions()
     }
)

response = model.invoke(final_prompt)

print(response.content)
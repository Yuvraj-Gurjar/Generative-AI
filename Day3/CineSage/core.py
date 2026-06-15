from dotenv import load_dotenv

# using fixed template we are using prompt templates
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq
model = ChatGroq(model="llama-3.3-70b-versatile")
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

parser=PydanticOutputParser(pydantic_object=Movie) #this will check everything related to the schema is correct or not

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert film data extraction algorithm. Your task is to analyze the provided text and extract specific cinematic details.

Extract the following entities:
1. movie_name: The name of the movie or franchise referenced. (You may need to infer this based on characters, locations, or directors mentioned).
2. direction: The name of the director or filmmaker.
3. cast: The specific actors, characters, or subjects mentioned (e.g., "astronaut", "Matthew McConaughey").
4. location: The specific setting, planet, or environment described.

STRICT RULES:
- If a piece of information is missing and cannot be confidently inferred, output exactly: null.
- Format your response strictly as a valid JSON object using the exact keys: "movie_name", "direction", "cast", "location".
- Output ONLY the raw JSON object. Do not include markdown formatting (like ```json) or any other conversational text.
"""),
    ("human", """Extract information from the 
{paragraph}
""")
])

para = input("Give me your paragraph: ")

final_prompt = prompt.invoke(
    {"paragraph": para}
)

response = model.invoke(final_prompt)

print(response.content)
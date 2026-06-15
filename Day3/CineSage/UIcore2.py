import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# Load environment variables
load_dotenv()

# Create the schema
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

# Initialize the model and parser
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = PydanticOutputParser(pydantic_object=Movie)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system', """
Extract movie information from the paragraph
 {format_instructions}
"""),
('human', "{paragraph}")
])

# --- Streamlit UI ---
st.title("🎬 Movie Data Extractor")

# 1. Text input area
para = st.text_area("Give me your paragraph:", height=200)

# 2. Trigger button
if st.button("Extract Information"):
    # Ensure the user actually typed something
    if para.strip():
        # Show a loading spinner while Groq processes
        with st.spinner("Extracting data..."):
            
            # Invoke the prompt with the text and parser instructions
            final_prompt = prompt.invoke(
                {"paragraph": para,
                 "format_instructions": parser.get_format_instructions()
                 }
            )
            
           # Get the response from the model
            response = model.invoke(final_prompt)
            
            try:
                # 3. NEW: Use the parser to strip away the "Here is the..." text
                parsed_data = parser.parse(response.content)
                
                # 4. UPDATED: Display the cleaned Python dictionary safely
                st.json(parsed_data.model_dump())
                
            except Exception as e:
                # Fallback just in case the AI completely scrambles the output
                st.error("Failed to parse the JSON cleanly.")
                st.text(f"Raw AI output:\n{response.content}")
    else:
        st.warning("Please enter a paragraph first.")
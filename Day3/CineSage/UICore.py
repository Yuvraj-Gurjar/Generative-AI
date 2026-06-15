import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate   
from langchain_groq import ChatGroq

# Load environment variables (.env file)
load_dotenv()

# Set up the basic page title
st.title("🎬 Cinematic Data Extractor")

# Define the Prompt Template
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

# Initialize the model
# We use @st.cache_resource so Streamlit doesn't reconnect to Groq on every single click
@st.cache_resource
def get_model():
    return ChatGroq(model="llama-3.3-70b-versatile")

# 1. User Input Area
para = st.text_area("Paste your cinematic paragraph here:", height=200)

# 2. Extract Button
if st.button("Extract Details"):
    if para.strip() == "":
        st.warning("Please enter a paragraph first.")
    else:
        # Show a loading spinner while Groq is thinking
        with st.spinner("Extracting data..."):
            model = get_model()
            
            # Format the prompt with the user's text
            final_prompt = prompt.invoke({"paragraph": para})
            
            # Send to the AI
            response = model.invoke(final_prompt)
            
            # 3. Display the Output nicely formatted as JSON
            st.subheader("Extraction Result:")
            st.code(response.content, language="json")
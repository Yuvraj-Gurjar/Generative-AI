# 🎬 Movie Data Extractor

A powerful, UI-driven web application that uses **Generative AI** to read messy, unstructured paragraphs about movies and extract the details into a clean, structured JSON format. 

Built with **Streamlit**, **LangChain**, **Pydantic**, and powered by **Groq** (Llama-3.3-70b-versatile).

---

## ✨ Features
* **Smart Extraction:** Automatically identifies the Title, Release Year, Genre, Director, Cast, Rating, and a brief Summary from any text.
* **Strict Formatting:** Uses `Pydantic` to force the AI to return strictly valid JSON data, completely eliminating "chatty" AI errors.
* **Blazing Fast:** Runs on Groq's LPU inference engine for near-instant text processing.
* **Clean UI:** A simple, intuitive web interface built with Streamlit.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following:
1. **Python 3.9+** installed on your machine.
2. A **Groq API Key**. You can get one for free at the [Groq Console](https://console.groq.com/).

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project
Create a folder for your project and save the provided Python script as `app.py`.

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to manage your dependencies. You can use standard `pip` or the blazing-fast `uv` package manager.

**Option A: Using `uv` (Recommended for Speed)**
```bash
# Create the virtual environment
uv venv

# Install the required packages
uv pip install streamlit langchain langchain-groq langchain-core pydantic python-dotenv

Option B: Using standard pip

# Create the virtual environment
python -m venv .venv

# Activate the environment
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install the required packages
pip install streamlit langchain langchain-groq langchain-core pydantic python-dotenv

3. Configure Your Environment Variables
Create a new file in the exact same folder as app.py and name it .env.
Open the .env file and add your Groq API key exactly like this:

Code snippet
GROQ_API_KEY=your_api_key_here
(Note: Never share your API key or upload your .env file to public repositories like GitHub!)

💻 How to Run the App
Once your packages are installed and your API key is set, run the following command in your terminal:

If using uv:

Bash
uv run streamlit run app.py
If using standard pip:

Bash
streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

🧪 Try It Out!
Copy and paste the following paragraph into the Streamlit app to test the extraction:

Set in a world where corporate espionage is executed through dream-sharing technology, the 2010 sci-fi action thriller Inception follows a skilled thief who steals valuable secrets from deep within the subconscious mind during the dream state. Boasting a stellar critical rating of 8.8 out of 10, this absolute masterpiece was written and directed by Christopher Nolan. The high-stakes heist narrative features an exceptional ensemble cast including Leonardo DiCaprio, Joseph Gordon-Levitt, and Elliot Page, culminating in a visually stunning sequence of shifting gravity and collapsing dreamscapes.

Click Extract Information and watch it instantly convert that text into a beautiful, structured JSON object!

🧠 How the Code Works Under the Hood
The Schema (Pydantic): We define a Python class called Movie that acts as a blueprint. We tell it exactly what data types to expect (e.g., title must be a string, release_year must be a number).

The Prompt (LangChain): We pass this blueprint to the AI inside the hidden system prompt.

The Try/Except Block: AI models naturally try to talk like humans (e.g., "Here is the data you requested..."). The parser.parse() function acts as a filter, hunting through the AI's response, stripping away the conversational filler, and safely extracting only the pure data
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize modern Tavily tool (note the plural 'max_results')
search_tool = TavilySearch(max_results=5)

# 2. Initialize Mistral model with explicit model parameter
llm = ChatMistralAI(model="mistral-medium-3-5")

# 3. Define the prompt template (fixed the quadruple quote typo)
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Summarize the following news into clear bullet points:
    {news}
    """
)

# 4. Build the LCEL pipeline
chain = prompt | llm | StrOutputParser()

print("Fetching latest AI news...")
# 5. Execute search and invoke the chain
news_result = search_tool.run("Latest AI news of 2026")
result = chain.invoke({"news": news_result})

print("\n--- SUMMARY ---")
print(result)
import streamlit as st

# 1. PAGE CONFIGURATION (Must be the absolute first Streamlit command)
st.set_page_config(page_title="AI News Summarizer", page_icon="📰", layout="centered")

# Render UI Header instantly so the screen doesn't appear blank while importing libraries
st.title("🤖 Live AI News Summarizer")
st.write("Search the live web for real-time updates and generate instant, structured AI bullet points.")
st.write("---")

# 2. HEAVY IMPORTS (Placed after UI rendering to protect loading states)
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_tavily import TavilySearch
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 3. SIDEBAR CONFIGURATION
st.sidebar.header("App Settings")
max_results = st.sidebar.slider("Number of search sources to parse", min_value=1, max_value=10, value=5)

# 4. ENVIRONMENT VARIABLE SAFETY CHECK
if not os.getenv("TAVILY_API_KEY") or not os.getenv("MISTRAL_API_KEY"):
    st.error("⚠️ **Missing API Keys!** Please check that your `.env` file contains both `TAVILY_API_KEY` (no R!) and `MISTRAL_API_KEY` without quotation marks.")
else:
    # 5. DYNAMIC USER INPUT FIELD
    search_topic = st.text_input("What topic or news event do you want to summarize?", value="Latest AI news of 2026")

    # 6. EXECUTION PIPELINE
    if st.button("Fetch & Summarize News", type="primary"):
        try:
            # Active status spinner block
            with st.spinner("🌐 Crawling the web and analyzing sources... Please wait..."):
                
                # Initialize modern Tavily search tool
                search_tool = TavilySearch(max_results=max_results)

                # Initialize Mistral Chat Model
                llm = ChatMistralAI(model="mistral-medium-3-5")

                # Define structural prompt template
                prompt = ChatPromptTemplate.from_template(
                    """
                    You are a helpful assistant.

                    Summarize the following news into clear bullet points:
                    {news}
                    """
                )

                # Assemble the LCEL Chain pipeline
                chain = prompt | llm | StrOutputParser()

                # Step A: Fetch real-time web context
                news_result = search_tool.run(search_topic)

                if not news_result.strip():
                    st.warning("The web search returned empty data. Try refining your keywords.")
                else:
                    # Step B: Pass live data into the LLM chain
                    summary_output = chain.invoke({"news": news_result})

                    # Step C: Render final styled layout results
                    st.success("✨ Summary Compiled Successfully!")
                    st.subheader(f"📊 Summary Report: {search_topic}")
                    st.markdown(summary_output)

                    # Collapsible dropdown layout showing what the AI read
                    with st.expander("🔍 View Raw Fetched Search Context"):
                        st.write(news_result)

        except Exception as e:
            st.error(f"An unexpected runtime error occurred: {e}")
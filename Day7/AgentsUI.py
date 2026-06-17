import os
import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from tavily import TavilyClient
from langchain.agents import create_agent

# ==========================================
# 🔧 TOOLS DEFINITION
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: Missing OpenWeather API Key."
        
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather')}"
        
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"Weather in {city}: {desc}, {temp}°C"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return "Error: Missing Tavily API Key."
        
    tavily_client = TavilyClient(tavily_key)
    try:
        response = tavily_client.search(
            query=f"latest news in {city}",
            search_depth="basic",
            max_results=3
        )
        results = response.get("results", [])
        
        if not results:
            return f"No news found for {city}"
        
        news_list = []
        for r in results:
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")
            news_list.append(f"- **{title}**\n  🔗 {url}\n  📝 {snippet[:100]}...")
        
        return f"Latest news in {city}:\n\n" + "\n\n".join(news_list)
    except Exception as e:
        return f"Error fetching news: {str(e)}"

# ==========================================
# 🖥️ STREAMLIT UI SETUP
# ==========================================

st.set_page_config(page_title="City Assistant AI", page_icon="🏙️", layout="centered")
st.title("🏙️ City Assistant AI")
st.caption("Ask about the weather or news. Built with Mistral AI & LangChain.")

# Initialize Session State for Chat History & Interruption Flow
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_tool_call" not in st.session_state:
    st.session_state.pending_tool_call = None

# Initialize LLM and Agent
llm = ChatMistralAI(model="mistral-small-2506")
tools = [get_weather, get_news]

# Display past chat messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage) and msg.content:
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Custom Human Approval Interceptor for Streamlit UI
def process_agent_step(user_query):
    # Construct total history context for the agent
    history = [HumanMessage(content=user_query)]
    
    # In a fully managed langgraph agent, middleware handles loops, 
    # but for standard langchain agents in Streamlit, we check the model's intent:
    with st.spinner("Thinking..."):
        response = llm.bind_tools(tools).invoke(history)
        
    if response.tool_calls:
        # Save the layout state so UI can show confirmation buttons
        st.session_state.pending_tool_call = response.tool_calls[0]
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.rerun()
    else:
        # Pure text response, no tools needed
        st.session_state.messages.append(HumanMessage(content=user_query))
        st.session_state.messages.append(AIMessage(content=response.content))
        st.rerun()

# --- HANDLE PENDING INTERRUPTIONS ---
if st.session_state.pending_tool_call:
    tool_call = st.session_state.pending_tool_call
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    
    st.warning(f"🤖 **Agent wants to use tool:** `{tool_name}` with arguments: `{tool_args}`")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve Tool Call", use_container_width=True):
            with st.spinner("Executing Tool..."):
                # Run the approved tool
                selected_tool = next(t for t in tools if t.name == tool_name)
                tool_output = selected_tool.invoke(tool_args)
                
                # Forward result back to LLM to generate final response
                history = st.session_state.messages + [
                    AIMessage(content="", tool_calls=[tool_call]),
                    ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                ]
                final_response = llm.invoke(history)
                
                # Append final sequence to history
                st.session_state.messages.append(AIMessage(content=final_response.content))
                st.session_state.pending_tool_call = None
                st.rerun()
                
    with col2:
        if st.button("❌ Deny Tool Call", use_container_width=True):
            st.session_state.messages.append(AIMessage(content="Tool call denied by user."))
            st.session_state.pending_tool_call = None
            st.rerun()

# --- INPUT CHAT BOX ---
# Disable input if we are waiting for human tool approval
if not st.session_state.pending_tool_call:
    if prompt := st.chat_input("What is the weather like in Delhi?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        process_agent_step(prompt)
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your pipeline functions
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# Set up page configuration
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
# Streamlit reruns the whole script on user interaction. We use session_state 
# to persist the pipeline results and chat history across reruns.
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------------------------------------------------------
# PIPELINE EXECUTION FUNCTION
# -----------------------------------------------------------------------------
def run_ui_pipeline(source: str, language: str):
    """Executes the pipeline and updates Streamlit's session state."""
    with st.spinner("Processing input (downloading/chunking)..."):
        chunks = process_input(source)
    
    with st.spinner("Transcribing audio..."):
        transcript = transcribe_all(chunks, language=language)
    
    with st.spinner("Analyzing content (Generating Title, Summary, Insights)..."):
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_item = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
    
    with st.spinner("Building RAG Engine for Q&A..."):
        rag_chain = build_rag_chain(transcript)
    
    # Store everything in session state
    st.session_state.pipeline_result = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }
    # Clear previous chat history if a new video is processed
    st.session_state.chat_history = []

# -----------------------------------------------------------------------------
# UI LAYOUT
# -----------------------------------------------------------------------------
st.title("🤖 AI Video & Meeting Assistant")
st.caption("Extract insights, summaries, and chat with your video or audio files.")
st.markdown("---")

# Sidebar for inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    
    source_input = st.text_input(
        "Source URL or Path", 
        placeholder="Enter YouTube URL or local file path..."
    ).strip()
    
    language_input = st.selectbox(
        "Language",
        options=["english", "hinglish"],
        index=0
    )
    
    process_btn = st.button("🚀 Process Video", use_container_width=True)
    
    if process_btn:
        if not source_input:
            st.error("Please enter a valid source URL or file path.")
        else:
            run_ui_pipeline(source_input, language_input)
            st.success("Processing complete!")

# Main Dashboard View
if st.session_state.pipeline_result:
    result = st.session_state.pipeline_result
    
    st.header(f"📌 {result['title']}")
    
    # Create Layout Tabs for Structured Insights
    tab_summary, tab_insights, tab_transcript, tab_chat = st.tabs([
        "📋 Summary", 
        "🔑 Key Insights", 
        "📝 Full Transcript", 
        "💬 Chat with Video"
    ])
    
    # 1. Summary Tab
    with tab_summary:
        st.subheader("Summary")
        st.markdown(result['summary'])
        
    # 2. Key Insights Tab
    with tab_insights:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### ✅ Action Items")
            st.markdown(result['action_items'])
            
        with col2:
            st.markdown("### 🔑 Key Decisions")
            st.markdown(result['key_decisions'])
            
        with col3:
            st.markdown("### ❓ Open Questions")
            st.markdown(result['open_questions'])
            
    # 3. Transcript Tab
    with tab_transcript:
        st.subheader("Raw Transcript")
        st.text_area(
            label="Complete transcription output", 
            value=result['transcript'], 
            height=400, 
            disabled=True
        )
        
    # 4. Interactive RAG Chat Tab
    with tab_chat:
        st.subheader("💬 Ask anything about the video")
        
        # Display chat container logs
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
        # Chat input element
        if prompt := st.chat_input("What was discussed about budget/deadlines?"):
            # Display user message immediately
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Fetch response from RAG
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(result["rag_chain"], prompt)
                except Exception as e:
                    answer = f"Error fetching answer: {str(e)}"
            
            # Display assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    st.write(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    # Landing message when no video has been processed yet
    st.info("Enter a YouTube URL or local file path in the sidebar on the left and click 'Process Video' to begin.")
"""
Streamlit UI for the Multi-Agent Research Pipeline.

This app wraps the existing pipeline.py logic (search agent -> reader agent ->
writer chain -> critic chain) in a web interface, showing live progress for
each step and rendering the final report + critique in a readable layout.

Run with:
    streamlit run app.py
"""

import os
import time
import traceback

import streamlit as st
from dotenv import load_dotenv

# Load API keys from .env (OpenAI / Anthropic / Tavily / etc.)
load_dotenv()

from agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from tools import web_search, scrape_url


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def extract_text(value) -> str:
    """
    Normalize output from chains/agents into a plain string.

    Handles:
      - plain strings
      - LangChain message objects (AIMessage, etc. with a .content attribute)
      - dicts that wrap content under a common key (e.g. {"content": "..."} or
        {"text": "..."})
      - anything else -> str() fallback
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "content"):
        content = value.content
        return content if isinstance(content, str) else str(content)
    if isinstance(value, dict):
        for key in ("content", "text", "output", "report", "feedback"):
            if key in value and isinstance(value[key], str):
                return value[key]
        return str(value)
    return str(value)


def init_session_state():
    defaults = {
        "state": None,          # final pipeline state dict
        "running": False,       # whether a run is in progress
        "topic": "",            # last topic researched
        "error": None,          # last error message, if any
        "history": [],          # list of past {topic, state} runs
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def run_pipeline_with_progress(topic: str, progress_container, status_container):
    """
    Re-implementation of run_research_pipeline's 4 steps, instrumented to
    update the Streamlit UI live at each stage instead of only printing
    to the terminal.
    """
    state = {}
    total_steps = 4
    step_labels = [
        "Searching the web",
        "Reading & scraping top result",
        "Writing the report",
        "Critiquing the report",
    ]

    def set_progress(step_idx, detail=""):
        progress_container.progress(
            step_idx / total_steps,
            text=f"Step {step_idx}/{total_steps}: {step_labels[step_idx - 1]}{detail}",
        )

    # ---------------- Step 1: Search ----------------
    with status_container.status("🔎 Search Agent is working...", expanded=True) as s:
        set_progress(1)
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "input": f"find recent, reliable and detailed information about: {topic}"
        })

        if getattr(search_result, "tool_calls", None):
            tool_call = search_result.tool_calls[0]
            s.write(f"Calling web search tool with query: `{tool_call['args'].get('query', topic)}`")
            state["search_result"] = web_search.invoke(tool_call["args"]["query"])
        else:
            state["search_result"] = extract_text(search_result.content if hasattr(search_result, "content") else search_result)

        state["search_result"] = extract_text(state["search_result"])
        s.write("Search complete.")
        s.update(label="✅ Search Agent finished", state="complete", expanded=False)

    # ---------------- Step 2: Read / Scrape ----------------
    with status_container.status("📖 Reader Agent is scraping the top resource...", expanded=True) as s:
        set_progress(2)
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "input": (
                f"Based on the following search result about the topic: {topic}\n"
                f"pick the most relevant url and scrape it for deeper results.\n\n"
                f"Search Result:\n{state['search_result']}"
            )
        })

        if getattr(reader_result, "tool_calls", None):
            tool_call = reader_result.tool_calls[0]
            s.write(f"Scraping URL: `{tool_call['args'].get('url', '')}`")
            state["scraped_content"] = scrape_url.invoke(tool_call["args"]["url"])
        else:
            state["scraped_content"] = extract_text(reader_result.content if hasattr(reader_result, "content") else reader_result)

        state["scraped_content"] = extract_text(state["scraped_content"])
        s.write("Scraping complete.")
        s.update(label="✅ Reader Agent finished", state="complete", expanded=False)

    # ---------------- Step 3: Write ----------------
    with status_container.status("✍️ Writer Chain is generating the report...", expanded=True) as s:
        set_progress(3)
        research_combined = (
            f"Search Result:\n{state['search_result']}\n\n"
            f"Scraped Content:\n{state['scraped_content']}"
        )
        report_raw = writer_chain.invoke({
            "topics": topic,
            "research": research_combined,
        })
        state["report"] = extract_text(report_raw)
        s.write("Report drafted.")
        s.update(label="✅ Writer Chain finished", state="complete", expanded=False)

    # ---------------- Step 4: Critique ----------------
    with status_container.status("🧐 Critic Chain is evaluating the report...", expanded=True) as s:
        set_progress(4)
        feedback_raw = critic_chain.invoke({"report": state["report"]})
        state["feedback"] = extract_text(feedback_raw)
        s.write("Critique complete.")
        s.update(label="✅ Critic Chain finished", state="complete", expanded=False)

    progress_container.progress(1.0, text="All steps complete!")
    return state


# --------------------------------------------------------------------------- #
# Page config & styling
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    .report-box {
        background-color: rgba(127, 127, 127, 0.07);
        border: 1px solid rgba(127, 127, 127, 0.2);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
    }
    .feedback-box {
        background-color: rgba(255, 193, 7, 0.07);
        border: 1px solid rgba(255, 193, 7, 0.25);
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
    }
    .step-pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        background: rgba(127,127,127,0.15);
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

init_session_state()

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("## 🧠 Research Pipeline")
    st.caption("Search Agent → Reader Agent → Writer Chain → Critic Chain")

    st.divider()

    env_keys_found = [
        k for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY")
        if os.getenv(k)
    ]
    if env_keys_found:
        st.success(f"Loaded from .env: {', '.join(env_keys_found)}")
    else:
        st.warning("No known API keys found in .env (OPENAI_API_KEY / ANTHROPIC_API_KEY / TAVILY_API_KEY). Make sure your .env is set up.")

    st.divider()
    st.markdown("### Run History")
    if st.session_state.history:
        for i, run in enumerate(reversed(st.session_state.history)):
            if st.button(f"📄 {run['topic'][:40]}", key=f"hist_{i}", use_container_width=True):
                st.session_state.state = run["state"]
                st.session_state.topic = run["topic"]
                st.rerun()
    else:
        st.caption("No runs yet. Past reports will appear here.")

    if st.session_state.history:
        st.divider()
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.state = None
            st.rerun()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

st.title("🧠 Multi-Agent Research System")
st.caption("Enter a topic and let the search, reader, writer, and critic agents do the work.")

# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #

with st.form("topic_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        topic_input = st.text_input(
            "Research topic",
            placeholder="e.g. Latest advances in solid-state batteries",
            label_visibility="collapsed",
            disabled=st.session_state.running,
        )
    with col2:
        submitted = st.form_submit_button(
            "🚀 Run",
            use_container_width=True,
            disabled=st.session_state.running,
        )

if submitted:
    if not topic_input or not topic_input.strip():
        st.error("Please enter a topic before running the pipeline.")
    else:
        st.session_state.running = True
        st.session_state.topic = topic_input.strip()
        st.session_state.error = None
        st.session_state.state = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Run pipeline (after rerun, so the "Run" button immediately shows disabled)
# --------------------------------------------------------------------------- #

if st.session_state.running and st.session_state.state is None and st.session_state.error is None:
    progress_container = st.empty()
    status_container = st.container()

    start_time = time.time()
    try:
        result_state = run_pipeline_with_progress(
            st.session_state.topic, progress_container, status_container
        )
        elapsed = time.time() - start_time
        result_state["_elapsed_seconds"] = round(elapsed, 1)

        st.session_state.state = result_state
        st.session_state.history.append({
            "topic": st.session_state.topic,
            "state": result_state,
        })
    except Exception as e:
        st.session_state.error = f"{type(e).__name__}: {e}"
        st.session_state.error_trace = traceback.format_exc()
    finally:
        st.session_state.running = False
        st.rerun()

# --------------------------------------------------------------------------- #
# Error display
# --------------------------------------------------------------------------- #

if st.session_state.error:
    st.error(f"Pipeline failed: {st.session_state.error}")
    with st.expander("Show full traceback"):
        st.code(st.session_state.get("error_trace", ""), language="python")
    if st.button("Dismiss"):
        st.session_state.error = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Results display
# --------------------------------------------------------------------------- #

if st.session_state.state:
    state = st.session_state.state

    st.divider()
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.subheader(f"Results for: {st.session_state.topic}")
    with header_col2:
        if "_elapsed_seconds" in state:
            st.metric("Time", f"{state['_elapsed_seconds']}s")

    tab_report, tab_feedback, tab_research = st.tabs(
        ["📄 Final Report", "🧐 Critic Feedback", "🔍 Raw Research"]
    )

    with tab_report:
        report_text = extract_text(state.get("report", ""))
        st.markdown(f'<div class="report-box">{report_text}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇️ Download report (.md)",
            data=report_text,
            file_name=f"{st.session_state.topic[:50].strip().replace(' ', '_')}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tab_feedback:
        feedback_text = extract_text(state.get("feedback", ""))
        st.markdown(f'<div class="feedback-box">{feedback_text}</div>', unsafe_allow_html=True)

    with tab_research:
        st.markdown("**Search Result**")
        st.text_area(
            "Search Result",
            value=extract_text(state.get("search_result", "")),
            height=250,
            label_visibility="collapsed",
        )
        st.markdown("**Scraped Content**")
        st.text_area(
            "Scraped Content",
            value=extract_text(state.get("scraped_content", "")),
            height=250,
            label_visibility="collapsed",
        )

elif not st.session_state.running and not st.session_state.error:
    st.info("👆 Enter a topic above and click **Run** to start the pipeline.")
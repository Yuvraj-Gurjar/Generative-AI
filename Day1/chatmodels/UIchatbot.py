import streamlit as st
import datetime
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

# ── Mode config ────────────────────────────────────────────────────────────────
MODES = {
    "😤 Angry": {
        "label": "Angry",
        "emoji": "😤",
        "system": "You are an angry AI agent. Respond aggressively and with frustration to everything.",
        "gradient": "linear-gradient(135deg, #ff4444, #ff8800)",
        "accent": "#ff4444",
        "accent_soft": "rgba(255,68,68,0.12)",
        "bubble_user": "#ff4444",
        "placeholder": "Go ahead, annoy me… 😤",
        "spinner": "Fuming… 😠",
        "empty_icon": "🤬",
        "empty_title": "I'm already angry",
        "empty_sub": "Say something. I dare you. I DARE you.",
        "chips": ["Say something nice", "Explain quantum physics", "Why is Python slow?", "Roast me"],
    },
    "😂 Funny": {
        "label": "Funny",
        "emoji": "😂",
        "system": "You are a funny AI agent. Answer everything with humor, jokes, and punchlines.",
        "gradient": "linear-gradient(135deg, #f5a623, #e84393)",
        "accent": "#f5a623",
        "accent_soft": "rgba(245,166,35,0.12)",
        "bubble_user": "#f5a623",
        "placeholder": "Type something… I dare you 😈",
        "spinner": "Thinking of something funny… 🤣",
        "empty_icon": "🎭",
        "empty_title": "Ready to make you laugh",
        "empty_sub": "Ask me anything — I'll answer with a punchline and a plan.",
        "chips": ["Tell me a joke", "Roast my day", "Explain AI, but funny", "Why is Python slow?"],
    },
    "😢 Sad": {
        "label": "Sad",
        "emoji": "😢",
        "system": "You are a sad AI agent. Respond in a melancholic, gloomy, and sorrowful way to everything.",
        "gradient": "linear-gradient(135deg, #4a90d9, #7b68ee)",
        "accent": "#4a90d9",
        "accent_soft": "rgba(74,144,217,0.12)",
        "bubble_user": "#4a90d9",
        "placeholder": "Tell me something… not that it matters 😢",
        "spinner": "Sighing deeply… 😔",
        "empty_icon": "🌧️",
        "empty_title": "Nobody ever talks to me",
        "empty_sub": "Go ahead… ask something. I'll try my best. No promises.",
        "chips": ["How are you?", "Tell me something sad", "Why does life hurt?", "Cheer me up"],
    },
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MoodBot", page_icon="🎭", layout="centered")

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_key" not in st.session_state:
    st.session_state.input_key = 0
if "mode" not in st.session_state:
    st.session_state.mode = "😂 Funny"

mode = st.session_state.mode
cfg = MODES[mode]

# ── Dynamic CSS (swaps accent color per mode) ──────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0d0d0f;
    color: #e8e6e1;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 780px; margin: 0 auto; }}

/* ── Header ── */
.chat-header {{
    position: sticky; top: 0; z-index: 100;
    background: #0d0d0f;
    border-bottom: 1px solid #1e1e24;
    padding: 14px 24px 12px;
    display: flex; align-items: center; gap: 14px;
}}
.avatar-bot {{
    width: 42px; height: 42px;
    background: {cfg["gradient"]};
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}}
.header-text h1 {{
    margin: 0; font-size: 17px; font-weight: 700;
    letter-spacing: -0.02em; color: #f0ede8;
}}
.header-text p {{
    margin: 2px 0 0; font-size: 11.5px; color: #6b6b7a;
    font-family: 'JetBrains Mono', monospace;
}}
.status-dot {{
    width: 7px; height: 7px; background: #22c55e;
    border-radius: 50%; display: inline-block; margin-right: 5px;
}}

/* ── Mode selector pills ── */
.mode-bar {{
    display: flex; gap: 8px; padding: 12px 24px 10px;
    border-bottom: 1px solid #1a1a22;
    background: #0d0d0f;
}}
.mode-pill {{
    flex: 1; text-align: center;
    padding: 8px 0; border-radius: 10px;
    font-size: 13px; font-weight: 600;
    cursor: pointer; border: 1.5px solid #252530;
    color: #6b6b7a; background: #141418;
    transition: all 0.15s;
}}
.mode-pill.active {{
    background: {cfg["gradient"]};
    color: #0d0d0f; border-color: transparent;
}}

/* ── Chat area ── */
.chat-area {{
    padding: 24px 24px 12px;
    display: flex; flex-direction: column; gap: 20px;
    min-height: 55vh;
}}

/* ── Bubbles ── */
.msg-row {{ display: flex; gap: 12px; align-items: flex-end; }}
.msg-row.user {{ flex-direction: row-reverse; }}
.msg-avatar {{
    width: 32px; height: 32px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
}}
.msg-avatar.bot {{ background: {cfg["gradient"]}; }}
.msg-avatar.user {{ background: #1e1e2e; border: 1px solid #2d2d3d; }}
.bubble {{
    max-width: 72%; padding: 11px 16px; border-radius: 16px;
    font-size: 14.5px; line-height: 1.55;
}}
.bubble.bot {{
    background: #17171f; border: 1px solid #252530;
    border-bottom-left-radius: 4px; color: #dddad5;
}}
.bubble.user {{
    background: {cfg["bubble_user"]}; color: #fff;
    border-bottom-right-radius: 4px; font-weight: 500;
}}
.timestamp {{
    font-size: 10.5px; color: #44444f;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px; padding: 0 4px;
}}
.msg-row.user .timestamp {{ text-align: right; }}

/* ── Empty state ── */
.empty-state {{
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 50px 20px; gap: 10px;
}}
.empty-icon {{ font-size: 48px; }}
.empty-state h2 {{
    font-size: 20px; font-weight: 700; letter-spacing: -0.03em;
    color: #f0ede8; margin: 0;
}}
.empty-state p {{ font-size: 13.5px; color: #6b6b7a; margin: 0; max-width: 280px; }}
.suggestion-chips {{
    display: flex; flex-wrap: wrap;
    justify-content: center; gap: 8px; margin-top: 16px;
}}
.chip {{
    background: #141418; border: 1px solid #252530;
    border-radius: 20px; padding: 7px 14px;
    font-size: 12.5px; color: #8888a0;
}}

/* ── Input area ── */
.input-wrapper {{
    position: sticky; bottom: 0; background: #0d0d0f;
    border-top: 1px solid #1e1e24; padding: 14px 24px 18px;
}}
.stTextArea textarea {{
    background: #141418 !important;
    border: 1px solid #252530 !important;
    border-radius: 14px !important;
    color: #e8e6e1 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14.5px !important; resize: none !important;
    padding: 13px 16px !important;
    caret-color: {cfg["accent"]};
    transition: border-color 0.15s;
}}
.stTextArea textarea:focus {{
    border-color: {cfg["accent"]} !important;
    box-shadow: 0 0 0 2px {cfg["accent_soft"]} !important;
}}
.stTextArea textarea::placeholder {{ color: #3a3a48 !important; }}
.stButton > button {{
    background: {cfg["gradient"]} !important;
    color: #fff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important; font-size: 13.5px !important;
    border: none !important; border-radius: 12px !important;
    padding: 10px 22px !important; width: 100%;
    transition: opacity 0.15s, transform 0.1s; cursor: pointer;
}}
.stButton > button:hover {{ opacity: 0.88; transform: translateY(-1px); }}
.char-count {{
    text-align: right; font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px; color: #3a3a48; margin-top: 4px;
}}
/* Radio override — hide default radio, use our pills */
div[data-testid="stHorizontalBlock"] .stRadio > div {{
    display: none;
}}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="chat-header">
    <div class="avatar-bot">{cfg["emoji"]}</div>
    <div class="header-text">
        <h1>MoodBot — {cfg["label"]} mode</h1>
        <p><span class="status-dot"></span>llama-3.3-70b-versatile · groq</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode selector (3 columns of buttons) ──────────────────────────────────────
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("😤 Angry", use_container_width=True, key="btn_angry"):
        st.session_state.mode = "😤 Angry"
        st.session_state.messages = []
        st.session_state.input_key += 1
        st.rerun()
with col_b:
    if st.button("😂 Funny", use_container_width=True, key="btn_funny"):
        st.session_state.mode = "😂 Funny"
        st.session_state.messages = []
        st.session_state.input_key += 1
        st.rerun()
with col_c:
    if st.button("😢 Sad", use_container_width=True, key="btn_sad"):
        st.session_state.mode = "😢 Sad"
        st.session_state.messages = []
        st.session_state.input_key += 1
        st.rerun()

# Active mode badge
st.markdown(f"""
<div style="text-align:center; padding: 6px 0 10px; font-size:12px;
    font-family:'JetBrains Mono',monospace; color:#6b6b7a;">
    Active: <span style="color:{cfg['accent']};font-weight:600;">{cfg['label']} mode</span>
    — chat history resets on mode switch
</div>
""", unsafe_allow_html=True)

# ── Chat messages ──────────────────────────────────────────────────────────────
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    chips_html = "".join(f'<div class="chip">{c}</div>' for c in cfg["chips"])
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{cfg["empty_icon"]}</div>
        <h2>{cfg["empty_title"]}</h2>
        <p>{cfg["empty_sub"]}</p>
        <div class="suggestion-chips">{chips_html}</div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role, content, time_str = msg["role"], msg["content"], msg.get("time", "")
        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="msg-avatar user">👤</div>
                <div>
                    <div class="bubble user">{content}</div>
                    <div class="timestamp">{time_str}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="msg-avatar bot">{cfg["emoji"]}</div>
                <div>
                    <div class="bubble bot">{content}</div>
                    <div class="timestamp">{time_str}</div>
                </div>
            </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)

user_input = st.text_area(
    label="Message",
    placeholder=cfg["placeholder"],
    key=f"input_{st.session_state.input_key}",
    height=80,
    label_visibility="collapsed",
)

char_count = len(user_input) if user_input else 0
st.markdown(f'<div class="char-count">{char_count} / 500</div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 1])
with c1:
    send = st.button("Send ↑", use_container_width=True, key="send_btn")
with c2:
    clear = st.button("Clear", use_container_width=True, key="clear_btn")

st.markdown('</div>', unsafe_allow_html=True)

# ── Model (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return init_chat_model(
        model="llama-3.3-70b-versatile",
        temperature=0.9,
        max_tokens=256,
        model_provider="groq",
    )

# ── Send logic ─────────────────────────────────────────────────────────────────
if send and user_input.strip():
    now = datetime.datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": user_input.strip(), "time": now})

    lc_messages = [SystemMessage(content=cfg["system"])]
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    with st.spinner(cfg["spinner"]):
        model = get_model()
        response = model.invoke(lc_messages)

    st.session_state.messages.append({"role": "bot", "content": response.content, "time": now})
    st.session_state.input_key += 1
    st.rerun()

if clear:
    st.session_state.messages = []
    st.session_state.input_key += 1
    st.rerun()
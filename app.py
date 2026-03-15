# app.py

import streamlit as st
from src.chain import load_vectorstore, build_chain, ask

st.set_page_config(
    page_title="FinSight — Canadian Market Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0A0E17;
    color: #E2E8F0;
}

.stApp {
    background-color: #0A0E17;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0D1220;
    border-right: 1px solid #1E2D40;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #64748B;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Header ── */
.fin-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1E2D40;
    margin-bottom: 1.5rem;
}

.fin-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #F0F4FF;
    letter-spacing: -0.02em;
    line-height: 1;
}

.fin-logo span {
    color: #3B82F6;
}

.fin-tagline {
    font-size: 12px;
    color: #475569;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
    margin-top: 6px;
}

/* ── Metric cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: #0D1220;
    border: 1px solid #1E2D40;
    border-radius: 8px;
    padding: 14px 16px;
}

.metric-label {
    font-size: 10px;
    color: #475569;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 20px;
    font-weight: 600;
    color: #F0F4FF;
    font-family: 'DM Mono', monospace;
}

.metric-delta {
    font-size: 11px;
    color: #22C55E;
    font-family: 'DM Mono', monospace;
    margin-top: 2px;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 !important;
}

[data-testid="stChatMessageContent"] {
    background: #0D1220 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    color: #CBD5E1 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

/* user message accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    border-color: #1E3A5F !important;
    background: #0F1E33 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0D1220 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 10px !important;
}

[data-testid="stChatInput"] textarea {
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    background: transparent !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: #0D1220 !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #1E2D40 !important;
    border-radius: 6px !important;
    color: #64748B !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    padding: 6px 12px !important;
}

.stButton > button:hover {
    border-color: #3B82F6 !important;
    color: #93C5FD !important;
    background: #0F1E33 !important;
}

/* ── Divider ── */
hr {
    border-color: #1E2D40 !important;
}

/* ── Ticker tape ── */
.ticker-tape {
    background: #0D1220;
    border: 1px solid #1E2D40;
    border-radius: 8px;
    padding: 10px 16px;
    margin-bottom: 1.5rem;
    display: flex;
    gap: 24px;
    overflow: hidden;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
}

.ticker-item {
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
}

.ticker-sym { color: #93C5FD; font-weight: 500; }
.ticker-up  { color: #22C55E; }
.ticker-dn  { color: #EF4444; }
.ticker-neu { color: #64748B; }

/* ── Source citation tag ── */
.source-tag {
    display: inline-block;
    background: #0F1E33;
    border: 1px solid #1E3A5F;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    color: #93C5FD;
    font-family: 'DM Mono', monospace;
    margin: 2px;
}

/* ── Section label ── */
.section-label {
    font-size: 10px;
    color: #334155;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    margin-bottom: 8px;
    margin-top: 16px;
}

/* ── Company badge ── */
.company-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0F1E33;
    border: 1px solid #1E3A5F;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #93C5FD;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
}

.dot { width: 6px; height: 6px; border-radius: 50%; background: #22C55E; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cache chain ────────────────────────────────────────────
@st.cache_resource
def load_chain_cached(company_filter=None):
    vectorstore = load_vectorstore()
    return build_chain(vectorstore, company_filter=company_filter)


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;
                    color:#F0F4FF; letter-spacing:-0.02em;">
            Fin<span style="color:#3B82F6;">Sight</span>
        </div>
        <div style="font-size:10px; color:#334155; font-family:'DM Mono',monospace;
                    letter-spacing:0.06em; margin-top:4px;">
            CANADIAN MARKET INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="section-label">Coverage Universe</div>',
                unsafe_allow_html=True)

    company = st.selectbox(
        label="company",
        options=[
            "All companies",
            "Shopify",
            "RBC",
            "TD Bank",
            "BMO",
            "CIBC",
            "Suncor",
            "CN Rail",
        ],
        label_visibility="collapsed",
    )

    company_filter = None if company == "All companies" else company

    st.divider()

    st.markdown('<div class="section-label">Quick Queries</div>',
                unsafe_allow_html=True)

    example_questions = [
        "What are RBC's main credit risk factors?",
        "How did Shopify's revenue grow this year?",
        "Compare TD and BMO capital ratios",
        "What is Suncor's energy transition strategy?",
        "What are CN Rail's biggest operational risks?",
        "How is CIBC managing interest rate exposure?",
        "Which bank has the highest CET1 ratio?",
        "What are the key ESG commitments across all companies?",
    ]

    for q in example_questions:
        if st.button(q, use_container_width=True, key=f"btn_{q}"):
            st.session_state["example_question"] = q

    st.divider()

    # Coverage stats
    st.markdown('<div class="section-label">Data Coverage</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:11px;
                color:#475569; line-height:2;">
        <div>◈ 7 companies indexed</div>
        <div>◈ ~12,000 chunks</div>
        <div>◈ FY2024–2025 reports</div>
        <div>◈ SEDAR+ sourced</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────

# Header
st.markdown("""
<div class="fin-header">
    <div class="fin-logo">Fin<span>Sight</span></div>
    <div class="fin-tagline">◈ CANADIAN ANNUAL REPORT INTELLIGENCE · POWERED BY RAG</div>
</div>
""", unsafe_allow_html=True)

# Ticker tape
st.markdown("""
<div class="ticker-tape">
    <div class="ticker-item">
        <span class="ticker-sym">SHOP</span>
        <span class="ticker-up">▲ +2.34%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">RY</span>
        <span class="ticker-up">▲ +0.87%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">TD</span>
        <span class="ticker-dn">▼ -0.42%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">BMO</span>
        <span class="ticker-up">▲ +1.12%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">CM</span>
        <span class="ticker-neu">── 0.00%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">SU</span>
        <span class="ticker-dn">▼ -1.05%</span>
    </div>
    <div class="ticker-item">
        <span class="ticker-sym">CNR</span>
        <span class="ticker-up">▲ +0.63%</span>
    </div>
    <div style="color:#334155; font-size:10px; margin-left:auto; align-self:center;">
        STATIC · FOR DISPLAY ONLY
    </div>
</div>
""", unsafe_allow_html=True)

# Metric cards
st.markdown("""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Documents indexed</div>
        <div class="metric-value">7</div>
        <div class="metric-delta">↑ FY2024–2025</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total chunks</div>
        <div class="metric-value">11,971</div>
        <div class="metric-delta">↑ 500 char avg</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Retrieval model</div>
        <div class="metric-value" style="font-size:13px;">MiniLM-L6</div>
        <div class="metric-delta">384-dim vectors</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">LLM</div>
        <div class="metric-value" style="font-size:13px;">GPT-4o</div>
        <div class="metric-delta">↑ temp=0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Active filter badge
if company_filter:
    st.markdown(f"""
    <div class="company-badge">
        <div class="dot"></div>
        {company} · annual report only
    </div>
    """, unsafe_allow_html=True)

# ── Chat ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome message on first load
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 2rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.5rem;
                    color:#1E2D40; margin-bottom:8px;">
            What would you like to know?
        </div>
        <div style="font-size:13px; color:#334155; font-family:'DM Mono',monospace;">
            Ask about financials, risks, strategy, or ESG across 7 Canadian companies
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example question click
default_question = st.session_state.pop("example_question", "")
user_input = st.chat_input("Ask about any Canadian company annual report...")
question = user_input or default_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(""):
            chain = load_chain_cached(company_filter)
            answer = ask(chain, question)

        st.markdown(answer)

        # Footer caption
        scope = company if company_filter else "all 7 companies"
        st.caption(f"◈ Sources retrieved from {scope} · {11971} chunks searched")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
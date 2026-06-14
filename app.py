"""
Zyro Dynamics HR Help Desk — Streamlit Chatbot
Deploy this file to https://share.streamlit.io

Required secrets (set in Streamlit Cloud → Settings → Secrets):
    GROQ_API_KEY = "your_groq_api_key"
    CORPUS_PATH  = "/path/to/hr/pdfs"   # or bundle PDFs in a /data folder
"""

import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.4rem 0 0; font-size: 0.95rem; opacity: 0.75; }

    .source-badge {
        display: inline-block;
        background: #e8f4fd;
        color: #1565c0;
        border: 1px solid #bbdefb;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 2px 3px;
    }
    .oos-badge {
        background: #ffeaea;
        color: #c62828;
        border: 1px solid #ffcdd2;
        border-radius: 20px;
        display: inline-block;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .sidebar-info {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        font-size: 0.85rem;
        line-height: 1.7;
    }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 Zyro Dynamics — HR Help Desk</h1>
    <p>Your intelligent HR assistant. Ask me about leave, WFH, compensation, POSH, travel expenses, and more.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Policy Coverage")
    st.markdown("""
    <div class="sidebar-info">
    This chatbot covers 11 Zyro Dynamics HR policy documents:<br><br>
    📄 Company Profile<br>
    📄 Employee Handbook<br>
    📄 Leave Policy<br>
    📄 Work From Home Policy<br>
    📄 Code of Conduct<br>
    📄 Performance Review Policy<br>
    📄 Compensation &amp; Benefits<br>
    📄 IT &amp; Data Security<br>
    📄 POSH Policy<br>
    📄 Onboarding &amp; Separation<br>
    📄 Travel &amp; Expense Policy
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Try these questions")
    sample_questions = [
        "How many earned leave days do I get per year?",
        "What is the WFH eligibility criteria?",
        "When is salary credited each month?",
        "How do I file a POSH complaint?",
        "What is the notice period for L5 employees?",
        "What are the travel entitlements for L7?",
        "How does the PIP process work?",
        "What benefits does Zyro Dynamics offer?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=f"sq_{q[:25]}"):
            st.session_state["pending_question"] = q

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**📧 HR Helpdesk**")
    st.markdown("hr.helpdesk@zyrodyanmics.com")
    st.markdown("**🔒 POSH / ICC**")
    st.markdown("icc@zyrodyanmics.com")


# ── Build RAG Pipeline (cached — runs once) ───────────────────────────
@st.cache_resource(show_spinner="🔧 Loading HR policy documents and building knowledge base...")
def build_pipeline():
    """Load PDFs, chunk, embed, and build FAISS + retriever + LLM."""

    # Corpus path: env var → Streamlit secrets → default local path
    corpus_path = (
        os.environ.get("CORPUS_PATH")
        or st.secrets.get("CORPUS_PATH", "./data")
    )

    # Load all PDF documents
    loader = PyPDFDirectoryLoader(corpus_path)
    docs   = loader.load()

    # Chunk with overlap for context continuity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Embed using lightweight but accurate model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # MMR retriever — diverse & relevant results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # Groq LLM — fast inference
    # Set env var directly so Groq client picks it up automatically
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=512
    )

    return retriever, llm, len(docs), len(chunks)


retriever, llm, num_docs, num_chunks = build_pipeline()

# ── Prompts ───────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are the Zyro Dynamics HR Help Desk assistant.
Answer the employee's question using ONLY the information provided in the context below.
Be clear, concise, and accurate. Cite specific policy details, numbers, or deadlines when available.
If the answer is not in the context, say: "I don't have enough information in the policy documents to answer this question. Please contact hr.helpdesk@zyrodyanmics.com for assistance."

Context:
{context}

Employee Question: {question}

Answer:
""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier for the Zyro Dynamics HR Help Desk.
Determine if the question below is related to HR or company policies.

HR-related topics include: leave (earned, sick, maternity, paternity, bereavement, marriage),
work from home, remote work, compensation, salary, CTC, bonus, payroll, benefits, insurance,
performance reviews, PIP, promotions, ratings, code of conduct, ethics, discipline, conflicts of interest,
POSH, sexual harassment, ICC, onboarding, probation, resignation, notice period, separation,
travel reimbursement, expense claims, IT security, data security, company profile, offices, leadership,
employee grades, designations, ESOP, provident fund, gratuity, wellness.

Question: {question}

Respond with ONLY one word: IN_SCOPE or OUT_OF_SCOPE
""")

REFUSAL_MESSAGE = (
    "I'm sorry, but I can only assist with **Zyro Dynamics HR and company policy questions**. "
    "Your question appears to be outside the scope of our HR policies. \n\n"
    "For HR-related queries, please ask about topics such as:\n"
    "- 🌴 Leave (earned leave, sick leave, maternity, paternity...)\n"
    "- 🏠 Work from home eligibility and rules\n"
    "- 💰 Compensation, salary, benefits, ESOP\n"
    "- 📊 Performance reviews and PIP\n"
    "- ✈️ Travel and expense reimbursement\n"
    "- 🔒 Code of conduct and POSH policy\n\n"
    "You can also reach the HR team at **hr.helpdesk@zyrodyanmics.com**."
)


def format_docs(docs: list) -> str:
    """Format retrieved chunks with source metadata."""
    parts = []
    for i, doc in enumerate(docs, 1):
        fname = doc.metadata.get("source", "").split("/")[-1]
        page  = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {fname}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask_bot(question: str) -> dict:
    """
    Guardrail-enabled HR chatbot:
    1. Classify scope
    2. Refuse if out-of-scope
    3. Run RAG if in-scope
    """
    # Step 1: Scope guard
    verdict = StrOutputParser().invoke(
        llm.invoke(OOS_PROMPT.format_messages(question=question))
    ).strip().upper()

    if "OUT_OF_SCOPE" in verdict:
        return {"answer": REFUSAL_MESSAGE, "sources": [], "in_scope": False}

    # Step 2: Retrieve relevant chunks
    docs    = retriever.invoke(question)
    context = format_docs(docs)

    # Step 3: Generate grounded answer
    answer = StrOutputParser().invoke(
        llm.invoke(RAG_PROMPT.format_messages(context=context, question=question))
    )

    sources = [
        {
            "file": d.metadata.get("source", "").split("/")[-1],
            "page": d.metadata.get("page", "?")
        }
        for d in docs
    ]
    return {"answer": answer, "sources": sources, "in_scope": True}


# ── Chat session state ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I'm the **Zyro Dynamics HR Help Desk** assistant.\n\n"
                f"I've loaded **{num_docs} policy document pages** ({num_chunks} searchable chunks) "
                "covering all 11 HR policy areas.\n\n"
                "Ask me anything about **leave, WFH, compensation, performance reviews, "
                "POSH, travel expenses, onboarding, IT security**, and more!"
            ),
            "sources": [],
            "in_scope": True,
        }
    ]

# ── Render chat history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Source citations
        if msg.get("sources"):
            src_html = " ".join(
                f'<span class="source-badge">📄 {s["file"]} p.{s["page"]}</span>'
                for s in msg["sources"]
            )
            st.markdown(
                f'<div style="margin-top:8px; color:#555; font-size:0.82rem;">'
                f'<b>Sources:</b> {src_html}</div>',
                unsafe_allow_html=True
            )

        # Out-of-scope badge
        if msg.get("in_scope") is False:
            st.markdown(
                '<span class="oos-badge">⚠️ Out of HR scope</span>',
                unsafe_allow_html=True
            )

# ── Handle sidebar quick-question clicks ─────────────────────────────
pending_question = st.session_state.pop("pending_question", None)

# ── Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask an HR policy question...")
question   = pending_question or user_input

if question:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": [], "in_scope": True}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show bot response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching policy documents..."):
            result = ask_bot(question)

        st.markdown(result["answer"])

        if result.get("sources"):
            src_html = " ".join(
                f'<span class="source-badge">📄 {s["file"]} p.{s["page"]}</span>'
                for s in result["sources"]
            )
            st.markdown(
                f'<div style="margin-top:8px; color:#555; font-size:0.82rem;">'
                f'<b>Sources:</b> {src_html}</div>',
                unsafe_allow_html=True
            )

        if result.get("in_scope") is False:
            st.markdown(
                '<span class="oos-badge">⚠️ Out of HR scope — redirected</span>',
                unsafe_allow_html=True
            )

    # Save bot response to history
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  result["answer"],
        "sources":  result.get("sources", []),
        "in_scope": result.get("in_scope", True),
    })

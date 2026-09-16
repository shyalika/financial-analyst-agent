import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Financial Analyst Agent", page_icon="📊", layout="centered")

if not os.getenv("OPENAI_API_KEY") or not os.getenv("COHERE_API_KEY"):
    st.error("Missing OPENAI_API_KEY or COHERE_API_KEY. Set them in .env before running this app.")
    st.stop()

from src.retrieval.two_stage_retriever import TwoStageRetriever
from src.generation.answer_generator import AnswerGenerator


@st.cache_resource
def load_pipeline():
    return TwoStageRetriever(), AnswerGenerator()


retriever, generator = load_pipeline()

st.title("📊 Financial Analyst Agent")
st.caption("Ask a question about CBA's 2026 Half Year Results ASX Announcement.")

with st.expander("Example questions"):
    st.markdown(
        "- What was CBA's statutory net profit after tax for the 2026 half year?\n"
        "- What interim dividend per share did CBA declare for 1H26?\n"
        "- What was CBA's Common Equity Tier 1 (CET1) capital ratio?\n"
        "- How much did CBA lend to businesses during the half?"
    )

question = st.text_input("Your question", placeholder="e.g. What was CBA's net interest margin for 1H26?")
ask = st.button("Ask", type="primary", disabled=not question)

if ask and question:
    with st.spinner("Searching, reranking, and generating an answer..."):
        try:
            result = retriever.retrieve(question, candidate_k=20, final_k=5)
            chunks = result["reranked"]
            answer = generator.generate_answer(question, chunks)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Cited source chunks")
    for rank, chunk in enumerate(chunks, start=1):
        with st.expander(f"#{rank} — {chunk['section_title']}  (relevance {chunk['relevance_score']:.3f})"):
            st.caption(f"Source: {chunk['source_file']}")
            st.text(chunk["chunk_content"])

    total_cost = retriever.finops.usage.total_cost_usd + generator.finops.usage.total_cost_usd
    st.caption(
        f"Retrieval latency: {result['stage1_latency_ms']:.0f}ms (search) + "
        f"{result['stage2_latency_ms']:.0f}ms (rerank) · "
        f"Session cost so far: ${total_cost:.6f} USD"
    )

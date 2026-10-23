"""Minimal Streamlit UI for the OPPO Enco W31 manual assistant."""

import streamlit as st

from src import rag

st.set_page_config(page_title="OPPO Enco W31 Manual Assistant")
st.title("OPPO Enco W31 Manual Assistant")

question = st.text_input("Ask a question about the OPPO Enco W31")
submitted = st.button("Ask")

if submitted and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        try:
            result = rag.query(question)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

    st.subheader("Answer")
    st.write(result.answer)

    if result.sources:
        st.subheader("Sources")
        for source in result.sources:
            st.markdown(f"- {source}")

    with st.expander("Retrieved sources (debug)"):
        for item in result.retrieved:
            st.text(f"{item['source_id']}  score={item['score']}")

import os
import sys

import streamlit as st


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from app.pipeline_langchain import run_pipeline



st.set_page_config(page_title="InsightFlow AI", layout="wide")

st.title("🧠 InsightFlow AI – Multi-Modal Assistant")

user_text = st.text_area("Enter your request / question", height=120, key="user_text")

uploaded_file = st.file_uploader(
    "Upload file (text / image / PDF / audio)",
    type=["txt", "png", "jpg", "jpeg", "pdf", "mp3", "wav", "m4a"],
)

def render_result(r):
    if not r:
        st.write("No result.")
        return

    # Plain string
    if isinstance(r, str):
        st.write(r)
        return

    # LangChain AIMessage or similar -> use .content
    if hasattr(r, "content"):
        st.write(r.content)
        return

    # Anything that is not a dict -> just show it
    if not isinstance(r, dict):
        st.write(r)
        return

    # ---- Summary ----
    if {"one_liner", "bullets", "paragraph"} <= r.keys():
        st.info("🧾 KEY ONE LINER:")
        st.markdown(f" {r['one_liner']}")
        st.info("**📌 KEY POINTS:**")
        st.markdown("\n".join(f"- {b}" for b in r["bullets"]))
        st.info("**📖 DETAILED:**")
        st.write(r["paragraph"])
        return

    # ---- Sentiment ----
    if {"label", "confidence", "justification"} <= r.keys():
        st.markdown(
            f"🎭 **Sentiment:** {r['label'].title()} "
            f"({round(float(r['confidence']) * 100, 1)}%)"
        )
        st.write(f"**Why?** {r['justification']}")
        return

    # ---- Code explain ----
    if {"high_level", "step_by_step", "issues"} <= r.keys():
        st.markdown(f"🧠 **High-level:** {r['high_level']}")
        st.markdown("**🪜 Steps:**")
        st.write(r["step_by_step"])
        st.markdown("**⚠ Issues:**")
        issues = r["issues"]
        if isinstance(issues, list):
            st.markdown("\n".join(f"- {i}" for i in issues))
        else:
            st.write(issues)
        if r.get("time_complexity"):
            st.markdown(f"**⏱ Complexity:** {r['time_complexity']}")
        return

    # Fallback: unknown dict shape
    st.json(r)



if st.button("Run"):
    if not user_text and not uploaded_file:
        st.warning("Please enter some text or upload a file.")
    else:
        file_bytes = uploaded_file.read() if uploaded_file else None
        filename = uploaded_file.name if uploaded_file else None

        with st.spinner("Processing..."):
            response = run_pipeline(
                user_text=user_text,
                file_bytes=file_bytes,
                filename=filename,
            )

        st.subheader("🔍 Extracted Content")
        st.code(response["extracted_text"][:4000])

        st.subheader("📜 Plan & Logs")
        st.write("\n".join(response["plan_log"]))

        if response["mode"] == "clarify":
            st.subheader("❓ Clarification needed")
            st.info(response["clarification_question"])
        else:
            st.subheader("✅ Result")
            render_result(response["result"])

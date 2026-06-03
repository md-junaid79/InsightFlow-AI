import os
import sys
from datetime import datetime

import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.pipeline_langchain import run_pipeline
from app.cost_tracker import get_cost_tracker


st.set_page_config(page_title="InsightFlow AI", layout="wide")

# Create tabs
tab1, tab2 = st.tabs(["🧠 Main", "💰 Cost Tracking"])

# ============================================
# TAB 1: MAIN PROCESSING
# ============================================
with tab1:
    st.title("🧠 InsightFlow AI – Multi-Modal Assistant")

    user_text = st.text_area(
        "Enter your request / question", height=120, key="user_text"
    )

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

    if st.button("Run", key="run_button"):
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

            st.subheader("🎯 Result")
            render_result(response.get("result"))

            st.subheader("📋 Metadata")
            st.json(response.get("extract_meta"))

            st.success("✅ Processing complete! Check the Cost Tracking tab for API usage.")


# ============================================
# TAB 2: COST TRACKING
# ============================================
with tab2:
    st.title("💰 Cost Tracking & Analytics")

    tracker = get_cost_tracker()
    stats = tracker.get_session_cost()
    breakdown = tracker.get_task_breakdown()
    daily_stats = tracker.get_daily_stats()

    # --- Summary Cards ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💵 Total Cost",
            f"${stats['total_cost']:.8f}",
            delta=f"{stats['total_calls']} calls",
        )

    with col2:
        st.metric(
            "🔤 Input Tokens",
            f"{stats['total_input_tokens']:,}",
        )

    with col3:
        st.metric(
            "📤 Output Tokens",
            f"{stats['total_output_tokens']:,}",
        )

    with col4:
        avg_cost_per_call = (
            stats["total_cost"] / stats["total_calls"] if stats["total_calls"] > 0 else 0
        )
        st.metric(
            "📊 Avg Cost/Call",
            f"${avg_cost_per_call:.8f}",
        )

    st.divider()

    # --- Breakdown by Task ---
    if breakdown:
        st.subheader("📈 Breakdown by Task Type")

        # Create breakdown table
        task_data = []
        for task, info in breakdown.items():
            if task == "audio_transcription":
                task_data.append({
                    "Task": task,
                    "Calls": info["count"],
                    "Cost": f"${info['total_cost']:.8f}",
                    "Duration": f"{info.get('duration_sec', 0):.1f}s",
                })
            else:
                task_data.append({
                    "Task": task,
                    "Calls": info["count"],
                    "Total Tokens": info["total_tokens"],
                    "Cost": f"${info['total_cost']:.8f}",
                })

        st.dataframe(task_data, use_container_width=True)

        # Pie chart by cost
        try:
            import plotly.express as px

            task_names = list(breakdown.keys())
            task_costs = [breakdown[t]["total_cost"] for t in task_names]

            fig = px.pie(
                values=task_costs,
                names=task_names,
                title="💰 Cost Distribution by Task",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate pie chart: {e}")

    # --- Daily Stats ---
    if daily_stats:
        st.subheader("📅 Daily Usage")

        daily_data = [
            {
                "Date": date,
                "Cost": f"${info['cost']:.8f}",
                "Calls": info["calls"],
                "Tokens": info["tokens"],
            }
            for date, info in sorted(daily_stats.items())
        ]

        st.dataframe(daily_data, use_container_width=True)

        # Line chart
        try:
            import plotly.graph_objects as go

            dates = sorted(daily_stats.keys())
            costs = [daily_stats[d]["cost"] for d in dates]

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=dates, y=costs, mode="lines+markers", name="Daily Cost"
                )
            )
            fig.update_layout(
                title="📊 Daily Cost Trend",
                xaxis_title="Date",
                yaxis_title="Cost ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate line chart: {e}")

    # --- Clear History ---
    st.divider()
    col1, col2 = st.columns([0.7, 0.3])

    with col2:
        if st.button("🗑️ Clear History"):
            tracker.clear_history()
            st.success("Cost history cleared!")
            st.rerun()

    with col1:
        st.info(
            "ℹ️ Cost data is saved locally in `app/storage/costs.json`. "
            "All API calls are logged automatically."
        )

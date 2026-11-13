"""
Evaluation & Benchmarking Dashboard.
Visualizes Precision@K, Recall@K, MRR, HitRate@K, Faithfulness, and Latency
across retrieval strategies with interactive Plotly graphs.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from frontend.components.api_client import api_client


def render_evaluation():
    st.title("📈 Evaluation & Retrieval Benchmarking")
    st.caption("Empirical performance evaluation across Semantic, BM25, Hybrid RRF, and Cross-Encoder Reranking.")

    # 1. Run Benchmark Trigger
    with st.expander("🚀 Run New Benchmark Suite", expanded=False):
        st.markdown(
            "Execute the standardized evaluation harness across all 4 retrieval configurations. "
            "Evaluates Information Retrieval metrics (Precision@K, Recall@K, MRR, HitRate@K) and "
            "Generation metrics (Faithfulness, Relevance, Latency) against verified ground truth."
        )
        col_k, col_run = st.columns([2, 2])
        with col_k:
            k_val = st.slider("Evaluation K", min_value=1, max_value=10, value=5)
        with col_run:
            st.write("")
            st.write("")
            run_btn = st.button("Start Benchmark Run", type="primary")

        if run_btn:
            with st.spinner("Executing benchmark across all strategies... This evaluates real queries against the corpus."):
                try:
                    report = api_client.run_benchmark(k=k_val)
                    st.success(f"Benchmark run `{report.get('run_id')}` completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Benchmark run failed: {str(e)}")

    st.markdown("---")

    # 2. Benchmark History & Result Selection
    previous_runs = api_client.get_benchmark_results()

    if not previous_runs:
        st.info("No saved benchmark runs found. Click 'Run New Benchmark Suite' above to generate the first empirical report!")
        # Also provide a button right here for quick execution
        if st.button("⚡ Run Initial Benchmark (1-Click)", type="primary"):
            with st.spinner("Executing initial benchmark suite..."):
                try:
                    api_client.run_benchmark(k=5)
                    st.success("Benchmark completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    # Select which run to view
    run_options = {r["run_id"]: f"{r['run_id']} ({r.get('timestamp', '')[:19]}) - K={r.get('k', 5)}" for r in previous_runs}
    selected_run_id = st.selectbox(
        "Select Benchmark Run",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x],
    )

    active_report = next((r for r in previous_runs if r["run_id"] == selected_run_id), previous_runs[0])
    strategies = active_report.get("strategies", {})

    if not strategies:
        st.warning("Selected report contains no strategy records.")
        return

    # 3. Strategy Comparison Metrics Table
    st.subheader(f"Strategy Performance Summary (K = {active_report.get('k', 5)})")

    table_data = []
    for strat, m in strategies.items():
        table_data.append(
            {
                "Strategy": strat.replace("_", " ").title(),
                "Precision@K": f"{m.get('precision_at_k', 0.0):.4f}",
                "Recall@K": f"{m.get('recall_at_k', 0.0):.4f}",
                "Hit Rate@K": f"{m.get('hit_rate_at_k', 0.0):.4f}",
                "MRR": f"{m.get('mrr', 0.0):.4f}",
                "Faithfulness": f"{m.get('faithfulness', 0.0):.4f}",
                "Relevance": f"{m.get('answer_relevance', 0.0):.4f}",
                "Mean Latency (ms)": f"{m.get('mean_latency_ms', 0.0):.1f}",
            }
        )

    df_metrics = pd.DataFrame(table_data)
    st.dataframe(df_metrics, use_container_width=True)

    # 4. Interactive Plotly Charts
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📊 IR Accuracy Comparison", "⏱️ Latency vs MRR", "🕸️ Multi-Metric Radar"])

    with chart_tab1:
        # Grouped Bar Chart of Precision, Recall, and Hit Rate
        chart_df = []
        for strat, m in strategies.items():
            name = strat.replace("_", " ").title()
            chart_df.append({"Strategy": name, "Metric": "Precision@K", "Value": m.get("precision_at_k", 0.0)})
            chart_df.append({"Strategy": name, "Metric": "Recall@K", "Value": m.get("recall_at_k", 0.0)})
            chart_df.append({"Strategy": name, "Metric": "Hit Rate@K", "Value": m.get("hit_rate_at_k", 0.0)})
            chart_df.append({"Strategy": name, "Metric": "MRR", "Value": m.get("mrr", 0.0)})

        fig_bar = px.bar(
            pd.DataFrame(chart_df),
            x="Strategy",
            y="Value",
            color="Metric",
            barmode="group",
            title=f"Retrieval Accuracy Comparison (Top-K = {active_report.get('k', 5)})",
            color_discrete_sequence=["#4f46e5", "#06b6d4", "#10b981", "#f59e0b"],
        )
        fig_bar.update_layout(yaxis_range=[0, 1.05], template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_tab2:
        # Latency vs MRR Scatter
        lat_df = [
            {
                "Strategy": s.replace("_", " ").title(),
                "Mean Latency (ms)": m.get("mean_latency_ms", 0.0),
                "MRR": m.get("mrr", 0.0),
                "Faithfulness": m.get("faithfulness", 0.0),
            }
            for s, m in strategies.items()
        ]
        fig_scatter = px.scatter(
            pd.DataFrame(lat_df),
            x="Mean Latency (ms)",
            y="MRR",
            text="Strategy",
            size="Faithfulness",
            color="Strategy",
            title="Retrieval Quality (MRR) vs Execution Speed (Latency)",
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with chart_tab3:
        # Radar Chart
        categories = ["Precision@K", "Recall@K", "Hit Rate@K", "MRR", "Faithfulness", "Relevance"]
        fig_radar = go.Figure()

        for strat, m in strategies.items():
            vals = [
                m.get("precision_at_k", 0.0),
                m.get("recall_at_k", 0.0),
                m.get("hit_rate_at_k", 0.0),
                m.get("mrr", 0.0),
                m.get("faithfulness", 0.0),
                m.get("answer_relevance", 0.0),
            ]
            vals.append(vals[0])  # Close radar loop
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=strat.replace("_", " ").title(),
                )
            )

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
            showlegend=True,
            title="Multi-Metric Retrieval Profile Radar Chart",
            template="plotly_white",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 5. Benchmark Dataset & Methodology Inspection
    st.markdown("---")
    with st.expander("📋 Inspect Ground-Truth Benchmark Dataset & Methodology"):
        dataset = api_client.get_benchmark_dataset()
        st.markdown(
            """
            ### Evaluation Methodology
            - **Ground Truth:** Standardized set of research questions paired with human-verified target documents, page numbers, and key evidence terms.
            - **Information Retrieval (IR) Formulas:**
                $$\\text{Precision@K} = \\frac{|\\text{Relevant} \\cap \\text{Retrieved}_{\\le K}|}{K}$$
                $$\\text{Recall@K} = \\frac{|\\text{Relevant} \\cap \\text{Retrieved}_{\\le K}|}{|\\text{Relevant}|}$$
                $$\\text{MRR} = \\frac{1}{|Q|} \\sum_{i=1}^{|Q|} \\frac{1}{\\text{rank}_i}$$
            - **Faithfulness:** Proportion of factual claims in generated answers supported by source passages.
            - **Zero Fake Benchmarks:** All numbers above represent actual executions against the indexed database.
            """
        )
        st.dataframe(dataset, use_container_width=True)

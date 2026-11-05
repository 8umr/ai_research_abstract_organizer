"""
AI Research Abstract Organizer
Streamlit Web Application for Batch Ingestion, Theme Clustering,
Method & Dataset Extraction, and Stated Limitations Analysis.
"""

import os
import json
import time
import pandas as pd
import streamlit as st

# Import local utilities
from utils.api_client import (
    test_api_connection,
    extract_paper_metadata_api,
    extract_paper_metadata_offline,
    DEFAULT_MODEL,
    FALLBACK_MODELS
)
from utils.analyzer import (
    compute_corpus_metrics,
    build_theme_method_matrix,
    generate_synthesis_markdown_report,
    categorize_limitation
)
from utils.visualizations import (
    create_theme_bar_chart,
    create_theme_donut_chart,
    create_frequency_bar_chart,
    create_limitation_category_chart,
    create_abstract_length_histogram
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Research Abstract Organizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished typography, badges, and cards
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Tag and Badge Styling */
    .badge-theme {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-method {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-dataset {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-limitation {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 10px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 4px;
    }

    /* Paper Card Container */
    .paper-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #3B82F6;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .paper-title {
        font-size: 18px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 8px;
    }
    .paper-contribution {
        font-size: 14px;
        font-style: italic;
        color: #334155;
        margin-bottom: 10px;
        background-color: #F1F5F9;
        padding: 8px 12px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = None
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "api_test_status" not in st.session_state:
    st.session_state["api_test_status"] = None

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔬 Abstract Organizer")
    st.caption("AI Literature Extraction & Analytics")
    
    st.divider()
    
    # Mode selection
    st.subheader("1. Execution Mode")
    analysis_mode = st.radio(
        "Select Pipeline Mode:",
        ["Live Model API", "Offline / Interactive Demo"],
        index=0,
        help="Choose 'Live Model API' to extract data with your API key, or 'Offline / Interactive Demo' to test instantaneously with rule-based heuristics without an API key."
    )
    
    api_key = ""
    selected_model = DEFAULT_MODEL
    temperature = 0.2
    request_delay = 0.3
    
    if analysis_mode == "Live Model API":
        st.subheader("2. API Credentials")
        
        # Load from env var if present
        env_key = os.environ.get("XAI_API_KEY", "")
        api_key = st.text_input(
            "API Key:",
            value=env_key,
            type="password",
            placeholder="xai-...",
            help="Enter your API key. Can also be set via the XAI_API_KEY environment variable."
        )
        
        model_choice = st.selectbox(
            "Model Engine:",
            ["grok-2-latest", "grok-beta", "Custom..."],
            index=0
        )
        if model_choice == "Custom...":
            selected_model = st.text_input("Custom Model Name:", value="grok-2-latest")
        else:
            selected_model = model_choice
            
        with st.expander("Advanced Inference Settings"):
            temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
            request_delay = st.slider("Rate Limit Delay (sec):", min_value=0.0, max_value=2.0, value=0.4, step=0.1)
            
        if st.button("🔌 Test API Connection", use_container_width=True):
            if not api_key:
                st.warning("Please enter an API key first.")
            else:
                with st.spinner("Connecting to model API..."):
                    success, msg = test_api_connection(api_key, selected_model)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
    else:
        st.info("💡 **Demo Mode Enabled**: Structured extraction will execute locally using heuristic taxonomy rules. No API key needed!")
        
    st.divider()
    
    # Data source selection
    st.subheader("3. Input Data Source")
    data_source = st.radio(
        "Choose Input:",
        ["Load Benchmark Sample (12 Papers)", "Upload Custom CSV"],
        index=0
    )
    
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state["raw_df"] = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(st.session_state['raw_df'])} rows.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        # Load sample papers
        sample_path = os.path.join(os.path.dirname(__file__), "sample_papers.csv")
        if os.path.exists(sample_path):
            if st.button("📥 Load Sample Papers", use_container_width=True) or st.session_state["raw_df"] is None:
                st.session_state["raw_df"] = pd.read_csv(sample_path)
                st.success("Loaded 12 benchmark AI papers.")
        else:
            st.warning("Sample CSV file not found at expected path.")
            
    st.divider()
    
    if st.session_state["results_df"] is not None:
        if st.button("🗑️ Reset Analysis Results", use_container_width=True):
            st.session_state["results_df"] = None
            st.rerun()

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.title("AI Research Abstract Organizer")
st.markdown("""
**Automated Research Theme Grouping, Methodology Extraction, Benchmark Mapping, and Stated Limitation Analysis.**  
Transform raw paper abstracts into structured knowledge graphs, categorized research trends, and interactive dashboards.
""")

st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="badge-theme">Theme Clustering</span>
    <span class="badge-method">Methodology Extraction</span>
    <span class="badge-dataset">Benchmark Detection</span>
    <span class="badge-limitation">Stated Limitations Analysis</span>
    <span class="badge-theme">Batch Text Pipeline</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Pre-Flight Data Validation & Column Mapping
# ---------------------------------------------------------
raw_df = st.session_state["raw_df"]

if raw_df is None:
    st.info("👈 Please select a data source in the sidebar: either load the 12-paper benchmark sample or upload your custom CSV file.")
    st.stop()

# Auto-detect title and abstract columns
columns = list(raw_df.columns)
def find_col(candidates, default_idx=0):
    for c in candidates:
        for col in columns:
            if c in col.lower():
                return col
    return columns[default_idx] if columns else None

default_title_col = find_col(["title", "name", "paper", "headline"], 0)
default_abstract_col = find_col(["abstract", "summary", "text", "description", "content"], min(1, len(columns)-1))

with st.expander("📋 Data Ingestion & Column Verification", expanded=(st.session_state["results_df"] is None)):
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        title_col = st.selectbox("Select Title Column:", columns, index=columns.index(default_title_col) if default_title_col in columns else 0)
    with col2:
        abstract_col = st.selectbox("Select Abstract Column:", columns, index=columns.index(default_abstract_col) if default_abstract_col in columns else min(1, len(columns)-1))
    with col3:
        st.metric("Total Records", len(raw_df))
        
    st.dataframe(raw_df[[title_col, abstract_col]].head(4), use_container_width=True)

# ---------------------------------------------------------
# Batch Processing Execution
# ---------------------------------------------------------
if st.session_state["results_df"] is None:
    st.subheader("🚀 Batch Extraction Pipeline")
    
    p_col1, p_col2 = st.columns([2, 1])
    with p_col1:
        st.write("Ready to process abstracts. Each paper will be analyzed to classify its primary research theme, extract methods, catalog benchmark datasets, and structure stated limitations.")
    with p_col2:
        batch_limit = st.selectbox(
            "Batch Scope:",
            ["Process All Papers", "Process First 3 (Quick Test)", "Process First 5"],
            index=0
        )
    
    start_batch = st.button("⚡ Run Batch Extraction Pipeline", type="primary", use_container_width=True)
    
    if start_batch:
        if analysis_mode == "Live Model API" and not api_key:
            st.error("Please provide an API key in the sidebar, or switch to 'Offline / Interactive Demo' mode.")
            st.stop()
            
        subset_df = raw_df.copy()
        if "First 3" in batch_limit:
            subset_df = subset_df.head(3)
        elif "First 5" in batch_limit:
            subset_df = subset_df.head(5)
            
        total_items = len(subset_df)
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        extracted_records = []
        start_time = time.time()
        
        for idx, row in subset_df.iterrows():
            current_count = len(extracted_records) + 1
            paper_title = str(row[title_col])
            paper_abstract = str(row[abstract_col])
            
            status_text.text(f"Processing ({current_count}/{total_items}): {paper_title[:65]}...")
            progress_bar.progress(current_count / total_items)
            
            try:
                if analysis_mode == "Live Model API":
                    meta = extract_paper_metadata_api(
                        title=paper_title,
                        abstract=paper_abstract,
                        api_key=api_key,
                        model=selected_model,
                        temperature=temperature
                    )
                    if request_delay > 0:
                        time.sleep(request_delay)
                else:
                    meta = extract_paper_metadata_offline(
                        title=paper_title,
                        abstract=paper_abstract
                    )
                    time.sleep(0.05)  # slight pause for smooth UI progress
                    
                record = {
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "primary_theme": meta["primary_theme"],
                    "sub_themes": meta["sub_themes"],
                    "methods": meta["methods"],
                    "datasets": meta["datasets"],
                    "stated_limitations": meta["stated_limitations"],
                    "key_contribution": meta["key_contribution"],
                    "maturity_stage": meta["maturity_stage"]
                }
                extracted_records.append(record)
            except Exception as e:
                st.warning(f"Error on '{paper_title[:40]}': {e}. Using offline fallback.")
                meta = extract_paper_metadata_offline(paper_title, paper_abstract)
                record = {
                    "title": paper_title,
                    "abstract": paper_abstract,
                    "primary_theme": meta["primary_theme"],
                    "sub_themes": meta["sub_themes"],
                    "methods": meta["methods"],
                    "datasets": meta["datasets"],
                    "stated_limitations": meta["stated_limitations"],
                    "key_contribution": meta["key_contribution"],
                    "maturity_stage": meta["maturity_stage"]
                }
                extracted_records.append(record)
                
        elapsed = round(time.time() - start_time, 2)
        progress_bar.progress(1.0)
        status_text.text(f"Batch processing completed! Analyzed {total_items} papers in {elapsed}s.")
        
        st.session_state["results_df"] = pd.DataFrame(extracted_records)
        st.success(f"Successfully processed {len(extracted_records)} papers.")
        st.rerun()

# ---------------------------------------------------------
# Interactive Dashboards (Rendered when results are ready)
# ---------------------------------------------------------
if st.session_state["results_df"] is not None:
    res_df = st.session_state["results_df"]
    metrics = compute_corpus_metrics(res_df)
    
    # Metric KPI Header Cards
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_papers']}</div>
            <div class="metric-label">Analyzed Papers</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['unique_themes']}</div>
            <div class="metric-label">Research Themes</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['unique_methods']}</div>
            <div class="metric-label">Distinct Methods</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['unique_datasets']}</div>
            <div class="metric-label">Target Benchmarks</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['total_limitations']}</div>
            <div class="metric-label">Stated Limitations</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("") # spacing
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Executive Overview",
        "🏷️ Research Themes",
        "🛠️ Methods & Benchmarks",
        "⚠️ Limitations & Bottlenecks",
        "🔍 Paper Explorer & Search",
        "📥 Export & Synthesis"
    ])
    
    # -----------------------------------------------------
    # TAB 1: Executive Overview
    # -----------------------------------------------------
    with tab1:
        st.subheader("Corpus Macro Trends & Distributions")
        
        c1, c2 = st.columns(2)
        with c1:
            try:
                fig_theme = create_theme_bar_chart(metrics["theme_counts"])
                st.plotly_chart(fig_theme, use_container_width=True)
            except Exception:
                st.bar_chart(pd.Series(metrics["theme_counts"]))
                
        with c2:
            try:
                fig_donut = create_theme_donut_chart(metrics["theme_counts"])
                st.plotly_chart(fig_donut, use_container_width=True)
            except Exception:
                st.write(metrics["theme_counts"])
                
        st.divider()
        
        c3, c4 = st.columns(2)
        with c3:
            try:
                fig_methods = create_frequency_bar_chart(
                    metrics["top_methods"],
                    title="Top Extracted Methodologies & Algorithms",
                    x_label="Mentions Across Papers",
                    color_hex="#059669"
                )
                st.plotly_chart(fig_methods, use_container_width=True)
            except Exception:
                st.write(dict(metrics["top_methods"][:8]))
        with c4:
            try:
                fig_datasets = create_frequency_bar_chart(
                    metrics["top_datasets"],
                    title="Most Frequently Evaluated Benchmarks",
                    x_label="Paper Citations",
                    color_hex="#7C3AED"
                )
                st.plotly_chart(fig_datasets, use_container_width=True)
            except Exception:
                st.write(dict(metrics["top_datasets"][:8]))
                
        with st.expander("📈 Abstract Text Statistics"):
            s1, s2, s3 = st.columns(3)
            s1.metric("Average Word Count", f"{metrics['avg_abstract_words']} words")
            s2.metric("Min Word Count", f"{metrics['min_abstract_words']} words")
            s3.metric("Max Word Count", f"{metrics['max_abstract_words']} words")
            
            lengths = [len(str(a).split()) for a in res_df["abstract"]]
            try:
                fig_hist = create_abstract_length_histogram(lengths)
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception:
                st.line_chart(lengths)

    # -----------------------------------------------------
    # TAB 2: Research Themes
    # -----------------------------------------------------
    with tab2:
        st.subheader("Thematic Clusters & Subfield Granularity")
        
        available_themes = ["All Themes"] + sorted(list(metrics["theme_counts"].keys()))
        selected_theme = st.selectbox("Filter by Primary Research Theme:", available_themes)
        
        if selected_theme == "All Themes":
            filtered_theme_df = res_df
        else:
            filtered_theme_df = res_df[res_df["primary_theme"] == selected_theme]
            
        st.info(f"Showing **{len(filtered_theme_df)}** papers assigned to theme: `{selected_theme}`")
        
        for _, row in filtered_theme_df.iterrows():
            st.markdown(f"""
            <div class="paper-card">
                <div class="paper-title">{row['title']}</div>
                <div class="paper-contribution">💡 <b>Key Contribution:</b> {row['key_contribution']}</div>
                <div>
                    <span class="badge-theme">Theme: {row['primary_theme']}</span>
                    {' '.join([f'<span class="badge-method">🛠️ {m}</span>' for m in row['methods']])}
                    {' '.join([f'<span class="badge-dataset">📊 {d}</span>' for d in row['datasets']])}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # TAB 3: Methods & Benchmarks
    # -----------------------------------------------------
    with tab3:
        st.subheader("Methodology Matrix & Benchmark Mapping")
        
        m_tab1, m_tab2 = st.tabs(["Extracted Methodologies", "Evaluation Benchmarks"])
        
        with m_tab1:
            st.write("Catalog of all specific algorithms, frameworks, and mechanisms extracted from abstracts:")
            method_rows = []
            for method, count in metrics["top_methods"]:
                # find papers using this method
                papers_using = res_df[res_df["methods"].apply(lambda ms: method in ms)]["title"].tolist()
                method_rows.append({
                    "Methodology / Algorithm": method,
                    "Total Mentions": count,
                    "Papers": "; ".join(papers_using)
                })
            st.dataframe(pd.DataFrame(method_rows), use_container_width=True)
            
        with m_tab2:
            st.write("Catalog of benchmark corpora, evaluation suites, and environments:")
            dataset_rows = []
            for ds, count in metrics["top_datasets"]:
                papers_using = res_df[res_df["datasets"].apply(lambda dlist: ds in dlist)]["title"].tolist()
                dataset_rows.append({
                    "Benchmark / Dataset": ds,
                    "Total Mentions": count,
                    "Evaluated In": "; ".join(papers_using)
                })
            st.dataframe(pd.DataFrame(dataset_rows), use_container_width=True)

    # -----------------------------------------------------
    # TAB 4: Stated Limitations & Bottlenecks
    # -----------------------------------------------------
    with tab4:
        st.subheader("Critical Limitations & Stated Caveats")
        st.write("Aggregating self-reported limitations allows researchers to identify high-value frontiers, deployment bottlenecks, and open problems.")
        
        try:
            fig_limitations = create_limitation_category_chart(metrics["limitation_categories"])
            st.plotly_chart(fig_limitations, use_container_width=True)
        except Exception:
            st.write(metrics["limitation_categories"])
            
        st.subheader("Categorized Limitations by Paper")
        
        for _, row in res_df.iterrows():
            limits = row["stated_limitations"]
            if limits and isinstance(limits, list):
                with st.expander(f"⚠️ {row['title']}"):
                    for lim in limits:
                        cat = categorize_limitation(lim)
                        st.markdown(f"- **[{cat}]**: {lim}")

    # -----------------------------------------------------
    # TAB 5: Interactive Paper Explorer & Search
    # -----------------------------------------------------
    with tab5:
        st.subheader("Interactive Paper Browser & Search")
        
        scol1, scol2 = st.columns([3, 1])
        with scol1:
            search_query = st.text_input("🔍 Search keyword across titles, abstracts, methods, datasets, or limitations:", "")
        with scol2:
            stage_filter = st.selectbox("Maturity Filter:", ["All Stages"] + sorted(list(res_df["maturity_stage"].unique())))
            
        filtered_df = res_df.copy()
        if stage_filter != "All Stages":
            filtered_df = filtered_df[filtered_df["maturity_stage"] == stage_filter]
            
        if search_query.strip():
            q = search_query.lower()
            def matches_query(row):
                text_to_search = f"{row['title']} {row['abstract']} {' '.join(row['methods'])} {' '.join(row['datasets'])} {' '.join(row['stated_limitations'])}".lower()
                return q in text_to_search
            filtered_df = filtered_df[filtered_df.apply(matches_query, axis=1)]
            
        st.caption(f"Showing {len(filtered_df)} of {len(res_df)} papers")
        
        for idx, row in filtered_df.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="paper-card">
                    <div class="paper-title">{row['title']}</div>
                    <div style="margin-bottom: 8px;">
                        <span class="badge-theme">Theme: {row['primary_theme']}</span>
                        <span class="badge-theme">Stage: {row['maturity_stage']}</span>
                    </div>
                    <div class="paper-contribution">💡 <b>Key Finding:</b> {row['key_contribution']}</div>
                    <div style="margin-bottom: 6px;">
                        <b>Methods:</b> {' '.join([f'<span class="badge-method">{m}</span>' for m in row['methods']])}
                    </div>
                    <div style="margin-bottom: 6px;">
                        <b>Datasets:</b> {' '.join([f'<span class="badge-dataset">{d}</span>' for d in row['datasets']])}
                    </div>
                    <div>
                        <b>Stated Limitations:</b> {' '.join([f'<span class="badge-limitation">{l}</span>' for l in row['stated_limitations']])}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"Read Full Abstract: {row['title'][:50]}..."):
                    st.write(row["abstract"])

    # -----------------------------------------------------
    # TAB 6: Export & Synthesis
    # -----------------------------------------------------
    with tab6:
        st.subheader("Data Export & Automated Synthesis Report")
        
        synthesis_md = generate_synthesis_markdown_report(res_df, metrics)
        
        ecol1, ecol2, ecol3 = st.columns(3)
        
        with ecol1:
            # Flatten lists for clean CSV export
            export_csv_df = res_df.copy()
            export_csv_df["sub_themes"] = export_csv_df["sub_themes"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            export_csv_df["methods"] = export_csv_df["methods"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            export_csv_df["datasets"] = export_csv_df["datasets"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
            export_csv_df["stated_limitations"] = export_csv_df["stated_limitations"].apply(lambda x: " | ".join(x) if isinstance(x, list) else str(x))
            
            csv_data = export_csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Enriched CSV",
                data=csv_data,
                file_name="ai_research_abstracts_organized.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with ecol2:
            json_data = res_df.to_json(orient="records", indent=2)
            st.download_button(
                label="📥 Download Full JSON",
                data=json_data,
                file_name="ai_research_abstracts_organized.json",
                mime="application/json",
                use_container_width=True
            )
            
        with ecol3:
            st.download_button(
                label="📥 Download Synthesis Report (.md)",
                data=synthesis_md.encode("utf-8"),
                file_name="research_synthesis_report.md",
                mime="text/markdown",
                use_container_width=True
            )
            
        st.divider()
        st.subheader("Synthesis Report Preview")
        st.markdown(synthesis_md)

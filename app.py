import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.resolver import EntityResolver
from src.sample_data import PRESET_CASES
from src.feature_extractor import FEATURE_NAMES, FEATURE_DESCRIPTIONS

st.set_page_config(
    page_title="Entity Resolution Engine | kumaraguruv05",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    .match-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 6px solid #10B981;
        background-color: #F0FDF4;
    }
    .no-match-card {
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 6px solid #EF4444;
        background-color: #FEF2F2;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resolver():
    return EntityResolver()

resolver = load_resolver()

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/Amazon%20ML%20Challenge-2026-FF9900?style=for-the-badge&logo=amazon&logoColor=white", use_container_width=True)
    st.title("⚙️ Engine Settings")
    
    threshold = st.slider(
        "Match Decision Threshold",
        min_value=0.20,
        max_value=0.90,
        value=0.55,
        step=0.05,
        help="Probability cutoff for declaring an entity match. V4 was calibrated at 0.55 to optimize Macro F0.5 (2x precision weighting)."
    )
    
    st.markdown("---")
    st.markdown("### 👤 Author")
    st.markdown("**Kumaraguru V**")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-kumaraguruv05-181717?style=flat&logo=github)](https://github.com/kumaraguruv05)")
    
    st.markdown("---")
    st.markdown("### 📦 System Tech Stack")
    st.markdown("""
    * **Algorithm:** LightGBM 610-Tree GBDT
    * **Feature Engine:** 34 Custom Dimensions
    * **Phonetics:** `anyascii` Indic Romanization
    * **String Distance:** `RapidFuzz` (C++ Levenshtein)
    * **Graph Resolution:** Global 1-to-1 Bipartite Argmax
    * **Memory Footprint:** Streaming $\\le 7$ GB RAM
    """)

# Header
st.markdown('<div class="main-title">🔗 Enterprise Entity Resolution Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High-Precision Multi-Source Record Linkage & Deduplication at Scale (1.73M Entities)</div>', unsafe_allow_html=True)

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-val">0.801</div><div class="metric-label">Macro F0.5 Score (Test)</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-val">34</div><div class="metric-label">Engineered Features</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-val">610</div><div class="metric-label">LightGBM Trees</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-val">0</div><div class="metric-label">Duplicate Candidates Allowed</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_live, tab_batch, tab_arch = st.tabs([
    "⚡ Interactive Resolver",
    "📁 Batch Linkage Studio",
    "📊 Architecture & Explainability"
])

# ==============================================================================
# TAB 1: INTERACTIVE RESOLVER
# ==============================================================================
with tab_live:
    st.subheader("Interactive Entity Matching & Explainability")
    st.write("Test single Source 1 query entities against candidate pools from Source 2 and Source 3. Observe how the engine resolves typos, Indic scripts, domain URLs, and suppresses false positives.")
    
    # Preset Selector
    selected_preset_key = st.selectbox(
        "🎯 Select a Benchmark Scenario or Input Custom Data:",
        options=list(PRESET_CASES.keys()) + ["✨ Custom Query & Candidates"]
    )
    
    if selected_preset_key != "✨ Custom Query & Candidates":
        case_info = PRESET_CASES[selected_preset_key]
        st.info(f"💡 **Case Focus:** {case_info['description']}")
        
        default_q_name = case_info["query"]["business_name"]
        default_q_addr = case_info["query"]["business_address"]
        default_q_country = case_info["query"]["country"]
        default_q_id = case_info["query"]["entity_id"]
        preset_candidates = case_info["candidates"]
    else:
        default_q_name = "Global Logistics Solutions Inc"
        default_q_addr = "100 Innovation Blvd, Suite 400, San Jose, CA 95134"
        default_q_country = "US"
        default_q_id = "S1-CUSTOM-1"
        preset_candidates = [
            {
                "entity_id": "S2-CUSTOM-101",
                "source": "S2",
                "business_name": "globallogistics.com",
                "business_address": "100 Innovation Boulevard San Jose California",
                "notes": "Domain name format with exact building address."
            },
            {
                "entity_id": "S3-CUSTOM-202",
                "source": "S3",
                "business_name": "Global Logistics Express LLC",
                "business_address": "500 Airport Way, San Jose, CA",
                "notes": "Distractor: Similar name, different address."
            }
        ]

    # Query Input Form
    with st.expander("📝 Query Entity (Source 1 Reference)", expanded=True):
        col_q1, col_q2, col_q3 = st.columns([2, 3, 1])
        with col_q1:
            q_name = st.text_input("Business Name", value=default_q_name)
            q_id = st.text_input("Entity ID", value=default_q_id)
        with col_q2:
            q_addr = st.text_input("Business Address", value=default_q_addr)
        with col_q3:
            q_country = st.selectbox("Country", ["US", "India", "France"], index=0 if default_q_country=="US" else (1 if default_q_country=="India" else 2))

    # Candidate Pool Form
    st.markdown("#### Candidate Pool (Source 2 & Source 3)")
    df_cands_display = pd.DataFrame(preset_candidates)
    
    edited_df = st.data_editor(
        df_cands_display[["entity_id", "source", "business_name", "business_address", "notes"]],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "entity_id": st.column_config.TextColumn("Candidate ID", required=True),
            "source": st.column_config.SelectboxColumn("Source", options=["S2", "S3"], required=True),
            "business_name": st.column_config.TextColumn("Candidate Name", required=True),
            "business_address": st.column_config.TextColumn("Candidate Address"),
            "notes": st.column_config.TextColumn("Scenario Notes")
        }
    )
    
    if st.button("🚀 Resolve Entity & Score Candidates", type="primary", use_container_width=True):
        q_record = {
            "entity_id": q_id,
            "business_name": q_name,
            "business_address": q_addr,
            "country": q_country
        }
        cand_records = edited_df.to_dict("records")
        
        with st.spinner("Extracting 34 features & scoring via LightGBM..."):
            res = resolver.resolve_single_query(q_record, cand_records, threshold=threshold)
            
        # Display Resolution Status
        st.markdown("---")
        if res["status"] == "MATCH_FOUND":
            st.success(f"### 🟢 RESOLUTION: MATCH FOUND ({res['matched_count']} matched entities above {threshold:.2f} threshold)")
        else:
            st.warning(f"### 🛡️ RESOLUTION: TRUE SINGLETON (0 candidates exceeded {threshold:.2f} threshold. Full singleton precision credit awarded.)")
            
        # Interactive Probability Plot
        cand_names = [f"{c['entity_id']}: {c['business_name'][:25]}" for c in (res["matches"] + res["unmatched"])]
        cand_probs = [c["probability"] for c in (res["matches"] + res["unmatched"])]
        cand_colors = ["#10B981" if p >= threshold else "#EF4444" for p in cand_probs]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cand_names,
            x=cand_probs,
            orientation='h',
            marker=dict(color=cand_colors),
            text=[f"{p*100:.1f}%" for p in cand_probs],
            textposition='auto',
            name="Match Confidence"
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="#F59E0B", annotation_text=f"Decision Cutoff ({threshold:.2f})")
        fig.update_layout(
            title="Candidate Match Probabilities vs Decision Threshold",
            xaxis=dict(title="Probability Score", range=[0, 1.05]),
            yaxis=dict(autorange="reversed"),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Candidate Cards
        st.markdown("#### Detailed Candidate Analysis & Feature Attribution")
        for item in (res["matches"] + res["unmatched"]):
            card_class = "match-card" if item["is_match"] else "no-match-card"
            badge = "🟢 MATCH" if item["is_match"] else "🔴 NO MATCH"
            
            with st.container():
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0;">{item['entity_id']} ({item['source']}) — {item['business_name']}</h4>
                        <span style="font-size: 1.1rem; font-weight: bold;">{badge} ({item['probability']*100:.2f}%)</span>
                    </div>
                    <p style="margin: 4px 0 8px 0; color: #475569;"><strong>Address:</strong> {item['business_address'] or '<span style="color:#94A3B8;">[Address Missing / NaN in Source]</span>'}</p>
                    {f"<p style='margin: 0; font-size: 0.9rem; color: #0284C7;'><em>💡 Scenario Note: {item['notes']}</em></p>" if item.get('notes') else ''}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"🔍 Feature Breakdown & Model Drivers for {item['entity_id']}"):
                    if item["top_features"]:
                        df_top_feat = pd.DataFrame(item["top_features"])
                        st.dataframe(
                            df_top_feat.rename(columns={"feature": "Feature", "value": "Value (0-1)", "desc": "Engineering Description"}),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No strong positive feature signals detected for this pair.")

# ==============================================================================
# TAB 2: BATCH LINKAGE STUDIO
# ==============================================================================
with tab_batch:
    st.subheader("Batch Entity Resolution & Bipartite Argmax Matching")
    st.write("Upload a batch of entity candidate pairs to run end-to-end resolution. The engine automatically enforces **Global 1-to-1 Bipartite Argmax Matching** to eliminate duplicate candidate assignments.")
    
    # Download sample template
    sample_batch_data = pd.DataFrame([
        {
            "s1_id": "S1-965667",
            "s1_name": "Maure Williams Colombier Inc",
            "s1_address": "85 Wayne Avenue, Ticonderoga, NY",
            "cand_id": "S2-681193310",
            "cand_source": "S2",
            "cand_name": "Maure Wilblims Colombier Inc",
            "cand_address": ""
        },
        {
            "s1_id": "S1-965667",
            "s1_name": "Maure Williams Colombier Inc",
            "s1_address": "85 Wayne Avenue, Ticonderoga, NY",
            "cand_id": "S3-775321672",
            "cand_source": "S3",
            "cand_name": "Dréxkor",
            "cand_address": "85 Wanye Avenue, Ticonderoga Townshiip, New York"
        },
        {
            "s1_id": "S1-965667",
            "s1_name": "Maure Williams Colombier Inc",
            "s1_address": "85 Wayne Avenue, Ticonderoga, NY",
            "cand_id": "S2-999900111",
            "cand_source": "S2",
            "cand_name": "Williams Financial Group LLC",
            "cand_address": "85 Wayne Avenue, Jersey City, NJ"
        },
        {
            "s1_id": "S1-55344266",
            "s1_name": "Raj Investments LLP",
            "s1_address": "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai",
            "cand_id": "S2-249013014",
            "cand_source": "S2",
            "cand_name": "ராஜ் இன்வெஸ்ட்மென்ட்ஸ்",
            "cand_address": "6/29 CIT Colony 2nd Main Rd Mylapore Chennai"
        },
        {
            "s1_id": "S1-99882211",
            "s1_name": "Quantum Horizon Biotech Labs Inc",
            "s1_address": "402 Silicon Way, Austin, TX",
            "cand_id": "S2-441029481",
            "cand_source": "S2",
            "cand_name": "Quantum Dynamics Inc",
            "cand_address": "1200 Tech Ridge Blvd, Austin, TX"
        }
    ])
    
    csv_template = sample_batch_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Sample CSV Template",
        data=csv_template,
        file_name="entity_pairs_sample.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV / TSV with candidate pairs", type=["csv", "tsv"])
    
    if uploaded_file is not None:
        sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
        df_batch = pd.read_csv(uploaded_file, sep=sep)
    else:
        st.info("Showing resolution on preset sample batch (5 candidate pairs):")
        df_batch = sample_batch_data
        
    st.dataframe(df_batch, use_container_width=True, height=200)
    
    if st.button("⚡ Run Batch Resolution", type="primary"):
        with st.spinner("Running inference and bipartite argmax resolution..."):
            df_resolved = resolver.resolve_batch_with_argmax(df_batch, threshold=threshold)
            
        st.success(f"Batch resolution completed! Found {df_resolved['is_resolved_match'].sum()} confirmed matches.")
        
        # Display Results
        col_res1, col_res2 = st.columns([3, 2])
        with col_res1:
            st.dataframe(
                df_resolved[["s1_id", "cand_id", "cand_name", "probability", "passed_threshold", "is_resolved_match"]],
                use_container_width=True,
                column_config={
                    "probability": st.column_config.ProgressColumn("Match Prob", min_value=0, max_value=1, format="%.3f"),
                    "is_resolved_match": st.column_config.CheckboxColumn("Final Match")
                }
            )
            
        with col_res2:
            match_counts = df_resolved["is_resolved_match"].value_counts().rename({True: "Matched Pairs", False: "Unmatched / Conflicts"})
            fig_pie = px.pie(
                values=match_counts.values,
                names=match_counts.index,
                color=match_counts.index,
                color_discrete_map={"Matched Pairs": "#10B981", "Unmatched / Conflicts": "#EF4444"},
                title="Batch Match Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Download resolved pairs
        res_csv = df_resolved.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Resolved Matches (CSV)",
            data=res_csv,
            file_name="resolved_entity_matches.csv",
            mime="text/csv"
        )

# ==============================================================================
# TAB 3: ARCHITECTURE & EXPLAINABILITY
# ==============================================================================
with tab_arch:
    st.subheader("System Architecture & LightGBM Model Explainability")
    st.write("Deep dive into the 4-stage entity resolution pipeline, feature importances, and engineering breakthroughs.")
    
    # Feature Importances Plot
    df_imp = resolver.get_feature_importances(importance_type="gain")
    top_15_imp = df_imp.head(15)
    
    fig_imp = px.bar(
        top_15_imp,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features by LightGBM Information Gain",
        labels={"importance": "Total Information Gain", "feature": "Feature"},
        color="importance",
        color_continuous_scale="Viridis"
    )
    fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=480)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Architectural Pillars
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("### 🏛️ The 4-Stage Production Pipeline")
        st.markdown("""
        1. **Stage 1: Multi-Key Inverted Index Blocking**
           * Country-partitioned inverted index across token n-grams ($DF \\le 400$).
           * Generates candidate pairs reducing $1.73M \\times 9.97M$ Cartesian product to sparse candidate pool.
        2. **Stage 2: 34-Dimensional Feature Extraction**
           * Multi-script phonetics via `anyascii` for Indic Unicode languages (Tamil, Hindi, Telugu).
           * RapidFuzz C++ Levenshtein, token-sort, token-set, character 3-grams.
           * Domain TLD stripping (.com, .org, .fr, .in).
           * Address numeric key extraction with conflict suppression penalties.
        3. **Stage 3: LightGBM Gradient Boosted Decision Forest**
           * 610 trees with learning rate 0.05, max depth 8, num leaves 63.
           * Calibrated with Fold-4 CV yielding **0.9556 Macro F0.5**.
        4. **Stage 4: Global Bipartite Argmax Matching**
           * Enforces the mathematical truth that each candidate entity belongs to **at most ONE** reference query.
           * Eliminated over **721,000 false duplicate matches**.
        """)
        
    with col_p2:
        st.markdown("### ⚡ Engineering Breakthroughs Under Constraints")
        st.markdown("""
        * **Streaming $\\le 7$ GB RAM Footprint:**
          * Processed 1.73 million test entities on a 16 GB laptop by streaming candidate pairs in 60,000-pair chunks with PyArrow Parquet.
        * **28-Thread Parallelization:**
          * Utilized all 28 logical cores of an Intel Core i7-14700HX with `ProcessPoolExecutor` for feature extraction.
        * **The Address-Anchor Revelation:**
          * Diagnosed that synthetic corruptions often replace company names with random aliases (`Dréxkor`) while preserving the physical street address.
          * Engineered `high_addr_anchor` and `name_addr_harmonic_mean` to rescue otherwise unmatchable entities.
        * **Singleton Precision Optimization:**
          * Macro $F_{0.5}$ weights Precision $2\\times$ over Recall. Singletons score 1.0 if empty, 0.0 if any false match is predicted. Hard thresholding preserved hundreds of thousands of singleton credits.
        """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.9rem;'>Entity Resolution Engine • Developed by <a href='https://github.com/kumaraguruv05' target='_blank'>Kumaraguru V (kumaraguruv05)</a> • Amazon ML Challenge 2026</p>", unsafe_allow_html=True)

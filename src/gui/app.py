"""Streamlit GUI for TennisPreview multi-match analysis."""
import asyncio
import sys
from pathlib import Path
from datetime import date
from typing import Optional, Dict, List

# Add project root to path BEFORE any imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.utils.logging import setup_logging, get_logger
from src.utils.config import config
from src.pipeline.daily import TennisDailyPipeline, PipelineResult, TennisMatch, Player
from src.data.merge.odds_merger import OddsMerger
from src.betting.value_detector import explain_value_bet

logger = get_logger("gui")


def init_pipeline():
    """Initialize and return the pipeline."""
    setup_logging(log_level="WARNING")
    config.load()
    pipeline = TennisDailyPipeline(demo_mode=False)
    return pipeline


async def analyze_single_match(pipeline: TennisDailyPipeline, p1: str, p2: str) -> Dict:
    """Analyze a single match between two players."""
    result = PipelineResult()
    result.demo_mode = False
    result.demo_odds = False

    # Load historical data if not already loaded
    if not pipeline.historical_matches:
        await pipeline._load_historical_data()

    # Create synthetic match
    match_key = f"{p1} vs {p2}"
    match = TennisMatch(
        id=match_key,
        match_key=match_key,
        player1=Player(name=p1),
        player2=Player(name=p2),
        tournament="Analisi Singola",
        level="UNKNOWN",
        surface="Hard",
        round="",
        date=date.today(),
    )
    result.matches = [match]

    # Get Sisal coverage
    result.sisal_flag = {match.id: pipeline._sisal_presence(match)}

    # Get odds
    odds_dict = await pipeline._get_odds_for_matches([match], demo=False)
    result.odds = odds_dict

    # Get Polymarket sentiment
    poly_dict = pipeline._get_poly_sentiment([match], demo=False)
    result.poly = poly_dict

    # Get news impact
    news_impact = pipeline._get_news_impact([match])
    result.news = news_impact

    # Build features
    features_dict = pipeline._build_features([match], odds_dict, news_impact, poly_dict)
    result.features = features_dict

    # Ensure models are fitted
    if not pipeline._models_fitted:
        pipeline._fit_models()

    # Generate predictions
    predictions = pipeline._generate_predictions(features_dict)
    result.predictions = predictions

    # Find value bets
    value_bets = pipeline._find_value_bets(predictions, odds_dict, [match])
    result.value_bets = value_bets

    # Decisions
    decisions, excluded = pipeline._select_decisions(value_bets, features_dict, odds_dict)
    result.decisions = decisions
    result.excluded = excluded

    # Decision summary
    result.decision_summary = {
        'decisions': len(decisions),
        'value_found': len(value_bets),
        'excluded': len(excluded),
        'sisal_yes': sum(1 for d in decisions if d.sisal == "si"),
    }

    return result


def main():
    """Streamlit app main function."""
    st.set_page_config(
        page_title="TennisPreview",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for dark theme
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #a8a8a8;
        font-size: 1.1rem;
        margin-top: 5px;
    }
    .match-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
    }
    .stat-box {
        background: #2a2a3e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        background-color: #e94560;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 25px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎾 TennisPreview</h1>
        <p>Analisi Multi-Match con Probabilità, Sentiment e Value Betting</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'matches' not in st.session_state:
        st.session_state.matches = [{"p1": "", "p2": ""}]
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'loading' not in st.session_state:
        st.session_state.loading = False

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configurazione")
        
        if st.session_state.pipeline is None:
            if st.button("🚀 Inizializza Pipeline", use_container_width=True):
                with st.spinner("Caricamento dati storici..."):
                    st.session_state.pipeline = init_pipeline()
                    st.success("Pipeline pronta!")
        
        st.divider()
        
        st.subheader("💡 Suggerimenti")
        st.info("""
        - Aggiungi più match per analisi simultanea
        - Inserisci nomi completi (es. "Alcaraz", "Sinner")
        - Premi "Analizza Tutti" per avviare
        """)
        
        st.divider()
        
        if st.session_state.results:
            st.subheader("📊 Riepilogo")
            total_decisions = sum(len(r.get('decisions', [])) for r in st.session_state.results)
            total_value = sum(r.get('value_found', 0) for r in st.session_state.results)
            st.metric("Decisioni totali", total_decisions)
            st.metric("Value trovate", total_value)

    # Main content area
    st.markdown("### ⚡ Inserisci i Match")
    
    match_count = len(st.session_state.matches)
    cols = st.columns([1, 1, 0.5, 0.5])
    
    with cols[0]:
        st.write("Giocatore 1")
    with cols[1]:
        st.write("Giocatore 2")
    with cols[2]:
        st.write("")
    with cols[3]:
        st.write("")

    new_matches = []
    for i, m in enumerate(st.session_state.matches):
        cols = st.columns([3, 3, 1, 1])
        with cols[0]:
            p1 = st.text_input(f"P1_{i}", value=m.get("p1", ""), placeholder="Es: Alcaraz", key=f"p1_{i}")
        with cols[1]:
            p2 = st.text_input(f"P2_{i}", value=m.get("p2", ""), placeholder="Es. Medvedev", key=f"p2_{i}")
        with cols[2]:
            if i > 0:
                if st.button("❌", key=f"del_{i}"):
                    del st.session_state.matches[i]
                    st.rerun()
        with cols[3]:
            if i == len(st.session_state.matches) - 1:
                if st.button("➕", key=f"add_{i}"):
                    st.session_state.matches.append({"p1": "", "p2": ""})
                    st.rerun()

    # Analyze button
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔍 ANALIZZA TUTTI", use_container_width=True, type="primary"):
            valid_matches = [m for m in st.session_state.matches if m.get("p1") and m.get("p2")]
            if not valid_matches:
                st.error("Inserisci almeno un match con entrambi i giocatori.")
            elif st.session_state.pipeline is None:
                st.error("Inizializza la pipeline prima dalla sidebar.")
            else:
                st.session_state.loading = True
                st.session_state.results = []
                st.rerun()

    with col2:
        if st.button("🗑️ Pulisci Tutto", use_container_width=True):
            st.session_state.matches = [{"p1": "", "p2": ""}]
            st.session_state.results = []
            st.session_state.loading = False
            st.rerun()

    # Run analysis
    if st.session_state.loading and st.session_state.pipeline:
        valid_matches = [m for m in st.session_state.matches if m.get("p1") and m.get("p2")]
        
        progress_bar = st.progress(0, text="Analisi in corso...")
        
        for idx, match in enumerate(valid_matches):
            p1 = match["p1"]
            p2 = match["p2"]
            
            progress_bar.progress((idx + 1) / len(valid_matches), 
                                   text=f"Analisi: {p1} vs {p2}...")
            
            with st.spinner(f"⚡ Analisi {p1} vs {p2}..."):
                result = asyncio.run(
                    analyze_single_match(st.session_state.pipeline, p1, p2)
                )
                st.session_state.results.append({
                    "p1": p1,
                    "p2": p2,
                    "result": result
                })
        
        progress_bar.progress(1.0, text="Analisi completata! ✅")
        st.session_state.loading = False
        st.rerun()

    # Display results
    if st.session_state.results:
        st.divider()
        st.markdown("### 📊 Risultati Analisi")
        
        for idx, res_data in enumerate(st.session_state.results):
            result = res_data["result"]
            p1 = res_data["p1"]
            p2 = res_data["p2"]
            
            st.markdown(f"""
            <div class="match-card">
                <h2 style="color: #e94560;">🎾 {p1} vs {p2}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            m = result.matches[0]
            pred = result.predictions.get(m.id)
            odds = result.odds.get(m.id)
            
            if not pred:
                st.warning("Nessuna predizione disponibile per questo match.")
                continue
            
            p1_prob = pred.p1_win_prob * 100
            p2_prob = pred.p2_win_prob * 100
            fav = p1 if p1_prob >= p2_prob else p2
            fav_pct = max(p1_prob, p2_prob)
            
            # Columns for main stats
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f"""
                <div class="stat-box">
                    <div style="color: #a8a8a8; font-size: 0.8rem;">{p1}</div>
                    <div style="color: {'#e94560' if p1 == fav else '#4ade80'}; 
                                  font-size: 1.8rem; font-weight: bold;">
                        {p1_prob:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="stat-box">
                    <div style="color: #a8a8a8; font-size: 0.8rem;">{p2}</div>
                    <div style="color: {'#e94560' if p2 == fav else '#4ade80'}; 
                                  font-size: 1.8rem; font-weight: bold;">
                        {p2_prob:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                if odds:
                    q_fav = odds.best_odds_home if p1_prob >= p2_prob else odds.best_odds_away
                    q_under = odds.best_odds_away if p1_prob >= p2_prob else odds.best_odds_home
                    book = odds.best_book_home if p1_prob >= p2_prob else odds.best_book_away
                    st.metric("Quota vincente", f"@{q_fav:.2f}", f"{book}")
                else:
                    st.metric("Quote", "N/A")
            with c4:
                poly = result.poly.get(m.id)
                if poly:
                    winner_poly = p1 if poly[0] >= poly[1] else p2
                    st.metric("Polymarket", f"🏆 {winner_poly}", f"crowd {max(poly[0], poly[1]):.0%}")
                else:
                    st.metric("Polymarket", "N/A")
            with c5:
                sisal = result.sisal_flag.get(m.id, ("?", ""))
                st.metric("Sisal", sisal[0], sisal[1])
            
            # Win probability chart
            st.markdown("**Probabilità di Vincita**")
            fig_prob = go.Figure(go.Bar(
                x=[p1, p2],
                y=[p1_prob, p2_prob],
                marker_color=['#e94560' if p1 == fav else '#4ade80', '#e94560' if p2 == fav else '#4ade80'],
                text=[f"{p1_prob:.1f}%", f"{p2_prob:.1f}%"],
                textposition='outside',
            ))
            fig_prob.update_layout(
                height=250,
                template='plotly_dark',
                showlegend=False,
                margin=dict(t=20, b=10),
            )
            fig_prob.update_layout(
                yaxis=dict(range=[0, 100], title="Probabilità (%)"),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig_prob, use_container_width=True)
            
            # Odds comparison chart
            if odds:
                st.markdown("**Confronto Quote**")
                fig_odds = go.Figure(go.Bar(
                    x=[f"{p1} ({odds.best_odds_home:.2f})" if p1_prob >= p2_prob else f"{p2} ({odds.best_odds_home:.2f})",
                       f"{p2} ({odds.best_odds_away:.2f})" if p1_prob >= p2_prob else f"{p1} ({odds.best_odds_away:.2f})"],
                    y=[odds.best_odds_home if p1_prob >= p2_prob else odds.best_odds_away,
                       odds.best_odds_away if p1_prob >= p2_prob else odds.best_odds_home],
                    marker_color=['#f59e0b', '#3b82f6'],
                ))
                fig_odds.update_layout(
                    height=250,
                    template='plotly_dark',
                    showlegend=False,
                    title_text="Quote Migliori",
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_odds, use_container_width=True)
            
            # Polymarket sentiment chart
            poly = result.poly.get(m.id)
            if poly:
                st.markdown("**Sentiment Polymarket**")
                fig_poly = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=poly[0] * 100,
                    title={"text": f"{p1} (crowd)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#e94560"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255,0,0,0.1)"},
                            {'range': [50, 100], 'color': "rgba(0,255,0,0.1)"},
                        ],
                    },
                ))
                fig_poly.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=poly[1] * 100,
                    title={"text": f"{p2} (crowd)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#4ade80"},
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(255,0,0,0.1)"},
                            {'range': [50, 100], 'color': "rgba(0,255,0,0.1)"},
                        ],
                    },
                ))
                fig_poly.update_layout(
                    height=300,
                    template='plotly_dark',
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(fig_poly, use_container_width=True)
            
            # News impact
            news = result.news
            if news:
                st.markdown("**📰 Impatto News**")
                n1 = news.get(p1, {})
                n2 = news.get(p2, {})
                
                news_cols = st.columns(2)
                with news_cols[0]:
                    if n1 and n1.get("article_count"):
                        impact = n1.get("impact", 0)
                        color = "#ef4444" if impact < 0 else ("#22c55e" if impact > 0 else "#a8a8a8")
                        st.markdown(f"""
                        <div style="background: {color}22; padding: 10px; border-radius: 8px; border-left: 4px solid {color};">
                            <strong>{p1}</strong>: impact {impact:+.2f} ({n1.get('article_count', 0)} articoli)
                        </div>
                        """, unsafe_allow_html=True)
                        if n1.get("articles"):
                            for a in n1["articles"][:2]:
                                st.caption(f"  → {a.get('title', 'N/A')}")
                    else:
                        st.markdown(f"<div style='padding:10px;'>{p1}: nessuna news rilevante</div>", unsafe_allow_html=True)
                
                with news_cols[1]:
                    if n2 and n2.get("article_count"):
                        impact = n2.get("impact", 0)
                        color = "#ef4444" if impact < 0 else ("#22c55e" if impact > 0 else "#a8a8a8")
                        st.markdown(f"""
                        <div style="background: {color}22; padding: 10px; border-radius: 8px; border-left: 4px solid {color};">
                            <strong>{p2}</strong>: impact {impact:+.2f} ({n2.get('article_count', 0)} articoli)
                        </div>
                        """, unsafe_allow_html=True)
                        if n2.get("articles"):
                            for a in n2["articles"][:2]:
                                st.caption(f"  → {a.get('title', 'N/A')}")
                    else:
                        st.markdown(f"<div style='padding:10px;'>{p2}: nessuna news rilevante</div>", unsafe_allow_html=True)
            
            # Value bet reasoning
            if result.decisions:
                st.markdown("**💡 Motivazione Decisione**")
                for d in result.decisions:
                    vb = d.value_bet
                    feat = result.features.get(m.id)
                    reason = explain_value_bet(vb, feat)
                    tier = vb.tier.value if hasattr(vb.tier, 'value') else vb.tier
                    
                    st.markdown(f"""
                    <div style="background: #2a2a3e; padding: 15px; border-radius: 10px; 
                                border-left: 4px solid #e94560; margin-bottom: 10px;">
                        <strong style="color: #e94560;">[{d.rank}] PUNTARE SU {vb.player_name} @ {vb.odds:.2f}</strong>
                        <br>
                        <span style="color: #a8a8a8;">
                            Edge: {vb.edge*100:+.1f}% | Tier: {tier} | Sisal: {d.sisal} | Book: {d.bookmaker}
                        </span>
                        <p style="color: #d4d4d4; margin-top: 5px;">{reason}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Excluded value bets
            if result.excluded:
                st.markdown("**❌ Value Scartate**")
                for d in result.excluded:
                    vb = d.value_bet
                    st.markdown(f"  - {vb.player_name} ({vb.match_key}) edge {vb.edge*100:+.1f}% → {d.excluded_reason}")
            
            st.divider()


if __name__ == "__main__":
    main()

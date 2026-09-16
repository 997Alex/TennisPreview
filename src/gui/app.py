"""Streamlit GUI for TennisPreview multi-match analysis."""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import concurrent.futures

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.utils.logging import setup_logging, get_logger, progress
from src.utils.config import config
from src.pipeline.daily import TennisDailyPipeline, PipelineResult, TennisMatch, Player
from src.utils.models import TournamentLevel, Surface
from src.betting.value_detector import explain_value_bet

logger = get_logger("gui")
progress.set_enabled(False)


def init_pipeline():
    setup_logging(log_level="WARNING")
    config.load()
    pipeline = TennisDailyPipeline(demo_mode=False)
    return pipeline


async def analyze_single_match(pipeline: TennisDailyPipeline, p1: str, p2: str) -> Dict:
    await pipeline._load_historical_data()
    result = PipelineResult()
    result.demo_mode = False
    result.demo_odds = False

    match_key = f"{p1} vs {p2}"
    match = TennisMatch(
        id=match_key, tournament="Analisi Singola",
        tournament_level=TournamentLevel.UNKNOWN, surface=Surface.HARD,
        round="", scheduled_time=datetime.now(),
        player1=Player(id=f"p1_{p1}", name=p1),
        player2=Player(id=f"p2_{p2}", name=p2),
    )
    result.matches = [match]
    result.sisal_flag = {match.id: pipeline._sisal_presence(match)}
    odds_dict = await pipeline._get_odds_for_matches([match], demo=False)
    result.odds = odds_dict
    poly_dict = pipeline._get_poly_sentiment([match], demo=False)
    result.poly = poly_dict
    news_impact = pipeline._get_news_impact([match])
    result.news = news_impact
    features_dict = pipeline._build_features([match], odds_dict, news_impact, poly_dict)
    result.features = features_dict
    if not pipeline._models_fitted:
        pipeline._fit_models()
    predictions = pipeline._generate_predictions(features_dict)
    result.predictions = predictions
    value_bets = pipeline._find_value_bets(predictions, odds_dict, [match])
    result.value_bets = value_bets
    decisions, excluded = pipeline._select_decisions(value_bets, features_dict, odds_dict)
    result.decisions = decisions
    result.excluded = excluded
    result.decision_summary = {
        'decisions': len(decisions), 'value_found': len(value_bets),
        'excluded': len(excluded), 'sisal_yes': sum(1 for d in decisions if d.sisal == "si"),
    }
    return result


def run_analysis(pipeline: TennisDailyPipeline, p1: str, p2: str) -> Dict:
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(analyze_single_match(pipeline, p1, p2))
        finally:
            loop.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run)
        return future.result(timeout=120)


def _main_impl():
    st.set_page_config(page_title="TennisPreview", page_icon="🎾", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .main-header { text-align:center; padding:20px; background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460); border-radius:15px; margin-bottom:30px; }
    .main-header h1 { color:#e94560; font-size:2.5rem; margin:0; }
    .main-header p { color:#a8a8a8; font-size:1.1rem; margin-top:5px; }
    .match-card { background:#1e1e2e; border:1px solid #333; border-radius:12px; padding:20px; margin-bottom:15px; }
    .stat-box { background:#2a2a3e; border-radius:10px; padding:15px; text-align:center; }
    .stButton>button { background-color:#e94560; color:white; border-radius:10px; border:none; padding:10px 25px; font-weight:bold; }
    .stButton>button:hover { background-color:#ff6b6b; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header"><h1>🎾 TennisPreview</h1><p>Analisi Multi-Match con Probabilità, Sentiment e Value Betting</p></div>
    """, unsafe_allow_html=True)

    if 'matches' not in st.session_state:
        st.session_state.matches = [{"p1": "", "p2": ""}]
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'loading' not in st.session_state:
        st.session_state.loading = False

    with st.sidebar:
        st.title("⚙️ Configurazione")
        if st.session_state.pipeline is None:
            if st.button("🚀 Inizializza Pipeline", use_container_width=True):
                with st.spinner("Caricamento..."):
                    st.session_state.pipeline = init_pipeline()
                    st.success("Pipeline pronta!")
        st.divider()
        st.subheader("💡 Suggerimenti")
        st.info("Inserisci nomi copiati da Sisal, poi premi ANALIZZA TUTTI")
        st.divider()
        if st.session_state.results:
            st.subheader("📊 Riepilogo")
            total_decisions = sum(len(r.get('decisions', [])) for r in st.session_state.results)
            total_value = sum(r.get('value_found', 0) for r in st.session_state.results)
            st.metric("Decisioni totali", total_decisions)
            st.metric("Value trovate", total_value)

    st.markdown("### ⚡ Inserisci i Match")

    new_list = []
    for i, m in enumerate(st.session_state.matches):
        cols = st.columns([3, 3, 1, 1])
        with cols[0]:
            p1 = st.text_input(f"P1_{i}", value=m.get("p1", ""), placeholder="Es: Alcaraz", key=f"p1_{i}")
        with cols[1]:
            p2 = st.text_input(f"P2_{i}", value=m.get("p2", ""), placeholder="Es. Medvedev", key=f"p2_{i}")
        with cols[2]:
            if i > 0 and st.button("❌", key=f"del_{i}"):
                del st.session_state.matches[i]
                st.rerun()
        with cols[3]:
            if i == len(st.session_state.matches) - 1 and st.button("➕", key=f"add_{i}"):
                st.session_state.matches.append({"p1": "", "p2": ""})
                st.rerun()
        st.session_state.matches[i]["p1"] = p1
        st.session_state.matches[i]["p2"] = p2

    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 ANALIZZA TUTTI", use_container_width=True, type="primary"):
            valid = [m for m in st.session_state.matches if m.get("p1") and m.get("p2")]
            if not valid:
                st.error("Inserisci almeno un match con entrambi i giocatori.")
            elif st.session_state.pipeline is None:
                st.error("Inizializza la pipeline dalla sidebar.")
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

    if st.session_state.loading and st.session_state.pipeline:
        valid = [m for m in st.session_state.matches if m.get("p1") and m.get("p2")]
        progress_bar = st.progress(0, text="Analisi in corso...")
        for idx, match in enumerate(valid):
            p1 = match["p1"]
            p2 = match["p2"]
            progress_bar.progress((idx + 1) / len(valid), text=f"Analisi: {p1} vs {p2}...")
            with st.spinner(f"⚡ {p1} vs {p2}..."):
                try:
                    result = run_analysis(st.session_state.pipeline, p1, p2)
                    st.session_state.results.append({"p1": p1, "p2": p2, "result": result})
                except Exception as e:
                    st.error(f"Errore {p1} vs {p2}: {e}")
                    logger.error(f"Analysis error: {e}", exc_info=True)
                    continue
        progress_bar.progress(1.0, text="Analisi completata! ✅")
        st.session_state.loading = False
        st.rerun()

    if st.session_state.results:
        st.divider()
        st.markdown("### 📊 Risultati Analisi")
        for idx, res_data in enumerate(st.session_state.results):
            result = res_data["result"]
            p1 = res_data["p1"]
            p2 = res_data["p2"]
            m = result.matches[0]
            pred = result.predictions.get(m.id)
            odds = result.odds.get(m.id)
            if not pred:
                st.warning(f"Nessuna predizione per {p1} vs {p2}.")
                continue
            p1_prob = pred.p1_win_prob * 100
            p2_prob = pred.p2_win_prob * 100
            fav = p1 if p1_prob >= p2_prob else p2
            fav_pct = max(p1_prob, p2_prob)

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="stat-box"><div style="color:#a8a8a8;font-size:0.8rem;">{p1}</div><div style="color:{"#e94560" if p1==fav else "#4ade80"};font-size:1.8rem;font-weight:bold;">{p1_prob:.1f}%</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-box"><div style="color:#a8a8a8;font-size:0.8rem;">{p2}</div><div style="color:{"#e94560" if p2==fav else "#4ade80"};font-size:1.8rem;font-weight:bold;">{p2_prob:.1f}%</div></div>', unsafe_allow_html=True)
            with c3:
                if odds:
                    q_fav = odds.best_odds_home if p1_prob>=p2_prob else odds.best_odds_away
                    book = odds.best_book_home if p1_prob>=p2_prob else odds.best_book_away
                    st.metric("Quota", f"@{q_fav:.2f}", book)
                else:
                    st.metric("Quote", "N/A")
            with c4:
                poly = result.poly.get(m.id)
                if poly:
                    w = p1 if poly[0]>=poly[1] else p2
                    st.metric("Polymarket", f"🏆 {w}", f"{max(poly[0],poly[1]):.0%}")
                else:
                    st.metric("Polymarket", "N/A")
            with c5:
                sisal = result.sisal_flag.get(m.id, ("?",""))
                st.metric("Sisal", sisal[0], sisal[1])

            st.markdown("**Probabilità di Vincita**")
            fig_prob = go.Figure(go.Bar(x=[p1,p2], y=[p1_prob,p2_prob],
                marker_color=['#e94560' if p1==fav else '#4ade80','#e94560' if p2==fav else '#4ade80'],
                text=[f"{p1_prob:.1f}%",f"{p2_prob:.1f}%"], textposition='outside'))
            fig_prob.update_layout(height=250, template='plotly_dark', showlegend=False, margin=dict(t=20,b=10))
            fig_prob.update_layout(yaxis=dict(range=[0,100],title="Probabilità (%)"), xaxis=dict(title=""))
            st.plotly_chart(fig_prob, use_container_width=True)

            if odds:
                st.markdown("**Confronto Quote**")
                fig_odds = go.Figure(go.Bar(
                    x=[f"{p1} ({odds.best_odds_home:.2f})" if p1_prob>=p2_prob else f"{p2} ({odds.best_odds_home:.2f})",
                       f"{p2} ({odds.best_odds_away:.2f})" if p1_prob>=p2_prob else f"{p1} ({odds.best_odds_away:.2f})"],
                    y=[odds.best_odds_home if p1_prob>=p2_prob else odds.best_odds_away,
                       odds.best_odds_away if p1_prob>=p2_prob else odds.best_odds_home],
                    marker_color=['#f59e0b','#3b82f6']))
                fig_odds.update_layout(height=250, template='plotly_dark', showlegend=False, title_text="Quote Migliori", margin=dict(t=40,b=10))
                st.plotly_chart(fig_odds, use_container_width=True)

            poly = result.poly.get(m.id)
            if poly:
                st.markdown("**Sentiment Polymarket**")
                fig_poly = go.Figure()
                fig_poly.add_trace(go.Indicator(mode="gauge+number", value=poly[0]*100, title={"text":f"{p1} (crowd)"}, gauge={'axis':{'range':[0,100]},'bar':{'color':"#e94560"},'steps':[{'range':[0,50],'color':"rgba(255,0,0,0.1)"},{'range':[50,100],'color':"rgba(0,255,0,0.1)"}]}))
                fig_poly.add_trace(go.Indicator(mode="gauge+number", value=poly[1]*100, title={"text":f"{p2} (crowd)"}, gauge={'axis':{'range':[0,100]},'bar':{'color':"#4ade80"},'steps':[{'range':[0,50],'color':"rgba(255,0,0,0.1)"},{'range':[50,100],'color':"rgba(0,255,0,0.1)"}]}))
                fig_poly.update_layout(height=300, template='plotly_dark', margin=dict(t=40,b=10))
                st.plotly_chart(fig_poly, use_container_width=True)

            news = result.news
            if news:
                st.markdown("**📰 Impatto News**")
                n1 = news.get(p1, {}); n2 = news.get(p2, {})
                nc1, nc2 = st.columns(2)
                with nc1:
                    if n1 and n1.get("article_count"):
                        impact = n1.get("impact",0)
                        color = "#ef4444" if impact<0 else ("#22c55e" if impact>0 else "#a8a8a8")
                        st.markdown(f'<div style="background:{color}22;padding:10px;border-radius:8px;border-left:4px solid {color};"><strong>{p1}</strong>: impact {impact:+.2f} ({n1.get("article_count",0)} articoli)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding:10px;'>{p1}: nessuna news</div>", unsafe_allow_html=True)
                with nc2:
                    if n2 and n2.get("article_count"):
                        impact = n2.get("impact",0)
                        color = "#ef4444" if impact<0 else ("#22c55e" if impact>0 else "#a8a8a8")
                        st.markdown(f'<div style="background:{color}22;padding:10px;border-radius:8px;border-left:4px solid {color};"><strong>{p2}</strong>: impact {impact:+.2f} ({n2.get("article_count",0)} articoli)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding:10px;'>{p2}: nessuna news</div>", unsafe_allow_html=True)

            if result.decisions:
                st.markdown("**💡 Motivazione Decisione**")
                for d in result.decisions:
                    vb = d.value_bet
                    feat = result.features.get(m.id)
                    reason = explain_value_bet(vb, feat)
                    tier = vb.tier.value if hasattr(vb.tier,'value') else vb.tier
                    st.markdown(f'<div style="background:#2a2a3e;padding:15px;border-radius:10px;border-left:4px solid #e94560;margin-bottom:10px;"><strong style="color:#e94560;">[{d.rank}] PUNTARE SU {vb.player_name} @ {vb.odds:.2f}</strong><br><span style="color:#a8a8a8;">Edge: {vb.edge*100:+.1f}% | Tier: {tier} | Sisal: {d.sisal} | Book: {d.bookmaker}</span><p style="color:#d4d4d4;margin-top:5px;">{reason}</p></div>', unsafe_allow_html=True)

            if result.excluded:
                st.markdown("**❌ Value Scartate**")
                for d in result.excluded:
                    vb = d.value_bet
                    st.markdown(f"  - {vb.player_name} ({vb.match_key}) edge {vb.edge*100:+.1f}% → {d.excluded_reason}")
            st.divider()


def main():
    try:
        _main_impl()
    except Exception as e:
        st.error(f"Errore critico: {e}")
        logger.error(f"Main error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

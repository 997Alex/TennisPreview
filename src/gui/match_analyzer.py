"""GUI module for single-match analysis."""
import asyncio
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, List, Tuple
import threading

from src.utils.logging import get_logger, setup_logging
from src.utils.config import config
from src.pipeline.daily import TennisDailyPipeline, PipelineResult, TennisMatch, Player
from src.data.ingest.espn import ESPNClient, normalize_name
from src.data.ingest.news import NewsIngester
from src.data.ingest.polymarket import PolymarketClient
from src.data.ingest.sackmann import SackmannIngester
from src.data.ingest.theoddsapi import TheOddsAPIClient
from src.data.ingest.sisal import SisalScraper
from src.data.merge.odds_merger import OddsMerger
from src.sports.tennis.models.ensemble import TennisEnsemble
from src.betting.value_detector import TennisValueDetector, explain_value_bet

logger = get_logger("gui")


class MatchAnalyzerApp:
    """Tkinter GUI for single-match tennis analysis."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TennisPreview - Analisi Match")
        self.root.geometry("950x750")
        self.root.minsize(800, 600)
        self.pipeline: Optional[TennisDailyPipeline] = None
        self._analysis_running = False

        self._build_ui()
        self._init_pipeline()

    def _build_ui(self):
        """Build the Tkinter interface."""
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("Sub.TLabel", font=("Helvetica", 10))
        style.configure("Result.TLabel", font=("Consolas", 11))
        style.configure("ResultBold.TLabel", font=("Consolas", 11, "bold"))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))

        # Title
        ttk.Label(self.root, text="TennisPreview - Analisi Singolo Match",
                  style="Title.TLabel").pack(pady=(10, 5))
        ttk.Label(self.root, text="Inserisci i nomi dei giocatori per l'analisi",
                  style="Sub.TLabel").pack(pady=(0, 10))

        # Player inputs frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=5)

        ttk.Label(input_frame, text="Giocatore 1:", font=("Helvetica", 11)).grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.player1_var = tk.StringVar()
        self.p1_entry = ttk.Entry(input_frame, textvariable=self.player1_var, width=30, font=("Helvetica", 11))
        self.p1_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="vs", font=("Helvetica", 11, "bold")).grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="Giocatore 2:", font=("Helvetica", 11)).grid(row=0, column=3, padx=5, pady=5, sticky="e")
        self.player2_var = tk.StringVar()
        self.p2_entry = ttk.Entry(input_frame, textvariable=self.player2_var, width=30, font=("Helvetica", 11))
        self.p2_entry.grid(row=0, column=4, padx=5, pady=5)

        # Analyze button
        self.analyze_btn = ttk.Button(self.root, text="🔍 Analizza Match", command=self._on_analyze)
        self.analyze_btn.pack(pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(pady=5, fill=tk.X, padx=50)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=5, fill=tk.BOTH, expand=True, padx=10)

        # Tab: Risultati
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Risultati")
        self._build_results_tab()

        # Tab: Dettaglio
        self.detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.detail_frame, text="Dettaglio")
        self._build_detail_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Pronto. Inserisci i nomi dei giocatori.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN,
                  anchor=tk.W, font=("Helvetica", 9)).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_results_tab(self):
        """Build the results tab."""
        self.results_text = scrolledtext.ScrolledText(
            self.results_frame, font=("Consolas", 10), wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)

    def _build_detail_tab(self):
        """Build the detail tab."""
        self.detail_text = scrolledtext.ScrolledText(
            self.detail_frame, font=("Consolas", 10), wrap=tk.WORD,
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
        )
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.detail_text.config(state=tk.DISABLED)

    def _init_pipeline(self):
        """Initialize the pipeline in a background thread."""
        def init():
            try:
                setup_logging(log_level="WARNING")
                config.load()
                self.pipeline = TennisDailyPipeline(demo_mode=False)
                self.root.after(0, self._set_status, "Pipeline pronta. Pronto per l'analisi.")
            except Exception as e:
                self.root.after(0, self._set_status, f"Errore inizializzazione: {e}")
                logger.error(f"Pipeline init failed: {e}")

        thread = threading.Thread(target=init, daemon=True)
        thread.start()

    def _on_analyze(self):
        """Handle the analyze button click."""
        p1 = self.player1_var.get().strip()
        p2 = self.player2_var.get().strip()

        if not p1 or not p2:
            messagebox.showwarning("Input Richiesto", "Inserisci entrambi i nomi dei giocatori.")
            return

        if p1.lower() == p2.lower():
            messagebox.showwarning("Errore", "I nomi dei giocatori devono essere diversi.")
            return

        self._analysis_running = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self._set_status(f"Analisi in corso per {p1} vs {p2}...")

        def run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self._analyze_match(p1, p2)
                )
                loop.close()
                self.root.after(0, self._display_results, result)
            except Exception as e:
                logger.error(f"Analysis failed: {e}", exc_info=True)
                self.root.after(0, self._set_status, f"Errore analisi: {e}")
            finally:
                self.root.after(0, self._analysis_complete)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    async def _analyze_match(self, p1: str, p2: str) -> PipelineResult:
        """Analyze a single match between two players."""
        result = PipelineResult()
        result.demo_mode = False
        result.demo_odds = False

        # Load historical data if not already loaded
        if not self.pipeline.historical_matches:
            await self.pipeline._load_historical_data()

        # Create synthetic match
        player1 = Player(name=p1)
        player2 = Player(name=p2)
        match_key = f"{p1} vs {p2}"
        match = TennisMatch(
            id=match_key,
            match_key=match_key,
            player1=player1,
            player2=player2,
            tournament="Torneo Personale",
            level="UNKNOWN",
            surface="Hard",
            round="Finale",
            date=_date.today(),
        )
        result.matches = [match]

        # Get Sisal coverage
        result.sisal_flag = {match.id: self.pipeline._sisal_presence(match)}

        # Get odds
        odds_dict = await self.pipeline._get_odds_for_matches([match], demo=False)
        result.odds = odds_dict

        # Get Polymarket sentiment
        poly_dict = self.pipeline._get_poly_sentiment([match], demo=False)
        result.poly = poly_dict

        # Get news impact
        news_impact = self.pipeline._get_news_impact([match])
        result.news = news_impact

        # Build features
        features_dict = self.pipeline._build_features([match], odds_dict, news_impact, poly_dict)
        result.features = features_dict

        # Ensure models are fitted
        if not self.pipeline._models_fitted:
            self.pipeline._fit_models()

        # Generate predictions
        predictions = self.pipeline._generate_predictions(features_dict)
        result.predictions = predictions

        # Find value bets
        value_bets = self.pipeline._find_value_bets(predictions, odds_dict, [match])
        result.value_bets = value_bets

        # Decisions
        decisions, excluded = self.pipeline._select_decisions(value_bets, features_dict, odds_dict)
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

    def _display_results(self, result: PipelineResult):
        """Display the analysis results in the GUI."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete(1.0, tk.END)

        target_date = __import__('datetime').date.today()
        n_odds = len(result.odds)
        n_m = len(result.matches)

        # Results tab
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, f"  RISULTATI MATCH  (LIVE)\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")

        if not result.matches:
            self.results_text.insert(tk.END, "Nessun match trovato.\n")
            self.results_text.config(state=tk.DISABLED)
            return

        m = result.matches[0]
        pred = result.predictions.get(m.id)
        odds = result.odds.get(m.id)

        if pred:
            p1_prob = pred.p1_win_prob * 100
            p2_prob = pred.p2_win_prob * 100
            fav = m.player1.name if p1_prob >= p2_prob else m.player2.name
            fav_pct = max(p1_prob, p2_prob)

            # Main result
            self.results_text.insert(tk.END, f"\n  {'Match':<40}\n", "title")
            self.results_text.insert(tk.END, f"  {m.match_key[:38]}\n\n")

            self.results_text.insert(tk.END, "  " + "-" * 50 + "\n")
            self.results_text.insert(tk.END, f"  {'Giocatore':<25} {'Prob.':>8}\n")
            self.results_text.insert(tk.END, f"  {'-'*25} {'-'*8}\n")
            self.results_text.insert(tk.END, f"  {m.player1.name:<25} {pred.p1_win_prob*100:>7.1f}%\n")
            self.results_text.insert(tk.END, f"  {m.player2.name:<25} {pred.p2_win_prob*100:>7.1f}%\n")
            self.results_text.insert(tk.END, "\n")

            # Winner announcement
            if fav_pct >= 70:
                self.results_text.insert(tk.END,
                    f"  >>> FAVORITO: {fav} ({fav_pct:.1f}%)\n\n", "winner")
            elif fav_pct >= 55:
                self.results_text.insert(tk.END,
                    f"  >>> Legger vantaggio: {fav} ({fav_pct:.1f}%)\n\n")
            else:
                self.results_text.insert(tk.END,
                    f"  >>> Pronostico incerto ({fav_pct:.1f}%)\n\n")

            # Odds info
            if odds:
                q1 = odds.best_odds_home if p1_prob >= p2_prob else odds.best_odds_away
                q2 = odds.best_odds_away if p1_prob >= p2_prob else odds.best_odds_home
                book = odds.best_book_home if p1_prob >= p2_prob else odds.best_book_away
                self.results_text.insert(tk.END,
                    f"  Quote Sisal: {fav} @{q1:.2f} ({book})\n")
                self.results_text.insert(tk.END,
                    f"  Avversario: @{q2:.2f}\n\n")

            # Polymarket sentiment
            poly = result.poly.get(m.id)
            if poly:
                self.results_text.insert(tk.END,
                    f"  Sentiment Polymarket: {poly[0]:.1%} vs {poly[1]:.1%} (crowd)\n\n")

            # Sisal flag
            sisal = result.sisal_flag.get(m.id, ("?", ""))
            self.results_text.insert(tk.END,
                f"  Presente in Palinsesto Sisal: {sisal[0]} {sisal[1]}\n")

            # Decisions
            if result.decisions:
                self.results_text.insert(tk.END, "\n  " + "=" * 50 + "\n")
                self.results_text.insert(tk.END, "  DECISIONI FINALI\n")
                self.results_text.insert(tk.END, "=" * 50 + "\n\n")
                for d in result.decisions:
                    vb = d.value_bet
                    tier = vb.tier.value if hasattr(vb.tier, 'value') else vb.tier
                    self.results_text.insert(tk.END,
                        f"  [{d.rank}] PUNTARE SU {vb.player_name} @ {vb.odds:.2f}\n"
                        f"    Edge: {vb.edge*100:+.1f}% | Tier: {tier} | Sisal: {d.sisal}\n")
                    self.results_text.insert(tk.END, "\n")
            else:
                self.results_text.insert(tk.END, "  Nessuna decisione (nessun edge sopra soglia).\n")

            self.results_text.insert(tk.END, "\n")
            summ = result.decision_summary or {}
            self.results_text.insert(tk.END,
                f"  Match analizzati: {len(result.matches)}\n"
                f"  Value trovate: {summ.get('value_found', len(result.value_bets))}\n"
                f"  Decisioni finali: {summ.get('decisions', len(result.decisions))}\n")

        else:
            self.results_text.insert(tk.END, "Nessuna predizione disponibile.\n")

        self.results_text.config(state=tk.DISABLED)

        # Detail tab - full reasoning
        if pred and odds:
            self.detail_text.insert(tk.END, "=" * 70 + "\n")
            self.detail_text.insert(tk.END, "  ANALISI COMPLETA DEL MATCH\n")
            self.detail_text.insert(tk.END, "=" * 70 + "\n\n")

            m = result.matches[0]
            feat = result.features.get(m.id)
            news = result.news
            poly = result.poly.get(m.id)
            p1_prob = pred.p1_win_prob * 100
            p2_prob = pred.p2_win_prob * 100

            # Player comparison
            self.detail_text.insert(tk.END, "  COMPARAZIONE GIOCATORI\n")
            self.detail_text.insert(tk.END, "-" * 50 + "\n")
            if feat:
                self.detail_text.insert(tk.END,
                    f"  Superficie: {feat.surface.value if hasattr(feat.surface, 'value') else feat.surface}\n")
                self.detail_text.insert(tk.END,
                    f"  Elo superficie: {m.player1.name[:20]} {feat.p1_surface_elo:.0f} vs "
                    f"{m.player2.name[:20]} {feat.p2_surface_elo:.0f}\n")
                self.detail_text.insert(tk.END,
                    f"  Forma (ultime 10): {m.player1.name[:20]} {feat.p1_form_10*10:.0f}V/"
                    f"{10-feat.p1_form_10*10:.0f}P vs {m.player2.name[:20]} {feat.p2_form_10*10:.0f}V/"
                    f"{10-feat.p2_form_10*10:.0f}P\n")
                self.detail_text.insert(tk.END,
                    f"  Battuta: {feat.p1_serve_rating:.0f} vs {feat.p2_serve_rating:.0f}\n"
                    f"  Risposta: {feat.p1_return_rating:.0f} vs {feat.p2_return_rating:.0f}\n")
                self.detail_text.insert(tk.END, "\n")

            # H2H
            if feat and feat.h2h_total > 0:
                self.detail_text.insert(tk.END, "  HEAD TO HEAD\n")
                self.detail_text.insert(tk.END, "-" * 50 + "\n")
                w1 = feat.h2h_p1_wins
                w2 = feat.h2h_total - feat.h2h_p1_wins
                if feat.h2h_surface > 0:
                    ws = feat.h2h_p1_wins_surface
                    wss = feat.h2h_surface - feat.h2h_p1_wins_surface
                    self.detail_text.insert(tk.END,
                        f"  Totale: {w1}-{w2} | Su {feat.surface.value if hasattr(feat.surface, 'value') else feat.surface}: {ws}-{wss}\n")
                else:
                    self.detail_text.insert(tk.END, f"  Totale: {w1}-{w2}\n")
                self.detail_text.insert(tk.END, "\n")

            # News impact
            if news:
                self.detail_text.insert(tk.END, "  IMPATTO NEWS\n")
                self.detail_text.insert(tk.END, "-" * 50 + "\n")
                n1 = news.get(m.player1.name, {})
                n2 = news.get(m.player2.name, {})
                for name, data in [(m.player1.name, n1), (m.player2.name, n2)]:
                    if data and data.get("article_count"):
                        impact = data.get("impact", 0)
                        sign = "+" if impact > 0 else ("-" if impact < 0 else "~")
                        self.detail_text.insert(tk.END,
                            f"  {name}: {sign} impatto {impact:+.2f} ({data.get('article_count', 0)} articoli)\n")
                        if data.get("articles"):
                            for a in data["articles"][:2]:
                                self.detail_text.insert(tk.END, f"    - {a.get('title', 'N/A')}\n")
                self.detail_text.insert(tk.END, "\n")

            # Polymarket
            if poly:
                self.detail_text.insert(tk.END, "  POLYMARKET SENTIMENT\n")
                self.detail_text.insert(tk.END, "-" * 50 + "\n")
                self.detail_text.insert(tk.END,
                    f"  Crowd: {poly[0]:.1%} vs {poly[1]:.1%} (senza vig)\n\n")

            # Value bet reasoning
            if result.decisions:
                self.detail_text.insert(tk.END, "  MOTIVAZIONE DECISIONE\n")
                self.detail_text.insert(tk.END, "-" * 50 + "\n")
                for d in result.decisions:
                    vb = d.value_bet
                    reason = explain_value_bet(vb, feat)
                    self.detail_text.insert(tk.END,
                        f"  PUNTARE SU {vb.player_name} @ {vb.odds:.2f} ({d.bookmaker})\n"
                        f"  Edge: {vb.edge*100:+.1f}% | Tier: {vb.tier.value if hasattr(vb.tier, 'value') else vb.tier}\n")
                    self.detail_text.insert(tk.END, f"  {reason}\n\n")

            # Excluded value bets
            if result.excluded:
                self.detail_text.insert(tk.END, "  VALUE SCARTATE\n")
                self.detail_text.insert(tk.END, "-" * 50 + "\n")
                for d in result.excluded:
                    vb = d.value_bet
                    self.detail_text.insert(tk.END,
                        f"  - {vb.player_name} ({vb.match_key}) "
                        f"edge {vb.edge*100:+.1f}% -> {d.excluded_reason}\n")

        self.detail_text.config(state=tk.DISABLED)

        # Switch to results tab
        self.notebook.select(self.results_frame)

    def _analysis_complete(self):
        """Reset UI after analysis completes."""
        self._analysis_running = False
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        self._set_status("Analisi completata. Seleziona un'altra tabella per dettagli.")

    def _set_status(self, msg: str):
        self.status_var.set(msg)

#!/usr/bin/env python3
"""Backtest con QUOTE VERE (Betexplorer archive) su 3 Slam x ATP/WTA.

Per ogni evento: train < cutoff (profili+fit, no lookahead) -> test sui
match reali con (a) outcome noto, (b) best-odds BE. Metriche per varianti
di modello (ablation), sweep del peso mercato, ROI vs quote reali a 1u flat.

Uso:
  python3 scripts/backtest_odds.py --events uso2025_atp uso2025_wta
  python3 scripts/backtest_odds.py --events ao2026_atp ao2026_wta rg2026_atp rg2026_wta
"""
import argparse
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.ingest.sackmann import SackmannIngester
from src.data.ingest.betexplorer import BetexplorerClient, names_match
from src.data.ingest.espn import normalize_name
from src.data.merge.odds_merger import OddsMerger
from src.sports.tennis.features.builder import TennisFeatureBuilder
from src.sports.tennis.models.ensemble import TennisEnsemble
from src.utils.config import config  # noqa: F401
from src.utils.logging import setup_logging, get_logger
from src.utils.models import Player, Surface, TennisMatch, MergedOdds

logger = get_logger("backtest_odds")

EVENTS = {
    "uso2025_atp": dict(tour="ATP", sack="Us Open", cutoff=20250825,
                        be="it/tennis/atp-singles/us-open-2025/"),
    "uso2025_wta": dict(tour="WTA", sack="Us Open", cutoff=20250825,
                        be="it/tennis/wta-singles/us-open-2025/"),
    "ao2026_atp": dict(tour="ATP", sack="Australian Open", cutoff=20260119,
                       be="it/tennis/atp-singles/australian-open/"),
    "ao2026_wta": dict(tour="WTA", sack="Australian Open", cutoff=20260119,
                       be="it/tennis/wta-singles/australian-open/"),
    "rg2026_atp": dict(tour="ATP", sack="Roland Garros", cutoff=20260525,
                       be="it/tennis/atp-singles/french-open/"),
    "rg2026_wta": dict(tour="WTA", sack="Roland Garros", cutoff=20260525,
                       be="it/tennis/wta-singles/french-open/"),
}

TIER1 = dict(min_edge=0.025, min_prob=0.10, max_odds=4.0)
SWEEP_KS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]


def norm_id(v):
    try:
        if v is None:
            return None
        if isinstance(v, float):
            return None if math.isnan(v) else str(int(v))
        s = str(v).strip()
        return None if s in ("", "nan", "None") else s
    except Exception:
        return None


def parse_dt(raw):
    try:
        if hasattr(raw, "strftime"):
            return raw.to_pydatetime() if hasattr(raw, "to_pydatetime") else raw
        s = str(int(float(str(raw)))) if str(raw).replace(".", "", 1).isdigit() else str(raw)
        return datetime.strptime(s, "%Y%m%d")
    except Exception:
        return datetime(2020, 1, 1)


def shin(o1, o2):
    ph, pa = 1.0 / o1, 1.0 / o2
    tot = ph + pa
    return (ph / tot, pa / tot) if tot > 0 else (0.5, 0.5)


def ece(probs, labels, bins=10):
    n = len(probs)
    if not n:
        return 0.0
    tot = 0.0
    for b in range(bins):
        idx = [i for i, p in enumerate(probs)
               if (max(p, 1 - p) >= 0.5 + b * 0.05 and max(p, 1 - p) < 0.5 + (b + 1) * 0.05)]
        if not idx:
            continue
        acc = sum(1 for i in idx if (probs[i] >= 0.5) == (labels[i] == 1)) / len(idx)
        conf = sum(max(probs[i], 1 - probs[i]) for i in idx) / len(idx)
        tot += abs(acc - conf) * len(idx) / n
    return tot


def metrics(probs, labels):
    n = len(probs)
    if not n:
        return dict(n=0)
    pc = [min(max(p, 1e-6), 1 - 1e-6) for p in probs]
    wins = sum(1 for p, y in zip(probs, labels) if (p >= 0.5) == (y == 1))
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    ll = -sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(pc, labels)) / n
    return dict(n=n, wins=wins, winrate=round(wins / n, 4),
                brier=round(brier, 4), logloss=round(ll, 4), ece=round(ece(probs, labels), 4))


def run_event(ev_name, df_all, rankings, be_client, years_note="",
              max_train=15000, max_baseline=60000, seed=7):
    cfg = EVENTS[ev_name]
    rng = random.Random(seed)
    sack = SackmannIngester()
    fbuilder = TennisFeatureBuilder()
    tour, cutoff = cfg["tour"], cfg["cutoff"]

    train_df = df_all[df_all["tourney_date"] < cutoff].copy()
    test_df = df_all[(df_all["tour"] == tour) & (df_all["tourney_name"] == cfg["sack"])
                     & (df_all["tourney_date"] == cutoff)].copy()
    logger.info(f"[{ev_name}] train {len(train_df):,} | test {len(test_df)}")

    # profili solo train
    pdb = {}
    for tv in ("ATP", "WTA"):
        sub = train_df[train_df["tour"] == tv]
        if not sub.empty:
            pdb.update(sack.build_player_profiles(sub.copy(), rankings))
    train_dicts = train_df.to_dict("records")

    # training set bilanciato
    hist = sorted(train_dicts, key=lambda m: str(m.get("tourney_date", "19000101")))
    if len(hist) > max_train:
        hist = hist[-max_train:]
    feats, labels = [], []
    rr = random.Random(42)
    for m in hist:
        wid, lid = norm_id(m.get("winner_id")), norm_id(m.get("loser_id"))
        if not wid or not lid:
            continue
        w, lo = pdb.get(wid), pdb.get(lid)
        if not w or not lo:
            continue
        surf = sack.parse_surface(m.get("surface", "Hard"))
        lvl = sack.parse_tournament_level(m.get("tourney_level", ""), m.get("tour", "atp"))
        w_is_p1 = rr.random() < 0.5
        tm = TennisMatch(
            id=f"tr_{m.get('tourney_date')}_{wid}_{lid}_{int(w_is_p1)}",
            tournament=m.get("tourney_name", "H"), tournament_level=lvl,
            surface=surf, round=m.get("round", "R64"),
            player1=w if w_is_p1 else lo, player2=lo if w_is_p1 else w,
            scheduled_time=parse_dt(m.get("tourney_date", "20200101")))
        feats.append(fbuilder.build_features(match=tm, player_db=pdb, merged_odds=None,
                                             news_impact=None, h2h_data=None))
        labels.append(1 if w_is_p1 else 0)

    base = sorted(train_dicts, key=lambda m: str(m.get("tourney_date", "19000101")))
    if len(base) > max_baseline:
        base = base[-max_baseline:]
    ens = TennisEnsemble()
    ens.fit(historical_matches=base, players=pdb, features_list=feats, labels=labels)

    # quote vere BE per l'evento
    be_rows = be_client.get_results_for_event(
        f"https://www.betexplorer.com/{cfg['be']}")
    logger.info(f"[{ev_name}] BE righe con quote: {len(be_rows)}")

    # match Sackmann <-> BE per coppia nomi (univoca nell'evento)
    test_rows = test_df.to_dict("records")
    matched, skipped_score, unmatched = [], 0, 0
    agree = disagree = 0
    for m in test_rows:
        score = str(m.get("score", "") or "")
        if not score or score.strip().upper().startswith("W"):
            skipped_score += 1  # WO/walkover: quote void, fuori metriche
            continue
        wn, ln = str(m.get("winner_name", "")), str(m.get("loser_name", ""))
        cands = []
        for r in be_rows:
            d = names_match(r.be_name1, wn) and names_match(r.be_name2, ln)
            s = names_match(r.be_name1, ln) and names_match(r.be_name2, wn)
            if d ^ s:
                cands.append((r, d))
        if len(cands) != 1:
            unmatched += 1
            continue
        r, direct = cands[0]
        # orienta quote a (winner, loser) di Sackmann
        ow, ol = (r.odds1, r.odds2) if direct else (r.odds2, r.odds1)
        # winner agreement BE (<strong>) vs Sackmann (surname matching:
        # BE abbrevia "Sabalenka A." vs "Aryna Sabalenka")
        if r.winner_is_p1 is not None:
            be_winner = r.be_name1 if r.winner_is_p1 else r.be_name2
            from src.data.ingest.betexplorer import names_match as _nm
            if _nm(be_winner, wn):
                agree += 1
            else:
                disagree += 1
                if disagree <= 5:
                    logger.warning(f"disaccordo esito: BE {be_winner} "
                                   f"vs Sackmann {wn} ({ln})")
        matched.append((m, ow, ol))

    # predizioni varianti (orientamento random per metriche non distorte)
    variants = {k: ([], []) for k in
                ["bt", "elo", "slog", "mkt", "full_raw", "full_cal"]}
    roi_rows = []
    roi_raw_rows = []
    n_partial = 0
    bt = ens.baseline_models.get("bradley_terry")
    elo_m = ens.baseline_models.get("elo_form")
    slog_m = ens.ml_models.get("surface_logistic")

    for m, ow, ol in matched:
        wid, lid = norm_id(m.get("winner_id")), norm_id(m.get("loser_id"))
        w, lo = pdb.get(wid), pdb.get(lid)
        if not w or not lo:
            continue
        surf = sack.parse_surface(m.get("surface", "Hard"))
        lvl = sack.parse_tournament_level(m.get("tourney_level", ""), m.get("tour", "atp"))
        w_is_p1 = rng.random() < 0.5
        p1, p2 = (w, lo) if w_is_p1 else (lo, w)
        label = 1 if w_is_p1 else 0
        tm = TennisMatch(
            id=f"bt_{m.get('tourney_date')}_{wid}_{lid}", tournament=m.get("tourney_name", "?"),
            tournament_level=lvl, surface=surf, round=m.get("round", "?"),
            player1=p1, player2=p2, scheduled_time=parse_dt(m.get("tourney_date", "20200101")))
        # quote orientate a p1/p2
        if w_is_p1:
            op1, op2 = ow, ol
        else:
            op1, op2 = ol, ow
        imp1, _imp2 = shin(op1, op2)
        mo = MergedOdds(match_id=tm.id, best_odds_home=op1, best_odds_away=op2,
                        best_book_home="betexplorer_best", best_book_away="betexplorer_best",
                        implied_prob_home=imp1, implied_prob_away=1 - imp1, vig=0.0,
                        avg_odds_home=op1, avg_odds_away=op2, sources_count=1)
        f_nomkt = fbuilder.build_features(match=tm, player_db=pdb, merged_odds=None,
                                          news_impact=None, h2h_data=None)
        f_mkt = fbuilder.build_features(match=tm, player_db=pdb, merged_odds=mo,
                                        news_impact=None, h2h_data=None)
        if getattr(f_mkt, "partial_profile", False):
            n_partial += 1
        probs = {}
        try:
            probs["bt"] = bt.predict(p1.id, p2.id, surf) if bt else 0.5
        except Exception:
            probs["bt"] = 0.5
        try:
            probs["elo"] = float(elo_m.predict(f_nomkt)) if elo_m else 0.5
        except Exception:
            probs["elo"] = 0.5
        try:
            probs["slog"] = float(slog_m.predict_proba(f_nomkt)) if getattr(
                slog_m, "is_fitted", False) else 0.5
        except Exception:
            probs["slog"] = 0.5
        probs["mkt"] = imp1
        try:
            probs["full_raw"] = float(ens._predict_uncalibrated(f_mkt).p1_win_prob)
        except Exception:
            probs["full_raw"] = 0.5
        try:
            probs["full_cal"] = float(ens.predict(f_mkt).p1_win_prob)
        except Exception:
            probs["full_cal"] = 0.5
        for k, p in probs.items():
            variants[k][0].append(float(p))
            variants[k][1].append(label)
        # ROI live-like: pick argmax, edge vs chiusura BE (sia cal che raw)
        for _key, _pc, _store in (("cal", probs["full_cal"], roi_rows),
                                  ("raw", probs["full_raw"], roi_raw_rows)):
            _pick = _pc >= 0.5
            _pp, _op = (_pc, op1) if _pick else (1 - _pc, op2)
            _ip = imp1 if _pick else 1 - imp1
            _won = ((_pick and label == 1) or ((not _pick) and label == 0))
            _store.append(dict(edge=_pp - _ip, won=_won, odds=_op, prob=_pp))

    out = {
        "event": ev_name, "cutoff": cutoff,
        "n_test_sack": len(test_rows), "skipped_wo": skipped_score,
        "n_matched_odds": len(matched), "unmatched": unmatched,
        "winner_agreement": {"agree": agree, "disagree": disagree},
        "n_partial": n_partial,
        "variants": {k: metrics(*v) for k, v in variants.items()},
    }
    # ROI a soglie edge (Tier1 2.5% + extra), sia cal che raw
    for thr in (0.025, 0.05, 0.08):
        for _tag, _store in (("", roi_rows), ("_raw", roi_raw_rows)):
            bets = [r for r in _store if r["edge"] >= thr]
            if not bets:
                out[f"roi{_tag}_{int(thr*1000)}"] = dict(n=0)
                continue
            w = sum(1 for r in bets if r["won"])
            pnl = sum((r["odds"] - 1) if r["won"] else -1 for r in bets)
            out[f"roi{_tag}_{int(thr*1000)}"] = dict(
                n=len(bets), wins=w, winrate=round(w / len(bets), 4),
                roi=round(pnl / len(bets), 4),
            avg_edge=round(sum(r["edge"] for r in bets) / len(bets), 4),
            avg_odds=round(sum(r["odds"] for r in bets) / len(bets), 2))
    return out, ens, pdb, sack


def sweep_market(ens, pdb, sack, df_all, ev_name, ks=SWEEP_KS, seed=11, n_sample=400):
    """Brier/logloss al variare del peso mercato (Tier1), senza refit modelli."""
    import numpy as np
    cfg = EVENTS[ev_name]
    rows = df_all[(df_all["tour"] == cfg["tour"]) & (df_all["tourney_name"] == cfg["sack"])
                  & (df_all["tourney_date"] == cfg["cutoff"])].to_dict("records")
    fbuilder = TennisFeatureBuilder()
    be = BetexplorerClient()
    be_rows = be.get_results_for_event(f"https://www.betexplorer.com/{cfg['be']}")
    rng = random.Random(seed)
    data = []
    for m in rows:
        wn, ln = str(m.get("winner_name", "")), str(m.get("loser_name", ""))
        score = str(m.get("score", "") or "")
        if not score or score.strip().upper().startswith("W"):
            continue
        cands = []
        for r in be_rows:
            d = names_match(r.be_name1, wn) and names_match(r.be_name2, ln)
            s = names_match(r.be_name1, ln) and names_match(r.be_name2, wn)
            if d ^ s:
                cands.append((r, d))
        if len(cands) != 1:
            continue
        r, direct = cands[0]
        wid, lid = norm_id(m.get("winner_id")), norm_id(m.get("loser_id"))
        w, lo = pdb.get(wid), pdb.get(lid)
        if not w or not lo:
            continue
        surf = sack.parse_surface(m.get("surface", "Hard"))
        lvl = sack.parse_tournament_level(m.get("tourney_level", ""), m.get("tour", "atp"))
        w_is_p1 = rng.random() < 0.5
        p1, p2 = (w, lo) if w_is_p1 else (lo, w)
        ow, ol = (r.odds1, r.odds2) if direct else (r.odds2, r.odds1)
        op1, op2 = (ow, ol) if w_is_p1 else (ol, ow)
        imp1, _ = shin(op1, op2)
        tm = TennisMatch(id="sw", tournament="x", tournament_level=lvl, surface=surf,
                         round="x", player1=p1, player2=p2,
                         scheduled_time=parse_dt(m.get("tourney_date", "20200101")))
        mo = MergedOdds(match_id="sw", best_odds_home=op1, best_odds_away=op2,
                        best_book_home="b", best_book_away="b",
                        implied_prob_home=imp1, implied_prob_away=1 - imp1, vig=0.0,
                        avg_odds_home=op1, avg_odds_away=op2, sources_count=1)
        f = fbuilder.build_features(match=tm, player_db=pdb, merged_odds=mo,
                                    news_impact=None, h2h_data=None)
        data.append((f, 1 if w_is_p1 else 0))
        if len(data) >= n_sample:
            break
    feats = [d[0] for d in data]
    labels = [d[1] for d in data]
    res = {}
    base_w = dict(ens.weights_tier1)
    for k in ks:
        ens.weights_tier1 = dict(base_w)
        ens.weights_tier1["market_implied"] = base_w.get("market_implied", 0.10) * k
        ens._fit_calibrators(feats, labels)
        probs = [float(ens.predict(f).p1_win_prob) for f in feats]
        res[str(k)] = metrics(probs, labels)
    ens.weights_tier1 = base_w
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", default=list(EVENTS),
                    choices=list(EVENTS))
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2027)))
    ap.add_argument("--sweep", action="store_true", help="sweep peso mercato")
    args = ap.parse_args()
    setup_logging(log_level="WARNING")

    sack = SackmannIngester()
    atp = sack.load_matches("atp", years=args.years)
    wta = sack.load_matches("wta", years=args.years)
    df_all = pd.concat([d for d in (atp, wta) if d is not None and not d.empty],
                       ignore_index=True)
    df_all["tourney_date"] = pd.to_numeric(df_all["tourney_date"], errors="coerce")
    df_all = df_all.dropna(subset=["tourney_date"])
    df_all["tourney_date"] = df_all["tourney_date"].astype(int)
    rdf = pd.concat([sack.load_rankings("atp"), sack.load_rankings("wta")],
                    ignore_index=True)
    rdf["ranking_date"] = pd.to_numeric(rdf["ranking_date"], errors="coerce")
    be_client = BetexplorerClient()

    for ev in args.events:
        cfg = EVENTS[ev]
        rk = rdf[rdf["ranking_date"] < cfg["cutoff"]].copy()
        out, ens, pdb, _s = run_event(ev, df_all, rk, be_client)
        if args.sweep:
            out["market_sweep"] = sweep_market(ens, pdb, _s, df_all, ev)
        fp = Path("output") / f"backtest_odds_{ev}.json"
        fp.write_text(json.dumps(out, indent=2, default=str))
        print(f"\n===== {ev} =====")
        print(f"match: {out['n_test_sack']} sack | {out['n_matched_odds']} con quote "
              f"| WO saltati: {out['skipped_wo']} | agreement BE-Sackmann: "
              f"{out['winner_agreement']}")
        for k, m in out["variants"].items():
            print(f"  {k:9s} n={m.get('n',0):>3} winrate={m.get('winrate',0):.2%} "
                  f"brier={m.get('brier',0):.4f} logloss={m.get('logloss',0):.4f} "
                  f"ece={m.get('ece',0):.4f}")
        for thr in ("roi_25", "roi_50", "roi_80",
                    "roi_raw_25", "roi_raw_50", "roi_raw_80"):
            r = out[thr]
            if r.get("n"):
                print(f"  {thr}: n={r['n']} W={r['wins']} roi={r['roi']:+.2%} "
                      f"(edge {r['avg_edge']:.1%}, quota {r['avg_odds']})")
            else:
                print(f"  {thr}: nessun bet")
        if args.sweep:
            print("  sweep peso mercato (Brier):",
                  {k: v["brier"] for k, v in out["market_sweep"].items()})
        print(f"  salvato: {fp}")


if __name__ == "__main__":
    main()

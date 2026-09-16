#!/usr/bin/env python3
"""Backtest settimanale dei pronostici TennisPreview su risultati storici.

Metodologia (niente lookahead, niente news):
  - train: soli match con tourney_date < TEST_START (profili giocatori, training
    set ML con label bilanciate, fit ensemble) -> nessun dato futuro usato
  - test: match con TEST_START <= tourney_date <= TEST_END, esito noto
    (winner_id) confrontato con il pronostico del modello
  - news_impact sempre None (le news altererebbero il test, come richiesto)

Uso:
  python3 scripts/backtest_week.py --start 20260519 --end 20260525
  python3 scripts/backtest_week.py --start 20260903 --end 20260909
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.sackmann import SackmannIngester
from src.sports.tennis.features.builder import TennisFeatureBuilder
from src.sports.tennis.models.ensemble import TennisEnsemble
from src.utils.config import config  # noqa: F401  (init singleton)
from src.utils.logging import setup_logging, get_logger
from src.utils.models import Player, Surface, TennisMatch

logger = get_logger("backtest")


def norm_id(v):
    """Normalize Sackmann player ids (int/float/str/NaN) to str or None."""
    try:
        import math
        if v is None:
            return None
        if isinstance(v, float):
            if math.isnan(v):
                return None
            return str(int(v))
        s = str(v).strip()
        if s in ("", "nan", "None"):
            return None
        return s
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


def build_h2h_index(train_dicts):
    index = {}
    for m in train_dicts:
        try:
            w, loser = m.get("winner_name", ""), m.get("loser_name", "")
            if not w or not loser:
                continue
            key = tuple(sorted([str(w), str(loser)]))
            index.setdefault(key, []).append(
                {
                    "winner": w,
                    "loser": loser,
                    "surface": m.get("surface", "Hard"),
                    "date": m.get("tourney_date", ""),
                    "score": m.get("score", ""),
                }
            )
        except Exception:
            continue
    return index


def run_backtest(test_start: int, test_end: int, years, max_train=15000,
                 max_baseline=60000, seed=7):
    rng = random.Random(seed)
    sack = SackmannIngester()
    fbuilder = TennisFeatureBuilder()

    logger.info(f"Carico match {years[0]}-{years[-1]}...")
    atp = sack.load_matches("atp", years=years)
    wta = sack.load_matches("wta", years=years)
    import pandas as pd

    df = pd.concat([d for d in (atp, wta) if d is not None and not d.empty],
                   ignore_index=True)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")
    df = df.dropna(subset=["tourney_date"])
    df["tourney_date"] = df["tourney_date"].astype(int)

    train_df = df[df["tourney_date"] < test_start].copy()
    test_df = df[(df["tourney_date"] >= test_start) & (df["tourney_date"] <= test_end)].copy()
    logger.info(f"Train: {len(train_df):,} match (< {test_start}) | "
                f"Test: {len(test_df):,} match [{test_start}-{test_end}]")

    if test_df.empty:
        return {"test_start": test_start, "test_end": test_end,
                "n_test": 0, "message": "nessun match nella finestra"}

    # Ranking solo pre-test (niente lookahead nemmeno lì)
    atp_r = sack.load_rankings("atp")
    wta_r = sack.load_rankings("wta")
    rank_frames = []
    for rdf in (atp_r, wta_r):
        if rdf is not None and not rdf.empty:
            rdf = rdf.copy()
            rdf["ranking_date"] = pd.to_numeric(rdf["ranking_date"], errors="coerce")
            rdf = rdf.dropna(subset=["ranking_date"])
            rank_frames.append(rdf[rdf["ranking_date"] < test_start])
    rankings = pd.concat(rank_frames, ignore_index=True) if rank_frames else None

    # Profili solo su train (split ATP/WTA come la pipeline)
    logger.info("Costruisco profili giocatori (solo train)...")
    pdb = {}
    for tour_val in ("ATP", "WTA"):
        sub = train_df[train_df["tour"] == tour_val]
        if sub.empty:
            continue
        pdb.update(sack.build_player_profiles(sub.copy(), rankings))

    train_dicts = train_df.to_dict("records")
    h2h_index = build_h2h_index(train_dicts)

    # Training set bilanciato (50% winner=P1/label 1, 50% winner=P2/label 0)
    hist_sorted = sorted(train_dicts, key=lambda m: str(m.get("tourney_date", "19000101")))
    if len(hist_sorted) > max_train:
        hist_sorted = hist_sorted[-max_train:]
    feats, labels = [], []
    rng_train = random.Random(42)
    for m in hist_sorted:
        wid, lid = norm_id(m.get("winner_id")), norm_id(m.get("loser_id"))
        if not wid or not lid:
            continue
        w, loser = pdb.get(wid), pdb.get(lid)
        if not w or not loser:
            continue
        surf = sack.parse_surface(m.get("surface", "Hard"))
        lvl = sack.parse_tournament_level(m.get("tourney_level", ""), m.get("tour", "atp"))
        w_is_p1 = rng_train.random() < 0.5
        tm = TennisMatch(
            id=f"tr_{m.get('tourney_date')}_{wid}_{lid}_{int(w_is_p1)}",
            tournament=m.get("tourney_name", "Historical"),
            tournament_level=lvl, surface=surf, round=m.get("round", "R64"),
            player1=w if w_is_p1 else loser, player2=loser if w_is_p1 else w,
            scheduled_time=parse_dt(m.get("tourney_date", "20200101")),
        )
        feats.append(fbuilder.build_features(match=tm, player_db=pdb,
                                             merged_odds=None, news_impact=None,
                                             h2h_data=None))
        labels.append(1 if w_is_p1 else 0)

    # Fit ensemble su train (con cap come la pipeline)
    base = sorted(train_dicts, key=lambda m: str(m.get("tourney_date", "19000101")))
    if len(base) > max_baseline:
        base = base[-max_baseline:]
    ens = TennisEnsemble()
    logger.info(f"Fit ensemble su {len(base):,} match + {len(feats):,} sample ML...")
    ens.fit(historical_matches=base, players=pdb, features_list=feats, labels=labels)

    # Pronostici sui test (orientamento random per non gonfiare i 50/50)
    test_rows = sorted(test_df.to_dict("records"),
                       key=lambda m: str(m.get("tourney_date", "")))
    details, n_skip, n_debut = [], 0, 0
    for m in test_rows:
        wid, lid = norm_id(m.get("winner_id")), norm_id(m.get("loser_id"))
        wn, ln = m.get("winner_name", "") or wid, m.get("loser_name", "") or lid
        if not wid or not lid:
            n_skip += 1
            continue
        w_is_p1 = rng.random() < 0.5
        p1 = pdb.get(wid) or Player(id=wid, name=str(wn))
        p2 = pdb.get(lid) or Player(id=lid, name=str(ln))
        if wid not in pdb or lid not in pdb:
            n_debut += 1
            db = dict(pdb)
            db.setdefault(wid, p1)
            db.setdefault(lid, p2)
        else:
            db = pdb
        if w_is_p1:
            p1o, p2o, label = p1, p2, 1
        else:
            p1o, p2o, label = p2, p1, 0
        surf = sack.parse_surface(m.get("surface", "Hard"))
        lvl = sack.parse_tournament_level(m.get("tourney_level", ""), m.get("tour", "atp"))
        tm = TennisMatch(
            id=f"bt_{m.get('tourney_date')}_{wid}_{lid}",
            tournament=m.get("tourney_name", "?"), tournament_level=lvl,
            surface=surf, round=m.get("round", "?"),
            player1=p1o, player2=p2o, scheduled_time=parse_dt(m.get("tourney_date", "20200101")),
        )
        key = tuple(sorted([str(p1o.name), str(p2o.name)]))
        h2h = h2h_index.get(key, [])
        # NEWS ESCLUSE: news_impact=None
        feat = fbuilder.build_features(match=tm, player_db=db, merged_odds=None,
                                       news_impact=None, h2h_data=h2h)
        pred = ens.predict(feat)
        p = pred.p1_win_prob
        ok = (p >= 0.5) == (label == 1)
        # Baseline Elo-only (solo se entrambi noti)
        elo_ok = None
        if wid in pdb and lid in pdb:
            e1 = pdb[wid].surface_elo.get(surf, pdb[wid].get_surface_elo(surf))
            e2 = pdb[lid].surface_elo.get(surf, pdb[lid].get_surface_elo(surf))
            if w_is_p1:
                elo_ok = (e1 >= e2)
            else:
                elo_ok = (e2 >= e1)
        tier = feat.data_tier.value if hasattr(feat.data_tier, "value") else feat.data_tier
        details.append({
            "match": f"{p1o.name} vs {p2o.name}",
            "tournament": m.get("tourney_name", "?"), "tour": m.get("tour", "?"),
            "surface": surf.value if isinstance(surf, Surface) else str(surf),
            "tier": tier, "label": label, "p1_prob": round(float(p), 4),
            "correct": bool(ok), "elo_correct": None if elo_ok is None else bool(elo_ok),
            "confidence": round(float(pred.confidence), 3),
        })

    n = len(details)
    wins = sum(1 for d in details if d["correct"])
    brier = sum((d["p1_prob"] - d["label"]) ** 2 for d in details) / n if n else 0
    elo_n = sum(1 for d in details if d["elo_correct"] is not None)
    elo_w = sum(1 for d in details if d["elo_correct"])
    by_tier, by_tour = {}, {}
    for d in details:
        for grp, k in ((by_tier, f"Tier {d['tier']}"), (by_tour, d["tour"])):
            g = grp.setdefault(k, {"n": 0, "w": 0})
            g["n"] += 1
            g["w"] += 1 if d["correct"] else 0
    for grp in (by_tier, by_tour):
        for g in grp.values():
            g["winrate"] = round(g["w"] / g["n"], 4) if g["n"] else 0

    report = {
        "test_start": test_start, "test_end": test_end, "n_test": n,
        "skipped": n_skip, "with_debutant": n_debut,
        "wins": wins, "winrate": round(wins / n, 4) if n else 0,
        "brier": round(brier, 4),
        "elo_baseline": {"n": elo_n, "winrate": round(elo_w / elo_n, 4) if elo_n else 0},
        "by_tier": by_tier, "by_tour": by_tour,
        "news": "escluse (news_impact=None)",
        "lookahead": "nessuno (train < test_start)",
        "details": details,
    }
    return report


def print_report(r):
    print()
    print("=" * 72)
    print(f"  BACKTEST {r['test_start']} - {r['test_end']}")
    print("=" * 72)
    if not r.get("n_test"):
        print(f"  {r.get('message', 'nessun match')}")
        return
    print(f"  Match testati : {r['n_test']}  (skippati: {r['skipped']}, "
          f"con debuttanti: {r['with_debutant']})")
    print(f"  WINRATE       : {r['wins']}/{r['n_test']} = {r['winrate']:.2%}")
    print(f"  Brier score   : {r['brier']:.4f}  (0 = perfetto)")
    e = r["elo_baseline"]
    print(f"  Baseline Elo  : {e['winrate']:.2%} su {e['n']} match "
          f"(delta modello: {r['winrate'] - e['winrate']:+.2%})")
    print("  Per tier:")
    for k in sorted(r["by_tier"]):
        g = r["by_tier"][k]
        print(f"    {k:<7}: {g['w']:>3}/{g['n']:<3} = {g['winrate']:.2%}")
    print("  Per tour:")
    for k in sorted(r["by_tour"]):
        g = r["by_tour"][k]
        print(f"    {k:<5}: {g['w']:>3}/{g['n']:<3} = {g['winrate']:.2%}")
    print(f"  News: {r['news']} | Lookahead: {r['lookahead']}")


def main():
    ap = argparse.ArgumentParser(description="Backtest pronostici su risultati storici")
    ap.add_argument("--start", required=True, help="inizio finestra test YYYYMMDD")
    ap.add_argument("--end", required=True, help="fine finestra test YYYYMMDD")
    ap.add_argument("--years", nargs="+", type=int, default=list(range(2020, 2027)))
    ap.add_argument("--max-train", type=int, default=15000)
    ap.add_argument("--max-baseline", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    setup_logging(log_level="WARNING")
    r = run_backtest(int(a.start), int(a.end), a.years, a.max_train, a.max_baseline, a.seed)
    print_report(r)
    out = Path("output") / f"backtest_{a.start}_{a.end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\n  Report salvato: {out}")


if __name__ == "__main__":
    main()

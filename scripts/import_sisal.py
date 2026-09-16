#!/usr/bin/env python3
"""Import manuale palinsesto Sisal in data/sisal_today.json.

Sisal.it blocca gli scraper automatici (anti-bot Imperva): se lo scraping
fallisce, copia il palinsesto di oggi in un file di testo con una riga
per match nel formato:

  Giocatore A;Giocatore B;1.85;2.10;Nome Torneo

poi lancia:
  venv\\Scripts\\python.exe scripts\\import_sisal.py palinsesto.txt
  venv\\Scripts\\python.exe -m src.main --no-progress
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.ingest.sisal import SisalMatch, save_cache


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = date.today()
    if len(sys.argv) >= 3:
        target = date.fromisoformat(sys.argv[2])
    matches = []
    for i, line in enumerate(Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 4:
            print(f"riga {i + 1} saltata (servono A;B;q1;q2[;torneo]): {line}")
            continue
        try:
            matches.append(SisalMatch(
                id=f"sisal_manual_{abs(hash(parts[0] + parts[1])) % 10**8}",
                tournament=parts[4] if len(parts) > 4 else "Sisal Tennis",
                player1=parts[0], player2=parts[1],
                odds1=float(parts[2].replace(",", ".")),
                odds2=float(parts[3].replace(",", ".")),
            ))
        except Exception as e:
            print(f"riga {i + 1} saltata ({e}): {line}")
    save_cache(matches, target)
    print(f"Salvati {len(matches)} match in data/sisal_today.json per {target}")


if __name__ == "__main__":
    main()

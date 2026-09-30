"""Con tre segnalazioni il recupero le porta tutte nel contesto, e il RAG conta giusto per caso.
Con quaranta non più: le domande «quante» e «quali, in tutto» sbagliano. È la prova di oggi.

uv run python -m scripts.genera_segnalazioni --n 40     # scrive in data/docs/compliance_only/
Poi l'ingestione del Giorno 6, o `python -m evals.ingest_fixtures`, e l'estrazione del grafo.
Scrive anche evals/datasets/grafo_generato.jsonl: le risposte attese le calcola il comando dal
corpus che ha scritto, non le scrive una persona. Stesso seme, stesso corpus, stesse attese.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CARTELLA = Path("data/docs/compliance_only")
DATASET = Path("evals/datasets/grafo_generato.jsonl")
OPERATORI = ["Marco Bianchi", "Piero Galli"]
CONTI = [  # dal seed del Giorno 3: conto, cliente, nome
    ("IT60X0542811101000000123", "C-10234", "Paolo Ferri"),
    ("IT60X0542811101000000456", "C-10234", "Paolo Ferri"),
    ("IT60X0542811101000000789", "C-20417", "Anna Greco"),
]
CONTROPARTI = [
    "Delta Trading Ltd",
    "Inversiones Caribe S.A.",
    "Nordhaven Shipping",
    "Alpina Logistica Srl",
    "Bergamo Tessile SpA",
]
# le tre segnalazioni scritte a mano: entrano nei conteggi come le altre
A_MANO = {"Marco Bianchi": 2, "Piero Galli": 1}


def main(n: int, seme: int) -> None:
    caso = random.Random(seme)
    aperte: Counter[str] = Counter(A_MANO)
    clienti_per_controparte: defaultdict[str, set[str]] = defaultdict(set)
    for i in range(n):
        codice = f"S-2026-{100 + i:03d}"
        operatore = caso.choice(OPERATORI)
        conto, cliente, nome = caso.choice(CONTI)
        controparte = caso.choice(CONTROPARTI)
        importo = f"{caso.randrange(3000, 25000, 100):,}".replace(",", ".")  # 12.300
        (CARTELLA / f"segnalazione_{codice}.md").write_text(
            f"# Segnalazione {codice}\n\n"
            f"La segnalazione {codice} è stata aperta dall'operatore {operatore}.\n"
            f"La segnalazione {codice} riguarda il conto {conto}.\n"
            f"Il conto {conto} è intestato al cliente {cliente}, {nome}.\n"
            f"La segnalazione {codice} riguarda un bonifico verso {controparte}, "
            f"per {importo} euro.\n",
            encoding="utf-8",
        )
        aperte[operatore] += 1
        clienti_per_controparte[controparte].add(nome)

    casi = [
        {
            "id": f"gen-aperte-{i}",
            "ruolo": "compliance_lead",
            "domanda": f"Quante segnalazioni ha aperto {operatore}?",
            "attese": [str(aperte[operatore])],
        }
        for i, operatore in enumerate(OPERATORI, start=1)
    ]
    # Nordhaven compare solo nel corpus generato: la risposta attesa non dipende dalle unioni
    casi.append(
        {
            "id": "gen-clienti-nordhaven",
            "ruolo": "compliance_lead",
            "domanda": "Quali clienti hanno segnalazioni con bonifici verso Nordhaven Shipping?",
            "attese": sorted(clienti_per_controparte["Nordhaven Shipping"]),
        }
    )
    DATASET.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in casi), encoding="utf-8"
    )
    print(f"{n} segnalazioni in {CARTELLA}/, attese in {DATASET}")
    for c in casi:
        print(f"  {c['domanda']}  ->  {', '.join(c['attese'])}")


if __name__ == "__main__":
    argomenti = argparse.ArgumentParser()
    argomenti.add_argument("--n", type=int, default=40)
    argomenti.add_argument("--seme", type=int, default=11)
    a = argomenti.parse_args()
    main(a.n, a.seme)
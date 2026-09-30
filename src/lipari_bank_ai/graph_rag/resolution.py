import re
import unicodedata
from difflib import SequenceMatcher

from lipari_bank_ai.graph_rag.modelli import Entita

# I codici del dominio: quando il testo li scrive, sono l'identità, e nessuna euristica fa meglio
CODICI: dict[str, re.Pattern[str]] = {
    "Segnalazione": re.compile(r"S-\d{4}-\d{3}"),
    "Cliente": re.compile(r"C-\d{5}"),
    "Conto": re.compile(r"IT\d{2}[A-Z]\d{10,30}"),
    "Circolare": re.compile(r"\d{1,3}/\d{4}"),
}
# sopra la soglia due nomi sono SIMILI: li guarda una persona, non si uniscono da soli
SOGLIA_REVISIONE = 0.75


def normalizza(testo: str) -> str:
    """Minuscole, niente accenti, niente punti, niente punteggiatura, spazi singoli.

    «Inversiones Caribe S.A.» e «INVERSIONES CARIBE SA» diventano la stessa stringa: è una
    regola meccanica, ripetibile e spiegabile, ed è per questo che si può applicare da sola.
    """
    senza_accenti = "".join(
        c for c in unicodedata.normalize("NFKD", testo) if not unicodedata.combining(c)
    )
    t = senza_accenti.lower().replace(".", "")
    t = re.sub(r"[^\w\s/-]", " ", t)
    return " ".join(t.split())


def chiave(e: Entita) -> str:
    """La chiave del nodo: il codice se c'è, altrimenti la forma normalizzata del nome."""
    for fonte in (e.identificativo or "", e.menzione):
        if (schema := CODICI.get(e.tipo)) and (m := schema.search(fonte)):
            return f"{e.tipo}:{m.group(0)}"
    return f"{e.tipo}:{normalizza(e.menzione)}"


def somiglianza(a: str, b: str) -> float:
    """Da 0 a 1, sulle forme normalizzate. Misura la somiglianza, non l'identità."""
    return SequenceMatcher(None, normalizza(a), normalizza(b)).ratio()


def candidati(nuova: str, esistenti: dict[str, str], tipo: str) -> list[tuple[str, float]]:
    """Le chiavi dello stesso tipo abbastanza simili da farle guardare a una persona.

    `esistenti` va da chiave a nome. Non unisce niente: produce la coda di revisione.
    """
    if not nuova.startswith(f"{tipo}:") or tipo in CODICI:
        return []  # chi ha un codice non si riconcilia per somiglianza
    nome = nuova.split(":", 1)[1]
    trovati = []
    for k, n in esistenti.items():
        if k != nuova and k.startswith(f"{tipo}:"):
            s = somiglianza(nome, n)
            if s >= SOGLIA_REVISIONE:
                trovati.append((k, round(s, 2)))
    return sorted(trovati, key=lambda x: -x[1])
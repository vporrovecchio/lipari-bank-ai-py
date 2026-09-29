from functools import lru_cache
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def load_prompt(nome: str) -> str:
    """Il testo di un prompt da `prompts/`, per nome senza estensione.

    Il nome porta la versione (`chat_system_v1`): il file è l'artefatto che si
    versiona, e il codice che lo usa cambia solo quando cambia l'artefatto.
    """
    percorso = PROMPTS / f"{nome}.md"
    if not percorso.is_file():
        disponibili = ", ".join(sorted(p.stem for p in PROMPTS.glob("*.md")))
        raise FileNotFoundError(
            f"Prompt {nome!r} non trovato in {PROMPTS}. Disponibili: {disponibili}"
        )
    return percorso.read_text(encoding="utf-8")

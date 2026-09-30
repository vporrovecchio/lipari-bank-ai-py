from typing import Literal

from pydantic import BaseModel, Field

# Elenchi chiusi: con una stringa libera il modello inventa tre nomi per la stessa relazione,
# e nessuna query li trova tutti. Aggiungere un tipo è una decisione, e passa da qui.
TipoEntita = Literal["Segnalazione", "Operatore", "Conto", "Cliente", "Controparte", "Circolare"]
TipoRelazione = Literal["APERTA_DA", "RIGUARDA", "INTESTATO_A", "VERSO", "ELENCA"]


class Entita(BaseModel):
    tipo: TipoEntita
    menzione: str = Field(
        description="Il nome ESATTAMENTE come compare nel testo, senza normalizzarlo."
    )
    identificativo: str | None = Field(
        default=None,
        description="Il codice, se il testo lo scrive: S-2026-014, C-10234, un IBAN. Se no, null.",
    )


class Relazione(BaseModel):
    da: str = Field(description="identificativo, o menzione, dell'entità di partenza")
    tipo: TipoRelazione
    a: str = Field(description="identificativo, o menzione, dell'entità di arrivo")
    citazione: str = Field(
        description="La frase del testo che afferma la relazione, copiata com'è. "
        "Se non c'è una frase che la afferma, la relazione non c'è."
    )


class GrafoEstratto(BaseModel):
    entita: list[Entita]
    relazioni: list[Relazione]


class EntitaNominate(BaseModel):
    """Le entità che una domanda nomina: il punto da cui parte la ricerca sul grafo."""

    entita: list[Entita]
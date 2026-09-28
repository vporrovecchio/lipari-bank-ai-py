import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from difflib import get_close_matches
from typing import cast
from uuid import uuid4

from openai import AsyncOpenAI, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import ValidationError

from lipari_bank_ai.agents.registry import Tool

logger = logging.getLogger(__name__)

MAX_STEPS = 6                         # il tetto dell'endpoint: la commessa ti chiede di misurarlo
BUDGET_PER_RUN = Decimal("0.05")      # euro
PREZZI_PER_1K = {                     # euro per mille token (ingresso, uscita), come al Giorno 4
    "gpt-4o-mini": (Decimal("0.00014"), Decimal("0.00056")),
    "gpt-4o": (Decimal("0.0023"), Decimal("0.0091")),
}


@dataclass
class RunResult:
    run_id: str
    reply: str = ""
    steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    stopped_by: str = ""              # "model" | "max_steps" | "budget"
    cost_eur: Decimal = Decimal("0")


async def run_agent(
    messaggi: list[ChatCompletionMessageParam], tools: list[Tool], client: AsyncOpenAI, model: str,
    max_steps: int = MAX_STEPS,
) -> RunResult:
    """Chiama il modello finché chiede tool, poi ritorna la sua risposta.

    Riceve la conversazione e non la domanda: così lo stesso ciclo serve una domanda
    nuova, una conversazione salvata da riprendere e uno specialista con metà dei tool.
    Esce per una delle tre ragioni in `stopped_by`, e chi legge il risultato
    deve poterle distinguere senza guardare il testo della risposta.
    """
    if model not in PREZZI_PER_1K:    # meglio fermarsi qui che scoprirlo dopo aver pagato
        raise ValueError(f"Nessun prezzo per {model}: il budget non sarebbe calcolabile.")
    messaggi = list(messaggi)                     # la lista di chi chiama non si tocca
    by_name = {t.name: t for t in tools}          # il dispatch vede SOLO i tool offerti
    schemi = [t.to_openai_schema() for t in tools]
    run = RunResult(run_id=str(uuid4()))          # la chiave di ogni riga di traccia

    for _ in range(max_steps):                    # il tetto è nella struttura
        run.steps += 1
        risposta = await client.chat.completions.create(
            model=model, messages=messaggi, max_tokens=500,
            tools=schemi or omit,                 # l'API rifiuta una lista vuota
        )
        run.cost_eur += _costo(risposta.usage, model)
        choice = risposta.choices[0].message

        if not choice.tool_calls:                 # uscita 1: ha risposto
            run.stopped_by = "model"
            run.reply = choice.content or ""
            return run
        if run.cost_eur > BUDGET_PER_RUN:         # uscita 3: il budget, prima di altri tool
            run.stopped_by = "budget"
            run.reply = ("Ho interrotto l'elaborazione perché la richiesta ha superato il "
                         "budget previsto. Prova a formularla in modo più circoscritto.")
            return run

        # PRIMA l'assistant con le chiamate, poi i risultati: altrimenti l'API rifiuta
        assistente = choice.model_dump(exclude_none=True)
        messaggi.append(cast(ChatCompletionAssistantMessageParam, assistente))
        for call in choice.tool_calls:
            if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                continue                          # offriamo solo function tool
            run.tool_calls.append(call.function.name)
            tool = by_name.get(call.function.name)
            inizio = time.perf_counter()
            esito = await _execute(by_name, call)
            logger.info("agent_step", extra={
                "run_id": run.run_id, "step": run.steps, "tool": call.function.name,
                # gli argomenti di un tool che scrive non vanno nel log: stanno nella sua tabella
                "tool_args": "<omessi>" if tool and tool.scrive else call.function.arguments[:200],
                "result_preview": esito[:200], "result_len": len(esito),
                "duration_ms": int((time.perf_counter() - inizio) * 1000),
            })
            messaggi.append({"role": "tool", "tool_call_id": call.id, "content": esito})

    run.stopped_by = "max_steps"                  # uscita 2: tetto raggiunto, NON è una risposta
    run.reply = ("Non ho completato la richiesta entro i passi previsti: la risposta non c'è. "
                 "Riformula la domanda in modo più specifico, o dividila in domande più semplici.")
    return run


def _costo(usage: CompletionUsage | None, model: str) -> Decimal:
    if usage is None:
        return Decimal("0")
    ingresso, uscita = PREZZI_PER_1K[model]
    return (usage.prompt_tokens * ingresso + usage.completion_tokens * uscita) / 1000


async def _execute(by_name: dict[str, Tool], call: ChatCompletionMessageFunctionToolCall) -> str:
    tool = by_name.get(call.function.name)
    if tool is None:
        # Il modello ha inventato un nome. Glielo diciamo, non solleviamo — e lo
        # registriamo: un nome inventato con insistenza è un segnale da guardare
        logger.warning("tool_inesistente", extra={"tool": call.function.name[:80]})
        vicini = get_close_matches(call.function.name, list(by_name), n=1)
        suggerimento = f"Forse intendevi '{vicini[0]}'. " if vicini else ""
        elenco = f"Tool disponibili: {', '.join(by_name)}." if len(by_name) <= 10 else ""
        return f"ERRORE: il tool '{call.function.name}' non esiste. {suggerimento}{elenco}".strip()
    try:
        args = tool.args_model.model_validate_json(call.function.arguments)
    except ValidationError as e:
        return f"ERRORE di validazione degli argomenti: {e.errors(include_url=False)}"
    try:
        return await tool.run(args)
    except Exception:
        # un tool che fallisce (database lento, vincolo violato) è un'osservazione, non un 500
        logger.exception("tool_fallito", extra={"tool": tool.name})
        return (f"ERRORE: il tool '{tool.name}' non ha potuto completare l'operazione. "
                "Non riprovare la stessa chiamata: riferisci all'utente che il dato ora non è "
                "disponibile.")
import asyncio

from openai.types.chat import ChatCompletionMessageFunctionToolCall

from lipari_bank_ai.agents.loop import _execute
from lipari_bank_ai.agents.registry import Tool


async def execute_all(
    by_name: dict[str, Tool], calls: list[ChatCompletionMessageFunctionToolCall]
) -> dict[str, str]:
    """Esegue insieme i tool in lettura, poi uno alla volta quelli che scrivono.

    Restituisce l'esito per id della chiamata: chi lo usa rimanda i messaggi `tool`
    nell'ordine delle chiamate, non in quello di completamento.
    """

    def in_lettura(call: ChatCompletionMessageFunctionToolCall) -> bool:
        tool = by_name.get(call.function.name)    # un nome inventato non esegue niente
        return tool is None or not tool.scrive

    insieme = [c for c in calls if in_lettura(c)]
    esiti = dict(zip(
        [c.id for c in insieme],
        await asyncio.gather(*(_execute(by_name, c) for c in insieme)),
        strict=True,
    ))
    for call in calls:                            # chi scrive: dopo, e in fila
        if call.id not in esiti:
            esiti[call.id] = await _execute(by_name, call)
    return esiti
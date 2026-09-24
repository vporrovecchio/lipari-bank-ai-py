TOOLS_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "get_account_balance",
        "description": (
            "Restituisce il saldo disponibile di un conto dell'utente. "
            "Usalo quando la domanda riguarda quanto c'è su un conto o la "
            "capienza per un'operazione. Non usarlo per l'elenco dei movimenti."
        ),
        "parameters": {
            "type": "object",
            "properties": {"account_id": {"type": "string", "description": "IBAN di un conto dell'utente."}},
            "required": ["account_id"],
        },
    },
}]
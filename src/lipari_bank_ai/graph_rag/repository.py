MAX_HOP = 2            # il tetto vive qui, non nei parametri di chi chiama


class GrafoRepository:
    async def upsert(self, estratto: GrafoEstratto, documento_ref: str) -> None:
        """Scrive entità e relazioni. Rieseguire l'ingestione non deve duplicare niente."""

    async def espandi(self, chiavi: list[str], hop: int = MAX_HOP) -> list[dict]:
        """I vicini delle entità indicate, entro un numero di passi VINCOLATO."""
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from lipari_bank_ai.auth.acl import visible_to
from lipari_bank_ai.db.models import GraphEdge, GraphNode, GraphReview
from lipari_bank_ai.graph_rag.modelli import Entita, Relazione
from lipari_bank_ai.graph_rag.resolution import candidati, chiave

# Il tetto vive qui, non nei parametri di chi chiama. Tre salti bastano alla domanda più lunga
# del corpus — dalla controparte al cliente, passando per la segnalazione e il conto — e il
# numero si ricava dalle tue domande, non si copia
MAX_HOP = 3
MAX_ARCHI = 200  # il secondo presidio: anche entro tre salti, un nodo molto collegato è tanto


@dataclass(frozen=True)
class Arco:
    da: str
    tipo: str
    a: str
    document_id: str
    citazione: str
    distanza: int


class GrafoRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def alias(self) -> dict[str, str]:
        """Le unioni confermate da una persona: da chiave a chiave canonica."""
        righe = await self.session.execute(
            select(GraphReview.chiave, GraphReview.candidata).where(GraphReview.stato == "unite")
        )
        return {k: c for k, c in righe.all()}

    async def upsert(
        self,
        entita: list[Entita],
        relazioni: list[Relazione],
        document_id: str,
        visibility: str,
    ) -> tuple[int, int]:
        """Scrive nodi e archi di un passaggio. Due volte lo stesso passaggio: gli stessi archi.

        Restituisce quanti archi ha scritto e quanti ne ha lasciati perché un estremo non era
        fra le entità estratte. Non conferma: la transazione è di chi chiama.
        """
        alias = await self.alias()

        def canonica(k: str) -> str:
            return alias.get(k, k)

        per_nome: dict[str, str] = {}  # da identificativo o menzione a chiave del nodo
        for e in entita:
            k = canonica(chiave(e))
            await self.session.execute(
                # il MERGE del grafo: se il nodo c'è, resta com'è
                insert(GraphNode)
                .values(chiave=k, tipo=e.tipo, nome=e.menzione)
                .on_conflict_do_nothing(index_elements=["chiave"])
            )
            for nome in (e.identificativo, e.menzione):
                if nome:
                    per_nome[nome] = k
        await self._accoda_simili(entita, document_id)

        scritti, orfani = 0, 0
        for r in relazioni:
            da, a = per_nome.get(r.da), per_nome.get(r.a)
            if da is None or a is None:
                orfani += 1  # un estremo che il modello non ha elencato fra le entità
                continue
            esito = await self.session.execute(
                insert(GraphEdge)
                .values(
                    da=da,
                    tipo=r.tipo,
                    a=a,
                    document_id=document_id,
                    citazione=r.citazione[:500],
                    visibility=visibility,
                )
                .on_conflict_do_nothing(constraint="uq_arco")
            )
            scritti += getattr(esito, "rowcount", 0)  # 0 se c'era già: si contano i nuovi
        return scritti, orfani

    async def dimentica(self, document_id: str) -> None:
        """Toglie gli archi di un documento, prima di rileggerlo: un testo cambiato non deve
        lasciare nel grafo i fatti che non dice più. I nodi restano, gli archi di altri pure."""
        await self.session.execute(
            text("DELETE FROM graph_edges WHERE document_id = :d"), {"d": document_id}
        )

    async def _accoda_simili(self, entita: list[Entita], document_id: str) -> None:
        """Il terzo livello: i nomi simili a uno già visto vanno in coda, non si uniscono."""
        esistenti = dict(
            (await self.session.execute(select(GraphNode.chiave, GraphNode.nome))).all()
        )
        # una coppia già in coda, o già decisa, non torna: nemmeno letta al contrario
        viste = {
            frozenset(c)
            for c in (
                await self.session.execute(select(GraphReview.chiave, GraphReview.candidata))
            ).all()
        }
        for e in entita:
            k = chiave(e)
            for candidata, s in candidati(k, esistenti, e.tipo):
                if frozenset((k, candidata)) in viste:
                    continue
                await self.session.execute(
                    insert(GraphReview)
                    .values(chiave=k, candidata=candidata, somiglianza=s, document_id=document_id)
                    .on_conflict_do_nothing(constraint="uq_revisione")
                )

    SQL_ESPANDI = text(
        """
        WITH RECURSIVE passo(nodo, arco, distanza, visti) AS (
            SELECT n.chiave, NULL::varchar, 0, ARRAY[n.chiave]::varchar[]
            FROM graph_nodes n WHERE n.chiave = ANY(:chiavi)
          UNION ALL
            SELECT CASE WHEN e.da = p.nodo THEN e.a ELSE e.da END, e.id, p.distanza + 1,
                   p.visti || (CASE WHEN e.da = p.nodo THEN e.a ELSE e.da END)::varchar
            FROM passo p
            JOIN graph_edges e ON p.nodo IN (e.da, e.a)
            WHERE p.distanza < :hop
              AND e.visibility = ANY(:livelli)
              AND NOT (CASE WHEN e.da = p.nodo THEN e.a ELSE e.da END) = ANY(p.visti)
        )
        SELECT e.da, e.tipo, e.a, e.document_id, e.citazione, min(p.distanza) AS distanza
        FROM passo p JOIN graph_edges e ON e.id = p.arco
        GROUP BY e.id, e.da, e.tipo, e.a, e.document_id, e.citazione
        ORDER BY distanza, e.da, e.tipo, e.a
        LIMIT :limite
        """
    )

    async def espandi(self, chiavi: list[str], role: str, hop: int = MAX_HOP) -> list[Arco]:
        """Gli archi entro `hop` salti dalle chiavi, fra quelli che questo ruolo può vedere.

        Il tetto si abbassa, non si alza: un valore più alto di MAX_HOP diventa MAX_HOP. Il
        filtro sui livelli sta DENTRO la ricorsione: un arco che il ruolo non vede non si
        attraversa nemmeno, quindi non porta a nodi che da lì non si raggiungerebbero.
        """
        if not chiavi:
            return []
        alias = await self.alias()
        righe = await self.session.execute(
            self.SQL_ESPANDI,
            {
                "chiavi": [alias.get(k, k) for k in chiavi],
                "hop": max(0, min(hop, MAX_HOP)),
                "livelli": visible_to(role),
                "limite": MAX_ARCHI,
            },
        )
        return [Arco(*r) for r in righe.all()]

    async def unisci(self, chiave_vecchia: str, canonica: str) -> None:
        """Porta gli archi di un nodo su quello canonico: dopo, i cammini passano da uno solo.

        Il nodo vecchio resta, senza archi; la decisione resta scritta nella coda. Due archi
        che diventano uguali — lo stesso fatto dallo stesso documento — restano uno.
        """
        await self.session.execute(
            text(
                """
                INSERT INTO graph_edges (id, da, tipo, a, document_id, citazione, visibility)
                SELECT gen_random_uuid()::text,
                       CASE WHEN da = :k THEN :c ELSE da END, tipo,
                       CASE WHEN a = :k THEN :c ELSE a END,
                       document_id, citazione, visibility
                FROM graph_edges WHERE :k IN (da, a)
                ON CONFLICT ON CONSTRAINT uq_arco DO NOTHING
                """
            ),
            {"k": chiave_vecchia, "c": canonica},
        )
        await self.session.execute(
            text("DELETE FROM graph_edges WHERE :k IN (da, a)"), {"k": chiave_vecchia}
        )

    async def esistenti(self, chiavi: list[str]) -> list[str]:
        """Quali di queste chiavi sono nodi del grafo, dopo le unioni confermate."""
        alias = await self.alias()
        canoniche = [alias.get(k, k) for k in chiavi]
        righe = await self.session.scalars(
            select(GraphNode.chiave).where(GraphNode.chiave.in_(canoniche))
        )
        return list(righe)
import asyncio
from datetime import date, timedelta
from decimal import Decimal

from sqlalchemy import select

from lipari_bank_ai.db.models import Account, Customer, Movement
from lipari_bank_ai.db.session import AsyncSessionLocal

CLIENTI = [  # (codice cliente, nome, operatore che lo ha in portafoglio)
    ("C-10234", "Paolo Ferri", "mbianchi"),
    ("C-20417", "Anna Greco", "pgalli"),          # un collega di un'altra filiale
]
CONTI = [  # (iban, codice cliente, etichetta, saldo)
    ("IT60X0542811101000000123", "C-10234", "principale", Decimal("48200.00")),
    ("IT60X0542811101000000456", "C-10234", "risparmio", Decimal("1200.00")),
    ("IT60X0542811101000000789", "C-20417", "principale", Decimal("15300.00")),
]
MOVIMENTI = [  # (iban, giorni fa, descrizione, importo)
    ("IT60X0542811101000000123", 3, "Bonifico estero verso Panama", Decimal("-9800.00")),
    ("IT60X0542811101000000123", 11, "Bonifico estero verso Panama", Decimal("-9500.00")),
    ("IT60X0542811101000000123", 15, "Accredito stipendio", Decimal("2850.00")),
    ("IT60X0542811101000000456", 20, "Giroconto da principale", Decimal("500.00")),
]


async def main() -> None:
    async with AsyncSessionLocal() as session:
        if await session.scalar(select(Customer.id).limit(1)) is not None:
            print("Clienti già presenti: nessuna azione.")
            return
        session.add_all(Customer(id=c, full_name=n, operator=o) for c, n, o in CLIENTI)
        session.add_all(Account(id=i, customer_id=c, label=e, balance=s) for i, c, e, s in CONTI)
        oggi = date.today()
        session.add_all(Movement(account_id=i, booking_date=oggi - timedelta(days=g),
                                 description=d, amount=a) for i, g, d, a in MOVIMENTI)
        await session.commit()
    print(f"{len(CLIENTI)} clienti, {len(CONTI)} conti e {len(MOVIMENTI)} movimenti creati.")


if __name__ == "__main__":
    asyncio.run(main())
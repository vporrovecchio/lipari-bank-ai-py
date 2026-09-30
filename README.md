# LipariBank AI

Assistente AI per il servizio clienti di una banca, costruito con **FastAPI**, **PostgreSQL + pgvector** e **uv**.

Questa guida è pensata per chi non ha mai avviato il progetto: segui i passaggi in ordine, dal primo all'ultimo.

---

## 0. Cosa ti serve prima di cominciare

| Strumento | Perché | Dove scaricarlo |
|---|---|---|
| **Python 3.12 o superiore** | il linguaggio del progetto | <https://www.python.org/downloads/> |
| **uv** | gestore di dipendenze e ambienti virtuali | <https://docs.astral.sh/uv/getting-started/installation/> |
| **Docker Desktop** | fa girare database, Ollama e opencode | <https://www.docker.com/products/docker-desktop/> |
| Una **API key OpenAI** | il modello che risponde alle domande | <https://platform.openai.com/api-keys> |

> **Windows:** puoi usare **PowerShell** per tutti i comandi. Quando trovi `cp`, usa `Copy-Item`.
> **macOS / Linux:** i comandi funzionano così come sono.

---

## 1. Verifica di avere tutto

Apri il terminale e controlla che i tre programmi rispondano:

```bash
python --version    # deve dire 3.12 o superiore
uv --version
docker --version
```

Se `uv` non è riconosciuto, installalo con:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
curl -LsSf https://astral.sh/uv/install.sh | sh                                       # macOS / Linux
```

Poi **chiudi e riapri il terminale**, così il comando diventa disponibile.

---

## 2. Posizionati nella cartella del progetto

Tutti i comandi di questa guida vanno eseguiti **dentro la cartella del progetto**, cioè quella che contiene il file `pyproject.toml`.

```bash
cd percorso/del/progetto/lipari-bank-ai-py
```

Se vuoi che `cd` funzioni anche con il percorso completo, scrivilo tra apici:

```bash
cd "C:\Users\tuo-utente\Documenti\lipari-bank-ai-py"
```

---

## 3. Installa le dipendenze

```bash
uv sync
```

Cosa succede: `uv` crea una cartella `.venv` (l'ambiente virtuale del progetto) e ci installa dentro tutto quello che serve. **Non usare `pip`**: il progetto è costruito con `uv`.

---

## 4. Crea il file di configurazione `.env`

Il file `.env` contiene le chiavi e gli indirizzi dei servizi. Deve trovarsi **nella cartella del progetto**, insieme a `pyproject.toml`.

Copia il file di esempio:

```bash
Copy-Item .env.example .env      # Windows
cp .env.example .env             # macOS / Linux
```

Ora aprilo con un editor di testo (`notepad .env`) e riempi i valori vuoti:

```ini
APP_NAME=LipariBank AI
DEBUG=true

DATABASE_URL=postgresql+asyncpg://lipari:lipari@localhost:5432/lipari_ai

OPENAI_API_KEY=sk-................
ANTHROPIC_API_KEY=
OPENCODE_API_KEY=

DEFAULT_MODEL=gpt-4o-mini

JWT_SECRET=cambia-questa-stringa-con-quello-che-vuoi

OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
MAX_TOKENS_PER_REQUEST=2000
```

Cosa serve sapere:

- `DATABASE_URL` — le credenziali sono già quelle di Docker Compose: utente `lipari`, password `lipari`, database `lipari_ai`. **Non cambiarle**, altrimenti il database non si collega.
- `OPENAI_API_KEY` — **obbligatoria**, prendila da platform.openai.com.
- `ANTHROPIC_API_KEY` e `OPENCODE_API_KEY` — puoi lasciarle vuote se usi solo OpenAI. Inseriscile solo se il modello che scegli è Claude o opencode.
- `DEFAULT_MODEL` — il modello che risponde. Con la sola API key OpenAI usa `gpt-4o-mini`. Con il server opencode locale usa `opencode/big-pickle`. I valori ammessi sono quelli che iniziano con `gpt`, `claude` o `opencode`.
- `JWT_SECRET` — una stringa a piacere, serve a firmare i token di accesso. Per imparare basta quella sopra.

> Salva il file e accertati che il nome sia esattamente `.env` (su Windows, quando chiede l'estensione, scegli "Tutti i file" per non ritrovarti con `.env.txt`).

---

## 5. Avvia i servizi di supporto

Il database e gli altri servizi girano dentro Docker. Con Docker Desktop acceso, dalla cartella del progetto:

```bash
docker compose up -d
```

Il comando scarica le immagini (la prima volta qualche minuto) e le avvia in background.

Per controllare che siano partiti:

```bash
docker compose ps
```

Devi vedere PostgreSQL, Ollama e opencode con stato `running`.

Scarica anche il modello per gli embedding, che serve al primo avvio:

```bash
docker compose exec ollama ollama pull nomic-embed-text
```

---

## 6. Crea le tabelle del database

```bash
uv run alembic upgrade head
```

Questo comando applica tutte le migrazioni e crea le tabelle. Se lo esegui una seconda volta non fa nulla di dannoso.

Per controllare che sia andato a buon fine:

```bash
uv run alembic current
```

Deve mostrare una revisione che termina con `(head)`.

---

## 7. Inserisci i dati di prova

```bash
uv run python -m lipari_bank_ai.scripts.seed_all
```

Il comando crea gli utenti, i clienti con i loro conti e indicizza i documenti interni. **È sicuro ripeterlo**: salta quello che c'è già. Se vuoi ricostruire l'indice dei documenti da capo, aggiungi `SEED_REINDEX=1` davanti (su PowerShell: `$env:SEED_REINDEX="1"; uv run python -m lipari_bank_ai.scripts.seed_all`).

Utenti creati, tutti con password `bootcamp`:

| Username | Ruolo |
|---|---|
| `mbianchi` | operatore |
| `grossi` | compliance |
| `lverdi` | responsabile di reparto |

---

## 8. Avvia l'applicazione

```bash
uv run uvicorn lipari_bank_ai.main:app --reload --port 8000
```

L'opzione `--reload` riavvia da sola il server a ogni modifica del codice: puoi lasciare il terminale aperto e continuare a lavorare.

Se tutto è andato bene vedrai una riga con `Uvicorn running on http://0.0.0.0:8000`.

Lascia **questo terminale aperto**: l'applicazione si ferma quando lo chiudi.

---

## 9. Verifica che funzioni

Apri il browser e vai su:

**http://localhost:8000/docs**

Vedrai la pagina interattiva con tutti gli endpoint: puoi provarli direttamente da lì senza scrivere codice.

Per un controllo rapido, in un altro terminale:

```bash
curl http://localhost:8000/health
```

Risposta attesa:

```json
{"status":"UP"}
```

E poi, per verificare anche il collegamento al database:

```bash
curl http://localhost:8000/ready
```

Risposta attesa:

```json
{"status":"ready","checks":{"database":"ok","cache":"spenta"}}
```

Se `database` fosse `ko`, il server è partito ma Postgres non è raggiungibile: ricontrolla che Docker sia acceso e che i passaggi 5 e 6 siano stati eseguiti.

---

## 10. Primo accesso

Dal browser, apri **http://localhost:8000/docs**, espandi `POST /api/auth/login`, clicca **Try it out** e accedi con `mbianchi` / `bootcamp`. Copia il token `access_token` che ti restituisce: ti servirà per chiamare gli altri endpoint, incollandolo nel campo **Authorize** in alto a destra.

---

## Riepilogo: i comandi, in ordine

```bash
uv sync
Copy-Item .env.example .env          # poi aprilo e riempi le chiavi
docker compose up -d
docker compose exec ollama ollama pull nomic-embed-text
uv run alembic upgrade head
uv run python -m lipari_bank_ai.scripts.seed_all
uv run uvicorn lipari_bank_ai.main:app --reload --port 8000
```

---

## Fermare tutto

- L'applicazione: `Ctrl+C` nel terminale dove gira il server.
- I servizi Docker: `docker compose down` (aggiungi `-v` se vuoi cancellare anche i dati del database e ricominciare da zero).

---

## Comandi utili dopo

| Cosa vuoi fare | Comando |
|---|---|
| Riavviare il server dopo aver cambiato qualcosa | salva il file: con `--reload` riparte da solo |
| Vedere i log del server in tempo reale | il terminale dove hai avviato `uvicorn` |
| Vedere i log dei container | `docker compose logs -f` |
| Controllare che il database risponda | `docker compose ps` |
| Eseguire i test | `uv run pytest -q` |
| Controllare la qualità del codice | `uv run ruff check .` e `uv run mypy src` |
| Vedere l'elenco delle tabelle | `docker compose exec postgres psql -U lipari -d lipari_ai -c "\dt"` |

---

## Problemi frequenti

**`uv: command not found`** — uv non è nel PATH. Ripeti il passo 1 e riapri il terminale.

**`ModuleNotFoundError: No module named 'lipari_bank_ai'`** — non sei nella cartella giusta, oppure hai dimenticato `uv sync`. Torna al passo 2 e ripeti il passo 3.

**`ValidationError` complaining about a missing key** — manca una variabile obbligatoria in `.env`. Controlla che `DATABASE_URL`, `OPENAI_API_KEY` e `JWT_SECRET` abbiano un valore.

**`Address already in use` / porta 8000 occupata** — un altro programma sta usando la porta 8000. Ripeti il comando con una porta libera: `uv run uvicorn lipari_bank_ai.main:app --reload --port 8001`, e apri <http://localhost:8001/docs>.

**`Connection refused` verso il database** — Docker Desktop non è acceso, oppure i container non sono partiti. Ripeti i passaggi 1 e 5, poi il 6.

**`unknown model` all'avvio** — `DEFAULT_MODEL` nel `.env` non corrisponde a nessun provider riconosciuto. Usa `gpt-4o-mini`, un modello che inizia con `claude`, oppure `opencode/big-pickle`.

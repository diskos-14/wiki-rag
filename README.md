# wiki-rag

AI question answering over English Wikipedia passages: a **FastAPI** backend with **RAG** (retrieval-augmented generation) on **PostgreSQL + pgvector**, and a multi-step **agent** built with **LangGraph** and **Pydantic AI**, using **Gemini on Vertex AI**.

- **`POST /ask`**: basic RAG. Hybrid search (vector + keyword) → evidence threshold → answer with citations.
- **`POST /agent/ask`**: an agent for harder questions (comparisons, facts spread over several articles): understand the question → research with tools → fact-check the answer against its sources.
- Both keep **chat history**, so follow-ups like "When did he die?" work.

## How it works

```
Offline (once)    dataset → clean → rebuild articles (titles) → chunk → embed → Postgres
                  → enrich: topics, summaries, entities (the semantic layer)

/ask              question → (rewrite follow-up) → hybrid search → threshold → answer + citations

/agent/ask        question → understand (intent, sub-questions) → research (tool loop:
                  search_passages / find_entity / get_article_profile) → grounding check
                  (one retry with feedback) → answer, partial answer, or refusal
```

- **Data:** [`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia): 3,200 passages from 41 articles. The passages have no titles and many never name their subject ("It is bordered by Brazil…"), so ingestion rebuilds the articles and adds each article's title to its chunks.
- **Design patterns:** Repository (all SQL), Factory (models and embeddings from config), Strategy (chunkers, retrievers), dependency injection, and a LangGraph state machine for the agent.
- **More detail:** [PLAN.md](PLAN.md) (design decisions and measurements).

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with billing and the **Vertex AI API** enabled
- [uv](https://docs.astral.sh/uv/): only for option B and the tests

Sign in so the app can call Vertex AI with your account (no API key is used anywhere):

```bash
gcloud services enable aiplatform.googleapis.com --project YOUR_PROJECT_ID
```

```bash
gcloud auth application-default login
```

```bash
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## Configure

Copy the template:

```bash
cp .env.example .env
```

Then edit `.env`:

- `GCP_PROJECT`: your project ID.
- `POSTGRES_PASSWORD`: any password, and the **same** password inside `DATABASE_URL`.
- `GCP_ADC_FILE` (option A only): the full path to the credentials file the login above created.
  - Windows: `C:/Users/<you>/AppData/Roaming/gcloud/application_default_credentials.json`
  - macOS / Linux: `/home/<you>/.config/gcloud/application_default_credentials.json`

## Option A: run everything in Docker (recommended)

**1. Start Postgres and the API** (the API image is built on the first run):

```bash
docker compose up -d --build --wait
```

**2. Create the tables:**

```bash
docker compose exec backend alembic upgrade head
```

**3. Load the data** (about 1 minute; downloads the dataset and embeds 2,495 chunks):

```bash
docker compose exec backend python -m scripts.ingest
```

**4. Add the semantic layer** (about 20 seconds; topics, summaries and entities for the agent):

```bash
docker compose exec backend python -m scripts.enrich
```

**5. Open the API docs** at http://localhost:8000/docs and try the endpoints below.

## Option B: run the API with uv (for development)

**1. Start only Postgres:**

```bash
docker compose up -d --wait db
```

**2. Install, migrate and load the data** (from `backend/`):

```bash
cd backend
```

```bash
uv sync
```

```bash
uv run alembic upgrade head
```

```bash
uv run python -m scripts.ingest
```

```bash
uv run python -m scripts.enrich
```

**3. Run the API with auto-reload:**

```bash
uv run uvicorn app.main:app --reload --port 8000
```

If the Docker `backend` container from option A is running, it already uses port 8000: stop it with `docker compose stop backend`, or run uvicorn on another port.

## Try it

Basic RAG:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What borders Uruguay?"}'
```

The agent, with its step-by-step trace:

```bash
curl -X POST localhost:8000/agent/ask -H "Content-Type: application/json" -d '{"question": "Who was born first, Isaac Newton or Blaise Pascal?", "debug": true}'
```

**Follow-ups:** send the `conversation_id` from the previous response with the next question, e.g. `{"question": "When did he die?", "conversation_id": "<id>"}`.

**Other endpoints:**
- `GET /health`: checks the database.
- `GET /conversations/{id}/messages`: the history of a conversation.

Good questions to try:

| Question | What it shows |
|---|---|
| What borders Uruguay? | The passage never says "Uruguay"; the article-title context finds it |
| Who was born first, Isaac Newton or Blaise Pascal? | `/ask` refuses (one search finds only Pascal); `/agent/ask` searches each article and answers |
| Did Nikola Tesla work with Thomas Edison? | Edison has no article; `find_entity` finds him in the Tesla article |
| Which country is larger, Uruguay or Finland? | An honest **partial** answer: the source text is missing Finland's area |
| What is the best JavaScript framework? | Refused as out of scope, without searching |

Answers take seconds on `/ask`. On `/agent/ask` they take about 10–30 seconds (several model calls), and longer when Vertex AI is slow.

## About the data scripts

- **Re-running is safe:** `ingest` replaces the whole dataset in one transaction, and `enrich` rebuilds the semantic layer. Run `enrich` again after every `ingest`.
- **No titling or profiling needed:** article titles and profiles are LLM-generated, but they're committed in `backend/seed/`, so these steps only embed and match text. Generating them from scratch took about an hour, because of Vertex AI rate limits.
- `ingest` options: `--limit N` loads the first N passages only, and leaves only that sample in the database. `--chunker fixed` uses fixed-size chunks instead of one chunk per passage.

## Tests

From `backend/`:

```bash
uv run pytest --cov=app
```

```bash
uv run ruff check .
```

- **About 120 tests, about 97% coverage.** No Google Cloud access is needed: the LLM is replaced by Pydantic AI's `TestModel`/`FunctionModel`, and embeddings by fakes.
- Tests in `tests/integration/` use the docker-compose Postgres. They're skipped when it isn't running, and each one runs in a transaction that is rolled back, so loaded data is never changed.

## Configuration

Settings come from `.env` (see [.env.example](.env.example)); environment variables override it.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | (required) | `postgresql+asyncpg://user:password@localhost:5432/db` |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `POSTGRES_PORT` | | Used by docker-compose to create the database |
| `GCP_PROJECT` | | Google Cloud project for Vertex AI |
| `GCP_LOCATION` | `global` | Vertex AI location |
| `GCP_ADC_FILE` | | Credentials file mounted into the Docker `backend` container |
| `EMBEDDING_MODEL` | `gemini-embedding-2` | Embedding model (768 dimensions) |
| `LLM_MODEL` | `gemini-3.8-flash` | Main model; always used for the agent's grounding check |
| `LLM_FAST_MODEL` | (unset) | Faster model for everything except grounding, e.g. `gemini-3.5-flash-lite` |
| `LLM_PROVIDER` | `google` | `google`, or `mock` (Pydantic AI `TestModel`, no network) |
| `LLM_TIMEOUT_SECONDS` | `30` | Per model call; the SDK retries timeouts and 429/5xx |
| `RETRIEVAL_STRATEGY` | `hybrid` | `hybrid` (vector + keyword) or `vector` |
| `MIN_SIMILARITY` | `0.63` | `/ask` refuses below this best cosine similarity, without calling the LLM |
| `HISTORY_MESSAGES` | `6` | Recent messages used to understand follow-ups |
| `BACKEND_PORT` | `8000` | Host port of the Docker `backend` service |

## Repo layout

```
wiki-rag/
├── backend/
│   ├── app/                # organised by feature
│   │   ├── core/           #   settings, database, Vertex AI client, errors, health
│   │   ├── knowledge/      #   documents + chunks: models, repository
│   │   ├── ingestion/      #   cleaning, article titles, chunkers, pipeline
│   │   ├── embeddings/     #   embedding provider + factory
│   │   ├── llm/            #   model factory (LLM_MODEL / LLM_FAST_MODEL)
│   │   ├── retrieval/      #   vector + hybrid retrievers, RRF, threshold
│   │   ├── semantic/       #   article profiles, entities, catalog, scope guard
│   │   ├── rag/            #   /ask
│   │   ├── agent/          #   /agent/ask: LangGraph workflow, understand/research/grounding
│   │   └── conversations/  #   chat history
│   ├── migrations/         # Alembic
│   ├── scripts/            # ingest.py, enrich.py
│   ├── seed/               # committed article titles and profiles
│   ├── tests/
│   └── Dockerfile
├── docker-compose.yml      # Postgres (pgvector) + the API
├── .env.example
└── PLAN.md                 # design decisions, measurements, checklists
```

## Data license

The passages are Wikipedia text, licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/).

from app.core.settings import Settings


def test_reads_values_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:secret@db:5432/test")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_DIM", "768")

    settings = Settings(_env_file=None)

    assert settings.embedding_provider == "mock"
    assert settings.llm_provider == "mock"
    assert settings.embedding_dim == 768
    assert "secret" in settings.database_url.get_secret_value()


def test_database_url_is_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:secret@db:5432/test")

    settings = Settings(_env_file=None)

    assert "secret" not in repr(settings)

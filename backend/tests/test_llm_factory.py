from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from app.core.settings import Settings
from app.llm.factory import get_llm_model


class Answer(BaseModel):
    text: str
    citations: list[int]


async def test_mock_llm_runs_an_agent_with_typed_output():
    settings = Settings(
        _env_file=None, database_url="postgresql+asyncpg://u:p@localhost/db", llm_provider="mock"
    )
    model = get_llm_model(settings)
    agent = Agent(model, output_type=Answer)

    result = await agent.run("What borders Uruguay?")

    assert isinstance(model, TestModel)
    assert isinstance(result.output, Answer)

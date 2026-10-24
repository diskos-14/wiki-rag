"""Article profiles: one LLM call per article gives its topic, a summary and its main entities."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

Topic = Literal["us_president", "scientist", "country", "animal", "other"]
EntityType = Literal["person", "place", "organization", "event"]

# Whole articles (median 22k, max 76k characters, about 19k tokens): entities that only appear
# later in an article (e.g. Edison in "Nikola Tesla") must not be missed.
MAX_ARTICLE_CHARS = 80_000

INSTRUCTIONS = """\
You profile one English Wikipedia article for a search system.

Return:
- topic: the single best category from the allowed list.
- summary: one sentence saying what the article is about.
- entities: up to 25 of the most important people, places, organizations and events
  named in the text, including the article's own subject.
  - name: the full proper name as written in the text, e.g. "Thomas Edison".
  - aliases: other forms the text uses for the same thing, e.g. ["Edison"]. Only proper
    names; never pronouns or common words.
"""


class EntityProfile(BaseModel):
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)


class ArticleProfile(BaseModel):
    topic: Topic
    summary: str
    entities: list[EntityProfile] = Field(max_length=25)


def build_profile_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[None, ArticleProfile]:
    agent = Agent(
        model,
        output_type=ArticleProfile,
        instructions=INSTRUCTIONS,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )

    @agent.output_validator
    def has_entities(output: ArticleProfile) -> ArticleProfile:
        if not output.entities:
            raise ModelRetry("List at least the article's own subject as an entity.")
        if not output.summary.strip():
            raise ModelRetry("The summary must not be empty.")
        return output

    return agent


def format_profile_prompt(title: str, content: str) -> str:
    return f"Title: {title}\n\nArticle:\n{content[:MAX_ARTICLE_CHARS]}"

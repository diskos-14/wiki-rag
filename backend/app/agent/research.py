"""The research agent: a Pydantic AI tool-calling loop over the knowledge base.

Pydantic AI runs the loop *inside* this agent (call tools, read results, call more tools,
answer). LangGraph (agent/graph.py) only runs the outer workflow loop (retry after a failed
grounding check). One loop each, no overlap.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModelSettings

from app.agent.evidence import Evidence, format_evidence
from app.embeddings.base import EmbeddingProvider
from app.retrieval.retrievers import Retriever
from app.semantic.profiles import Topic
from app.semantic.repository import SemanticRepository

INSTRUCTIONS = """\
You are a research assistant answering questions from a knowledge base of English Wikipedia
articles. Use the tools to find evidence, then answer.

{catalog}

How to research:
- Search before answering. Every search result is numbered evidence like [3].
- For a comparison, research each subject separately: search_passages with title= set to that
  subject's article when it has one.
- find_entity tells you which articles mention someone or something (e.g. a person who has no
  article of their own); then search those articles.
- get_article_profile gives an article's summary and main entities.
- Stop searching once you have the facts; you have a small budget of steps.

How to answer:
- Use only facts from the evidence. No outside knowledge, even if you know the answer.
- Put the number of every piece of evidence you used in `citations`; you may also mark
  claims in the answer like [3].
- For yes/no questions start with "Yes" or "No". Be concise: one to four sentences.
- If the evidence doesn't contain the answer, set `answerable` to false and leave
  `citations` empty.
"""


class DraftAnswer(BaseModel):
    answer: str
    citations: list[int] = Field(description="Numbers of the evidence the answer is based on.")
    answerable: bool = Field(description="False if the evidence doesn't contain the answer.")


@dataclass
class ResearchDeps:
    """Everything the tools need, passed in (no globals). `evidence` grows as the agent works."""

    embeddings: EmbeddingProvider
    retriever: Retriever
    semantic: SemanticRepository
    dataset: str
    catalog: str
    evidence: Evidence = field(default_factory=Evidence)
    search_k: int = 5


def build_research_agent(
    model: Model, *, timeout_seconds: float | None = None
) -> Agent[ResearchDeps, DraftAnswer]:
    agent = Agent(
        model,
        deps_type=ResearchDeps,
        output_type=DraftAnswer,
        model_settings=GoogleModelSettings(
            google_thinking_config={"thinking_level": "low"}, timeout=timeout_seconds
        ),
    )

    @agent.instructions
    def with_catalog(ctx: RunContext[ResearchDeps]) -> str:
        return INSTRUCTIONS.format(catalog=ctx.deps.catalog)

    @agent.tool
    async def search_passages(
        ctx: RunContext[ResearchDeps],
        query: str,
        title: str | None = None,
        topic: Topic | None = None,
    ) -> str:
        """Hybrid search (meaning + keywords) over the knowledge base.

        Args:
            query: What to look for, as a short question or phrase.
            title: Only search this article (an exact title from the catalog).
            topic: Only search articles of this topic.
        """
        deps = ctx.deps
        vector = await deps.embeddings.embed_query(query)
        found = await deps.retriever.retrieve(
            vector, query, deps.search_k, title=title, topic=topic
        )
        if not found.hits:
            return "No passages found. Try other words, or another article or topic."
        return format_evidence(deps.evidence.add(found.hits))

    @agent.tool
    async def get_article_profile(ctx: RunContext[ResearchDeps], title: str) -> str:
        """Topic, one-sentence summary and most-mentioned entities of one article.

        Args:
            title: An exact article title from the catalog.
        """
        profile = await ctx.deps.semantic.article_profile(title, dataset=ctx.deps.dataset)
        if profile is None:
            return f"No article titled {title!r}. Use a title from the catalog."
        entities = ", ".join(f"{name} ({type_})" for name, type_, _ in profile.entities)
        return (
            f"{profile.title} [{profile.topic}]: {profile.summary}\n"
            f"Main entities: {entities or 'none'}"
        )

    @agent.tool
    async def find_entity(ctx: RunContext[ResearchDeps], name: str) -> str:
        """Which articles mention a person, place, organization or event, and how often.

        Args:
            name: The name to look up, e.g. "Thomas Edison" or "Edison".
        """
        mentions = await ctx.deps.semantic.find_entity(name, dataset=ctx.deps.dataset)
        if not mentions:
            return f"No entity named {name!r} in the knowledge base."
        return "\n".join(
            f"{m.name} ({m.type}): mentioned in "
            + ", ".join(f"{title} ({n} passages)" for title, n in m.articles[:5])
            for m in mentions[:3]
        )

    @agent.output_validator
    def citations_are_evidence(ctx: RunContext[ResearchDeps], output: DraftAnswer) -> DraftAnswer:
        unknown = sorted(set(output.citations) - ctx.deps.evidence.numbers)
        if unknown:
            raise ModelRetry(
                f"Citations {unknown} are not evidence you retrieved. Cite only numbers from "
                "search results, or search for what you need."
            )
        if output.answerable and not output.citations:
            raise ModelRetry("An answer must cite at least one piece of evidence.")
        return output

    return agent


def format_research_prompt(
    question: str, sub_questions: Sequence[str] = (), feedback: str | None = None
) -> str:
    """The research task. On a grounding retry, `feedback` says what to fix."""
    lines = [f"Question: {question}"]
    if sub_questions:
        lines += ["", "Research each of these:", *(f"- {q}" for q in sub_questions)]
    if feedback:
        lines += [
            "",
            "Your previous answer failed a fact check. Fix it using the evidence:",
            feedback,
        ]
    return "\n".join(lines)

from dataclasses import dataclass, field

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from app.agent.evidence import Evidence
from app.agent.grounding import format_grounding_prompt
from app.agent.research import ResearchDeps, build_research_agent, format_research_prompt
from app.agent.understand import UnderstandDeps, build_understand_agent, format_understand_prompt
from app.conversations.schemas import Turn
from app.knowledge.schemas import ChunkHit
from app.retrieval.retrievers import RetrievedChunks
from app.semantic.repository import ArticleSummary, EntityMention

CATALOG = "The knowledge base has 2 English Wikipedia articles:\n- US presidents: ..."


def hit(chunk_id: int, title: str = "Abraham Lincoln", content: str = "Born 1809.") -> ChunkHit:
    return ChunkHit(
        chunk_id=chunk_id, document_id=1, title=title, chunk_index=0, content=content, score=0.8
    )


def output(info, data: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, data)])


# --- Evidence ---


def test_evidence_numbers_are_stable_and_deduplicated():
    evidence = Evidence()

    first = evidence.add([hit(10), hit(11)])
    again = evidence.add([hit(11), hit(12)])

    assert [i.number for i in first] == [1, 2]
    assert [i.number for i in again] == [2, 3]  # chunk 11 keeps number 2
    assert evidence.numbers == {1, 2, 3}
    assert [i.hit.chunk_id for i in evidence.get([3, 1, 99])] == [12, 10]


# --- Understand ---


async def test_understand_sees_the_catalog_and_retries_a_comparison_without_sub_questions():
    seen_instructions: list[str] = []
    replies = [
        {"intent": "comparison", "standalone_question": "Who was born first?", "sub_questions": []},
        {
            "intent": "comparison",
            "standalone_question": "Who was born first, Lincoln or Coolidge?",
            "entities": ["Abraham Lincoln", "Calvin Coolidge"],
            "sub_questions": ["When was Lincoln born?", "When was Coolidge born?"],
        },
    ]

    def respond(messages, info):
        seen_instructions.append(messages[-1].instructions or "")
        return output(info, replies.pop(0))

    agent = build_understand_agent(FunctionModel(respond))
    result = await agent.run("Question: ...", deps=UnderstandDeps(CATALOG))

    assert result.output.intent == "comparison"
    assert len(result.output.sub_questions) == 2
    assert CATALOG in seen_instructions[0]


def test_understand_prompt_includes_history_only_when_present():
    history = [Turn(role="user", content="Who was Lincoln?"), Turn(role="assistant", content="A.")]

    assert format_understand_prompt("When was he born?") == "Question: When was he born?"
    with_history = format_understand_prompt("When was he born?", history)
    assert "User: Who was Lincoln?" in with_history
    assert with_history.endswith("Question: When was he born?")


# --- Research (a real tool-calling loop with a scripted model) ---


@dataclass
class FakeEmbeddings:
    dim: int = 3

    async def embed_documents(self, documents):
        raise NotImplementedError

    async def embed_query(self, query):
        return [1.0, 0.0, 0.0]


@dataclass
class FakeRetriever:
    hits: list[ChunkHit]
    calls: list[dict] = field(default_factory=list)

    async def retrieve(self, query_vector, query_text, k, *, title=None, topic=None):
        self.calls.append({"query": query_text, "k": k, "title": title, "topic": topic})
        return RetrievedChunks(self.hits, 0.8)


@dataclass
class FakeSemantic:
    async def article_profile(self, title, *, dataset):
        if title != "Nikola Tesla":
            return None
        return ArticleSummary(
            "Nikola Tesla", "scientist", "Inventor.", [("Thomas Edison", "person", 9)]
        )

    async def find_entity(self, name, *, dataset=None):
        if name != "Edison":
            return []
        return [EntityMention("Thomas Edison", "person", [("Nikola Tesla", 9)])]


def research_deps(hits) -> ResearchDeps:
    return ResearchDeps(
        embeddings=FakeEmbeddings(),
        retriever=FakeRetriever(hits),
        semantic=FakeSemantic(),
        dataset="d",
        catalog=CATALOG,
    )


def tool_returns(messages) -> list[str]:
    return [
        p.content
        for m in messages
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart)
    ]


async def test_research_agent_searches_then_answers_citing_evidence():
    seen: list[str] = []

    def respond(messages, info):
        returns = tool_returns(messages)
        if not returns:  # step 1: search Lincoln's article
            return ModelResponse(
                parts=[
                    ToolCallPart("search_passages", {"query": "born", "title": "Abraham Lincoln"})
                ]
            )
        seen.extend(returns)  # step 2: answer from the evidence
        return output(info, {"answer": "1809 [1].", "citations": [1], "answerable": True})

    deps = research_deps([hit(10)])
    result = await build_research_agent(FunctionModel(respond)).run("When?", deps=deps)

    assert result.output.citations == [1]
    assert deps.retriever.calls == [
        {"query": "born", "k": 5, "title": "Abraham Lincoln", "topic": None}
    ]
    assert seen == ["[1] (Abraham Lincoln) Born 1809."]
    assert len(deps.evidence) == 1


async def test_research_agent_must_cite_retrieved_evidence():
    replies = [
        {"answer": "1809 [7].", "citations": [7], "answerable": True},  # never retrieved
        {"answer": "1809 [1].", "citations": [1], "answerable": True},
    ]

    def respond(messages, info):
        if not tool_returns(messages):
            return ModelResponse(parts=[ToolCallPart("search_passages", {"query": "born"})])
        return output(info, replies.pop(0))

    deps = research_deps([hit(10)])
    result = await build_research_agent(FunctionModel(respond)).run("When?", deps=deps)

    assert result.output.citations == [1]
    assert replies == []  # the first answer was rejected and retried


async def test_research_semantic_tools_report_profiles_and_entities():
    def respond(messages, info):
        returns = tool_returns(messages)
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart("find_entity", {"name": "Edison"}),
                    ToolCallPart("get_article_profile", {"title": "Nikola Tesla"}),
                    ToolCallPart("get_article_profile", {"title": "Mars"}),
                ]
            )
        respond.returns = returns
        return output(info, {"answer": "Not sure.", "citations": [], "answerable": False})

    await build_research_agent(FunctionModel(respond)).run("q", deps=research_deps([]))

    found, profile, missing = respond.returns
    assert found == "Thomas Edison (person): mentioned in Nikola Tesla (9 passages)"
    assert profile.startswith("Nikola Tesla [scientist]: Inventor.")
    assert "Thomas Edison (person)" in profile
    assert "No article titled 'Mars'" in missing


def test_research_prompt_lists_sub_questions_and_retry_feedback():
    prompt = format_research_prompt(
        "Who was born first?",
        ["When was Lincoln born?", "When was Coolidge born?"],
        feedback="Coolidge's birth date is missing.",
    )

    assert "- When was Coolidge born?" in prompt
    assert prompt.endswith("Coolidge's birth date is missing.")
    assert "fact check" not in format_research_prompt("q")


# --- Grounding ---


def test_grounding_prompt_contains_only_the_cited_evidence():
    evidence = Evidence()
    evidence.add([hit(10, content="Born 1809."), hit(11, content="Unrelated.")])

    prompt = format_grounding_prompt("When?", "1809 [1].", evidence.get([1]))

    assert "[1] (Abraham Lincoln) Born 1809." in prompt
    assert "Unrelated." not in prompt

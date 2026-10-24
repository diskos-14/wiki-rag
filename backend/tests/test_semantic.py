import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from app.semantic.catalog import format_catalog
from app.semantic.enrichment import ProfilingIncompleteError, profile_articles
from app.semantic.linking import EntityMatcher, EntityNames
from app.semantic.profiles import build_profile_agent, format_profile_prompt
from app.semantic.repository import ArticleRow

# --- Entity matching ---


def test_matcher_finds_names_and_aliases_as_whole_words():
    matcher = EntityMatcher(
        [
            EntityNames(1, ["Thomas Edison", "Edison"]),
            EntityNames(2, ["Ulysses S. Grant", "Grant"]),
            EntityNames(3, ["Gerald Ford", "Ford"]),
        ]
    )

    assert matcher.match("Tesla worked for Edison in New York.") == {1}
    assert matcher.match("President Grant signed it.") == {2}
    # Case-sensitive: "grant" the verb is not Ulysses S. Grant.
    assert matcher.match("They grant degrees.") == set()
    # Whole words: "Fordham" is not Ford.
    assert matcher.match("He studied at Fordham.") == set()
    assert matcher.match("Ford pardoned Nixon; Edison invented things.") == {1, 3}


def test_matcher_ignores_very_short_aliases():
    matcher = EntityMatcher([EntityNames(1, ["United States", "US"])])
    assert matcher.match("The US economy grew.") == set()
    assert matcher.match("The United States economy grew.") == {1}


# --- Catalog ---


def test_catalog_groups_titles_by_topic():
    catalog = format_catalog(
        [
            ("Uruguay", "country"),
            ("Abraham Lincoln", "us_president"),
            ("Duck", "animal"),
            ("Finland", "country"),
            ("Mystery", None),
        ]
    )

    assert catalog.startswith("The knowledge base has 5 English Wikipedia articles:")
    assert "- US presidents: Abraham Lincoln" in catalog
    assert "- Countries: Finland, Uruguay" in catalog
    assert "- Other: Mystery" in catalog


# --- Profiles ---

PROFILE = {
    "topic": "country",
    "summary": "Uruguay is a country in South America.",
    "entities": [{"name": "Uruguay", "type": "place", "aliases": []}],
}


def returning(*outputs: dict) -> FunctionModel:
    remaining = list(outputs)

    def respond(messages, info):
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, remaining.pop(0))])

    return FunctionModel(respond)


def test_profile_prompt_truncates_long_articles():
    prompt = format_profile_prompt("Uruguay", "x" * 200_000)
    assert prompt.startswith("Title: Uruguay")
    assert len(prompt) < 80_100


async def test_profile_agent_retries_when_no_entities():
    model = returning({**PROFILE, "entities": []}, PROFILE)

    result = await build_profile_agent(model).run("Title: Uruguay")

    assert result.output.entities[0].name == "Uruguay"


async def test_profiles_are_cached_and_failures_resume(tmp_path):
    cache = tmp_path / "profiles.jsonl"
    articles = [ArticleRow(1, "Uruguay", "text", None), ArticleRow(2, "Finland", "text", None)]

    def flaky(messages, info):
        if "Finland" in messages[-1].parts[-1].content:
            raise ModelHTTPError(429, "gemini")
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, PROFILE)])

    with pytest.raises(ProfilingIncompleteError, match="1 articles failed"):
        await profile_articles(articles, build_profile_agent(FunctionModel(flaky)), cache)
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 1  # Uruguay kept

    finland = {**PROFILE, "summary": "Finland is a country."}
    profiles = await profile_articles(articles, build_profile_agent(returning(finland)), cache)

    assert profiles["Uruguay"].summary == PROFILE["summary"]  # from the cache, no call
    assert profiles["Finland"].summary == "Finland is a country."


# --- Entity normalization ---


def test_normalize_drops_common_noun_aliases_and_bare_first_names():
    from app.semantic.enrichment import normalize_entities

    result = normalize_entities(
        [
            ("Gray wolf", "place", ["wolf", "wolves", "Canis lupus"]),
            ("John", "person", []),
            ("Abraham Lincoln", "person", ["Lincoln", "Abe"]),
        ]
    )

    assert result == {
        ("Gray wolf", "place"): {"Canis lupus"},
        ("Abraham Lincoln", "person"): {"Lincoln", "Abe"},
    }


def test_normalize_merges_the_same_entity_from_several_articles():
    from app.semantic.enrichment import normalize_entities

    result = normalize_entities(
        [
            ("Thomas Edison", "person", ["Edison"]),
            ("Thomas Edison", "person", ["Thomas Alva Edison"]),
        ]
    )

    assert result == {("Thomas Edison", "person"): {"Edison", "Thomas Alva Edison"}}


def test_normalize_merges_entities_that_name_each_other():
    from app.semantic.enrichment import normalize_entities

    result = normalize_entities(
        [
            ("United States", "place", ["US", "America"]),
            ("United States of America", "place", ["United States", "USA"]),
            ("United States", "organization", []),  # different type: kept apart
        ]
    )

    # Merged under the longest name.
    assert result == {
        ("United States of America", "place"): {"US", "America", "United States", "USA"},
        ("United States", "organization"): set(),
    }


def test_normalize_adds_surnames_and_drops_ambiguous_one_word_aliases():
    from app.semantic.enrichment import normalize_entities

    result = normalize_entities(
        [
            ("Thomas Edison", "person", []),  # the model gave no aliases
            ("Edison Machine Works", "organization", []),  # starts with "Edison": still ok
            ("Edward M. House", "person", ["House", "Colonel House"]),
            ("White House", "place", []),  # "House" inside another name: ambiguous
            ("Theodore Roosevelt Jr.", "person", []),  # suffix skipped for the surname
        ]
    )

    assert result[("Thomas Edison", "person")] == {"Edison"}
    assert result[("Edward M. House", "person")] == {"Colonel House"}
    assert result[("Theodore Roosevelt Jr.", "person")] == {"Roosevelt"}


def test_normalize_gives_shared_surnames_to_article_subjects():
    from app.semantic.enrichment import normalize_entities

    result = normalize_entities(
        [("Abraham Lincoln", "person", []), ("Mary Todd Lincoln", "person", [])],
        subjects={"Abraham Lincoln"},
    )

    assert result[("Abraham Lincoln", "person")] == {"Lincoln"}
    assert result[("Mary Todd Lincoln", "person")] == set()


def test_normalize_drops_aliases_found_in_too_many_articles():
    from app.semantic.enrichment import normalize_entities

    texts = ["An American story."] * 3 + ["Nothing here."] * 5  # "American" in 3 of 8 articles

    result = normalize_entities(
        [("Native American", "person", []), ("King John", "person", ["John"])],
        article_texts=texts,
        max_alias_article_share=0.25,
    )

    assert result[("Native American", "person")] == set()  # "American": 3 > 2 articles
    assert result[("King John", "person")] == {"John"}  # model's alias; "John" in 0 articles

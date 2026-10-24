"""The knowledge catalog: what the knowledge base covers, as text for agent instructions.

41 articles is small enough to list every title, so the agent knows exactly what it can
answer from (and what is out of scope) without searching first.
"""

from collections.abc import Iterable

TOPIC_LABELS = {
    "us_president": "US presidents",
    "scientist": "Scientists",
    "country": "Countries",
    "animal": "Animals",
    "other": "Other",
}


def format_catalog(articles: Iterable[tuple[str, str | None]]) -> str:
    """`articles` is (title, topic) pairs; articles without a topic go under Other."""
    by_topic: dict[str, list[str]] = {}
    for title, topic in articles:
        by_topic.setdefault(topic or "other", []).append(title)

    total = sum(len(titles) for titles in by_topic.values())
    lines = [f"The knowledge base has {total} English Wikipedia articles:"]
    for topic, label in TOPIC_LABELS.items():
        if titles := by_topic.get(topic):
            lines.append(f"- {label}: {', '.join(sorted(titles))}")
    return "\n".join(lines)

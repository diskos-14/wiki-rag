"""A deterministic guard against wrong "out of scope" refusals.

`understand` (an LLM, possibly a small fast one) decides whether the knowledge base can answer.
Refusing wrongly is the costliest mistake: the agent doesn't even search. So before a refusal,
the question is checked against every article title, entity name and alias in the knowledge
base; if it names one ("Did Tesla work with Edison?"), it's in scope whatever the model said.
"""

import re
from collections.abc import Iterable
from functools import lru_cache

MIN_NAME_CHARS = 3


class ScopeGuard:
    def __init__(self, names: Iterable[str]) -> None:
        unique = sorted(
            {n.strip() for n in names if len(n.strip()) >= MIN_NAME_CHARS}, key=len, reverse=True
        )
        alternatives = "|".join(re.escape(n) for n in unique)
        # Questions are typed casually ("did tesla…"), so matching here ignores case.
        self._pattern = (
            re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE) if unique else None
        )

    def mentioned(self, text: str) -> list[str]:
        """Knowledge-base names found in the text, in order, without duplicates."""
        if self._pattern is None:
            return []
        return list(dict.fromkeys(m.group(0) for m in self._pattern.finditer(text)))


@lru_cache(maxsize=4)
def scope_guard_for(names: tuple[str, ...]) -> ScopeGuard:
    """One compiled guard per distinct set of names (the knowledge base rarely changes)."""
    return ScopeGuard(names)

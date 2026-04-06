"""Regex patterns used by the consistency strategy.

These are pure constants with no runtime dependencies, extracted from
ConsistencyEngine for readability and potential reuse.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Base patterns (raw strings, used inside structured patterns via f-string)
# ---------------------------------------------------------------------------

ENTITY_PATTERN = r"[A-Z][A-Za-z0-9&/@+.\-]*(?:\s+[A-Z0-9][A-Za-z0-9&/@+.\-]*)*"
VALUE_PATTERN = r"[^;\n]+?(?=\s*\.\s*$|\s*$)"

# ---------------------------------------------------------------------------
# Compiled helpers
# ---------------------------------------------------------------------------

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\s*;\s*")

# ---------------------------------------------------------------------------
# 19 structured claim-detection patterns (order matters)
# ---------------------------------------------------------------------------

STRUCTURED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:reports|lists|states|shows|notes|"
        rf"confirms|records|indicates|specifies)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+"
        rf"(?:as|is|at|to)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+confirms\s+"
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:is\s+set\s+to\s+release\s+in|"
        rf"will\s+(?:actually\s+)?be\s+released\s+in|will\s+now\s+release\s+in)\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?screening\s+status\s+of\s+(?P<entity>{ENTITY_PATTERN})\s+"
        rf"(?:is|was|remains)\s+(?:currently\s+)?(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?best\s+metric\s+for\s+(?P<entity>{ENTITY_PATTERN})\s+"
        rf"(?:is|was|remains)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})'s\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)"
        rf"\s+(?:is|was|remains)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+has\s+(?:a|an|the)\s+(?P<attribute>"
        rf"[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:boasts|achieves)\s+(?:a|an|the)\s+"
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of(?:\s+just)?\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:shows|operates with|experiences|"
        rf"has shown|has been recorded with|reveals|"
        rf"exhibits)\s+(?:a|an|the)\s+(?P<attribute>"
        rf"[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of(?:\s+just|(?:\s+only)?)?\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+recorded\s+(?:a|an|the)\s+(?P<attribute>"
        rf"[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+is\s+known\s+for\s+(?:its\s+)?(?:low\s+)?"
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:for|within)\s+(?P<entity>{ENTITY_PATTERN}),\s+(?P<attribute>"
        rf"[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+(?:is|was|remains)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<entity>"
        rf"{ENTITY_PATTERN})\s+(?:is|was|remains)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<entity>"
        rf"{ENTITY_PATTERN})\s+(?:is|was)\s+(?:measured at|recorded at|reported as|"
        rf"reported to be)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+for\s+(?P<entity>"
        rf"{ENTITY_PATTERN})\s+(?:is|was|remains|is reported as|is reported to be|"
        rf"has been reported as|has been reported to be|"
        rf"has been measured at|is measured at|is recorded at|has been noted as|"
        rf"has increased to|has decreased to)\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+has\s+been\s+reported\s+to\s+have\s+"
        rf"(?:a|an|the)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:the\s+)?(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<entity>"
        rf"{ENTITY_PATTERN})\s+is\s+actually\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+for\s+(?P<entity>{ENTITY_PATTERN})"
        rf"\s+(?:is|was|remains)\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"with\s+(?:a|an|the)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+"
        rf"(?P<value>{VALUE_PATTERN}),\s+(?P<entity>{ENTITY_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s*[:\-]\s*(?P<attribute>[a-zA-Z_]"
        rf"[a-zA-Z0-9_\-\s]*?)\s*(?:=|:)\s*(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:is\s+)?characterized\s+by\s+"
        rf"(?:a|an|the)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:according\s+to\s+)?(?P<entity>{ENTITY_PATTERN}),\s+"
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+(?:equals|is|stands\s+at|reaches)\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+demonstrates\s+(?:a|an|the)\s+"
        rf"(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<entity>{ENTITY_PATTERN})\s+(?:features|includes|contains)\s+"
        rf"(?:a|an|the)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+"
        rf"(?P<value>{VALUE_PATTERN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?:a|an|the)\s+(?P<attribute>[a-zA-Z_][a-zA-Z0-9_\-\s]*?)\s+of\s+"
        rf"(?P<value>{VALUE_PATTERN})\s+(?:makes|defines|characterizes|distinguishes)\s+"
        rf"(?P<entity>{ENTITY_PATTERN})",
        re.IGNORECASE,
    ),
)

# ---------------------------------------------------------------------------
# Hint strings for fast pre-filtering (parallel to STRUCTURED_PATTERNS)
# ---------------------------------------------------------------------------

STRUCTURED_PATTERN_HINTS: tuple[tuple[str, ...], ...] = (
    (
        " reports ",
        " lists ",
        " states ",
        " shows ",
        " notes ",
        " confirms ",
        " records ",
        " specifies ",
    ),
    (" confirms ",),
    (" release in ", " released in "),
    ("screening status of",),
    ("best metric for",),
    ("'s ",),
    (" has ", " of "),
    (" boasts ", " achieves "),
    (
        " shows ",
        " operates with ",
        " experiences ",
        " has shown ",
        " recorded with ",
        " reveals ",
        " exhibits ",
    ),
    (" recorded ",),
    (" known for ",),
    ("for ", "within "),
    (" of ", " is ", " remains "),
    (" of ", " measured at ", " recorded at ", " reported as "),
    (
        " for ",
        " reported as ",
        " measured at ",
        " recorded at ",
        " increased to ",
        " decreased to ",
    ),
    (" has been reported to have ", " reported to have "),
    (" actually ",),
    (" for ",),
    ("with ", " of ", ","),
    (":", "=", " - "),
)

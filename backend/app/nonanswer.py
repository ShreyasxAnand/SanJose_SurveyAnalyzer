"""Sentinel non-answer detection, shared by ingest (flags rows at write
time) and induction (filters rows at read time — and remains the fallback
for exports written before the is_nonanswer column existed).

Deliberately import-free so both sides can depend on it without cycles.
"""

# Pure null tokens only. "none"/"nothing" are deliberately NOT here: for a
# question like "what makes you feel unsafe" they are a real answer (nothing
# does) and the taxonomy should surface that as a category, not lose it.
SENTINEL_NON_ANSWERS = {
    "n/a", "na", "n.a", "n.a.", "idk", "i don't know", "i dont know",
    "dont know", "don't know", "no comment", "nil", "nada", "x", "xx",
    "xxx", "?", "??", "???", "-", "--", ".", "..", "...", "unsure",
    "not sure",
}


def is_nonanswer_text(text: str) -> bool:
    """True for sentinel non-answers and effectively-blank strings. The
    punctuation strip is what makes "." / "..." / "n/a." all match."""
    s = text.strip().lower()
    s = s.strip(" \t.!?")
    return not s or s in SENTINEL_NON_ANSWERS

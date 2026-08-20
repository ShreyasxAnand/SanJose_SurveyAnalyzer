"""The generated prompt reference must never drift from the prompts.

A hand-written prompt doc drifted twice in review-consuming ways: it showed the
ROUTE summary as bare category lines (hiding where the lexicon and Locations
lists render) and filed the section-plan merge rule under the wrong grain. Both
sent a review after a defect that did not exist. docs/PROMPTS.md is generated
from the prompt strings; this test is what keeps it that way.
"""
from __future__ import annotations

import pytest

from scripts import prompt_reference


def test_generated_prompt_reference_is_current():
    """Fails when a prompt changed and docs/PROMPTS.md was not regenerated."""
    if not prompt_reference.OUT_PATH.exists():
        pytest.fail("docs/PROMPTS.md is missing — run "
                    "`python -m scripts.prompt_reference` from backend/")
    committed = prompt_reference.OUT_PATH.read_text(encoding="utf-8")
    assert committed == prompt_reference.render(), (
        "docs/PROMPTS.md is stale. Regenerate it with "
        "`python -m scripts.prompt_reference` from backend/.")


def test_every_prompt_constant_is_documented():
    """A new prompt added to a module but not to STAGES would be silently
    absent from the reference — the generator only knows what it is told."""
    import importlib
    import inspect
    import re

    documented = {(mod, const)
                  for _h, mod, _b, consts in prompt_reference.STAGES
                  for const, _role in consts}
    missing = []
    for modname in {mod for _h, mod, _b, _c in prompt_reference.STAGES}:
        module = importlib.import_module(modname)
        src = inspect.getsource(module)
        for const in re.findall(r"^([A-Z][A-Z0-9_]*(?:_SYSTEM|_USER|_BLOCK))\s*=",
                                src, re.M):
            if (modname, const) not in documented:
                missing.append(f"{modname}.{const}")
    assert not missing, (
        f"prompt constants absent from prompt_reference.STAGES: {missing}")


def test_reference_records_the_built_prompt_surface():
    """The doc must name the code-built prompt text, not just the templates —
    that omission is what made the first hand-written version misleading."""
    text = prompt_reference.render()
    for func in ("render_summary", "render_section_plan", "build_synth_prompts",
                 "review_subthemes"):
        assert func in text, func
    # and the specific fact a reviewer needs: the lists live inside {summary}
    assert "lexicon concept list" in text
    assert "Locations list" in text

"""Simple paraphrase and mutation utilities for consistency testing."""

from typing import List


def paraphrase_variants(text: str) -> List[str]:
    """Return lightweight paraphrase-like variants using rule-based transforms.

    Note: This avoids external models to keep tests deterministic.
    """
    variants = set()
    base = text.strip()
    variants.add(base)

    synonyms = {
        "how to": ["how do i", "what is the way to", "what's the method to"],
        "restart": ["reboot", "start again", "bring back up"],
        "connector": ["plugin", "integration"],
        "steps": ["procedure", "process"],
    }

    # Replace phrases
    lowered = base.lower()
    for k, vals in synonyms.items():
        if k in lowered:
            for v in vals:
                variants.add(lowered.replace(k, v))

    # Add punctuation/casing variants
    variants.update({
        base + "?",
        base.capitalize(),
        base.upper(),
        base.replace(" ", "  "),
    })

    # Trim and deduplicate
    return [v.strip() for v in variants if v.strip()]


def small_perturbations(text: str) -> List[str]:
    """Introduce small variants that should not change semantics."""
    variants = set()
    variants.add(text)
    variants.add(text.replace("  ", " "))
    variants.add(text.replace(" .", "."))
    variants.add(text + " ")
    variants.add(text.replace("docker ", "docker\t"))
    return [v.strip() for v in variants if v.strip()]



"""
A/B test variants for FeliNet.

Compare two LLM system prompts:
  - Variant A (control):   the current production prompt.
  - Variant B (treatment): a more explicit prompt we hope improves faithfulness.
"""

from felinet.schemas import RAGConfig

# Variant A = current prompt
VARIANT_A_PROMPT = (
    "You are FeliNet, a feline health and breed knowledge assistant. "
    "Answer questions about cats using ONLY the provided context. "
    "Cite your sources. If the context does not contain enough information, "
    "say so clearly - do not guess or hallucinate."
)

# Variant B = more explicit rules.
# Hypothesis: clearer, numbered grounding rules -> fewer hallucinations -> higher faithfulness.
VARIANT_B_PROMPT = (
    "You are FeliNet, a feline health and breed knowledge assistant.\n"
    "Follow these rules strictly:\n"
    "1. Use ONLY the facts in the provided context. Never add outside knowledge.\n"
    "2. After every claim, cite the chunk number it came from, like [1] or [2].\n"
    "3. If the context does not fully answer the question, state exactly what is "
    "missing instead of guessing.\n"
    "4. Stay focused on what was asked - no filler."
)

VARIANTS = {"A": VARIANT_A_PROMPT, "B": VARIANT_B_PROMPT}


def make_variant_config(variant: str, base: RAGConfig | None = None) -> RAGConfig:
    """
    Build a RAGConfig for the given variant ("A" or "B").

    Start from a base config (so retrieval/embedding settings are shared),
    then override ONLY the system prompt for that variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}. Use one of {list(VARIANTS)}.")

    base = base or RAGConfig()
    # Pydantic v2: model_copy(update=...) returns a copy with fields swapped.
    new_generation = base.generation.model_copy(update={"system_prompt": VARIANTS[variant]})
    return base.model_copy(update={"generation": new_generation})

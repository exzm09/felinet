"""
Guardrails and Safety Layer.
As a gatekeeper for the RAG pipeline it checks things before the query reaches LLM (input guardrails) and after the LLM responds (output guardrails).
Three layers of defense:
    1. Input guardrials: topic check, prompt injestion detection, PII filtering
    2. Confidence gate: reject queries when retrieval scores are too low
    3. Output guardrails: hallucination check, response length, citation verification
Each guardrail returns a GuardrailResult - either PASS (continue) or BLOCK (reject) with a human-readable reason explaining what triggered it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# Result types
class GuardrailAction(str, Enum):
    """
    What the guardrail decided to do.
    """

    PASS = "pass"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """
    The output of any guardrail check.
    - action: did it pass or get blocked?
    - guardrail_name: which check ran
    - reason: why it was blocked
    - details: extra info for debugging
    """

    action: GuardrailAction
    guardrail_name: str
    reason: str = ""
    details: str = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """
        Convenience: True if the guardrail passed
        """
        return self.action == GuardrailAction.PASS

    @property
    def blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK


# Safe fallback messages
FALLBACK_MESSAGES = {
    "off_topic": (
        "I'm FeliNet, a feline health and breed knowledge assistant. "
        "I can only help with cat-related questions - things like health, "
        "nutrition, breeds, behavior, and toxicology. "
        "Could you rephrase your question about cats?"
    ),
    "prompt_injection": (
        "I wasn't able to process that query. Could you rephrase your " "question about cats?"
    ),
    "pii_detected": (
        "It looks like your message contains personal information "
        "(like an email or phone number). For your privacy, I can't process "
        "queries with personal data. Please rephrase without personal details."
    ),
    "low_confidence": (
        "Based on the available sources, I don't have enough information "
        "to answer this question reliably. Try rephrasing, or ask about a "
        "different aspect of feline health and breeds."
    ),
    "hallucination": (
        "I generated a response but it didn't stay grounded in my source "
        "documents. For safety, I won't show an unreliable answer. "
        "Please try rephrasing your question."
    ),
    "too_long": (
        "Something went wrong - my response was unusually long. "
        "Please try asking a more specific question."
    ),
}

# Input Guardrails

# 1. Topic classification (is this about cats?)
# Any of the words/phrases that strongly signal the query is about cats is on-topic

CAT_KEYWORDS = {
    # Animal terms
    "cat",
    "cats",
    "kitten",
    "kittens",
    "kitty",
    "feline",
    "felines",
    "tomcat",
    "tabby",
    "calico",
    "tortoiseshell",
    # Breed names (common ones — not exhaustive)
    "persian",
    "siamese",
    "maine coon",
    "ragdoll",
    "bengal",
    "abyssinian",
    "sphynx",
    "scottish fold",
    "british shorthair",
    "russian blue",
    "birman",
    "burmese",
    "devon rex",
    "cornish rex",
    "oriental",
    "norwegian forest",
    "turkish angora",
    "himalayan",
    "manx",
    "exotic shorthair",
    "savannah",
    "siberian",
    "tonkinese",
    "bombay",
    "chartreux",
    "korat",
    "ocicat",
    "selkirk rex",
    "somali",
    "american shorthair",
    "balinese",
    "javanese",
    "singapura",
    "turkish van",
    "snowshoe",
    "ragamuffin",
    "lykoi",
    "munchkin",
    "nebelung",
    "egyptian mau",
    "havana brown",
    "laperm",
    # Health terms often associated with cats
    "hairball",
    "litter box",
    "litterbox",
    "catnip",
    "scratching post",
    "felv",
    "fiv",
    "fip",
    "feline leukemia",
    "feline immunodeficiency",
    "feline infectious peritonitis",
    # Food/nutrition terms specific to cats
    "wet food",
    "dry food",
    "kibble",
    "taurine",
    # Behavior terms
    "purring",
    "kneading",
    "zoomies",
    "meow",
    "meowing",
    "hissing",
    "spraying",
}
# Queries that are too short or too vague - we can't meaningfully classify them, but we also don't want to block "What breed is my cat?" which is only 6 words.
MIN_QUERY_LENGTH = 3  # characters


def check_topic(query: str) -> GuardrailResult:
    """
    Check if the query is about cats/feline topics
    1. Normalize the query to lowercase
    2. Check if any cat-related keyword appears in the query
    """
    query_lower = query.lower().strip()

    # Very short queries
    if len(query_lower) < MIN_QUERY_LENGTH:
        return GuardrailResult(
            action=GuardrailAction.PASS,
            guardrail_name="topic_check",
            reason="Query too short to classify, allong through",
        )

    # Check for any cat keyword in the query
    for keyword in CAT_KEYWORDS:
        if keyword in query_lower:
            return GuardrailResult(
                action=GuardrailAction.PASS,
                guardrail_name="topic_check",
                details={"matched_keyword": keyword},
            )

    # No cat keyword found
    return GuardrailResult(
        action=GuardrailAction.BLOCK,
        guardrail_name="topic_check",
        reason=f"Query does not appear to be about cats: '{query[:50]}'",
        details={"query_preview": query[:80]},
    )


# 2. Prompt injection detection
# Detect injection attack by looking for sus patterns, it catches the most common attacks but is not 100% bulletproof
INJECTION_PATTERNS = [
    # Direct instruction overrides
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?prior\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"override\s+(all\s+)?(your|system)\s+instructions",
    # Role manipulation - trying to make the LLM pretend to be something else
    r"you\s+are\s+now\s+(a|an)\s+",
    r"act\s+as\s+(a|an)\s+(?!cat|kitten|feline)",
    r"pretend\s+(to\s+be|you're)\s+(a|an)\s+",
    r"switch\s+to\s+.{0,20}mode",
    r"enter\s+.{0,20}mode",
    # System prompt extraction - trying to see the hidden instructions
    r"(reveal|show|display|print|output)\s+.{0,15}(your|the|system)\s+(prompt|instructions|rules)",
    r"what\s+(are|is)\s+your\s+(system\s+)?prompt",
    r"repeat\s+(your|the)\s+(system\s+)?prompt",
    # Jailbreak attempts
    r"do\s+anything\s+now",
    r"DAN\s+mode",
    r"developer\s+mode",
    r"no\s+restrictions",
    r"bypass\s+(your|all|the)\s+(safety|filter|restriction)",
    # Delimiter attacks — trying to "close" the system prompt context
    r"```\s*(system|end|admin)",
    r"<\|?(system|end|admin)\|?>",
    r"\[SYSTEM\]",
    r"\[INST\]",
]

# Pre-compile for speed
_COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
]


def check_prompt_injection(query: str) -> GuardrailResult:
    """
    Scan the query for known prompt injection patterns
    """
    for pattern in _COMPILED_INJECTION_PATTERNS:
        match = pattern.search(query)
        if match:
            logger.warning(f"Prompt injection detected: '{match.group()}' in query")
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                guardrail_name="prompt_injection",
                reason=f"Query contains prompt injection pattern: '{match.group()}'",
                details={
                    "matched_pattern": match.group(),
                    "pattern_index": _COMPILED_INJECTION_PATTERNS.index(pattern),
                },
            )
    return GuardrailResult(action=GuardrailAction.PASS, guardrail_name="prompt_injection")


# 3. PII filtering
PII_PATTERNS = {
    "email": re.compile(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    ),
    "phone_us": re.compile(
        # Matches: (123) 456-7890, 123-456-7890, 123.456.7890, 1234567890
        r"(?<!\d)(\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4})(?!\d)",
    ),
    "ssn": re.compile(
        # Matches: 123-45-6789 (US Social Security Number format)
        r"\b\d{3}-\d{2}-\d{4}\b",
    ),
    "credit_card": re.compile(
        # Matches: 16-digit numbers with optional spaces/dashes
        r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    ),
}


def check_pii(query: str) -> GuardrailResult:
    """
    Scan the query for PII
    """
    for pii_type, pattern in PII_PATTERNS.items():
        match = pattern.search(query)
        if match:
            logger.warning(f"PII detected ({pii_type}) in query")
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                guardrail_name="pii_filter",
                reason=f"Query contains {pii_type} - not sending to LLM for privacy",
                details={"pii_type": pii_type},
            )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        guardrail_name="pii_filter",
    )


# Combines input guardrail runner


def run_input_guardrails(query: str) -> list[GuardrailResult]:
    """
    Run all input guardrails on a query. Return a list of results.

    The order matters:
    1. Prompt injection first - if it's an attack, don't bother checking topic
    2. PII filter second - privacy before topic relevance
    3. Topic check last - only matters if the query is safe

    Returns:
        List of GuardrailResult objects (one per check).
        If ANY has action=BLOCK, the pipeline should stop.
    """
    results = []

    # Prompt Injection (most critical)
    injection_result = check_prompt_injection(query)
    results.append(injection_result)
    if injection_result.blocked:
        return results

    # PII
    pii_result = check_pii(query)
    results.append(pii_result)
    if pii_result.blocked:
        return results

    # Topic check
    topic_result = check_topic(query)
    results.append(topic_result)

    return results


# Confidence Gate - after retrieval, before generation
DEFAULT_MIN_SCORE = 0.25  # Minimum retrieval score threshold
DEFAULT_MIN_CHUNKS = 1  # Need at least 1 chunk above threshold


def check_retrieval_confidence(
    chunks: list, min_score: float = DEFAULT_MIN_SCORE, min_chunks: int = DEFAULT_MIN_CHUNKS
) -> GuardrailResult:
    """
    Check if retrieval returned sufficiently relevant chunks.
    Parameters
    ----------
    chunks : list[RetrievedChunk]
        The chunks returned by retrieval.
    min_score : float
        Minimum score a chunk must have to count as "relevant."
    min_chunks : int
        Minimum number of relevant chunks needed to proceed.
    """
    if not chunks:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="retrieval_confidence",
            reason="No chunks retrieved - cannot generate a grounded answer",
            details={"num_chunks": 0, "min_score": min_score},
        )

    # Count chunks above threshold
    scores = [getattr(c, "score", 0.0) for c in chunks]
    relevant_count = sum(1 for s in scores if s >= min_score)
    top_score = max(scores) if scores else 0.0

    if relevant_count < min_chunks:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="retrieval_confidence",
            reason=(
                f"Only {relevant_count} chunk(s) above threshold {min_score} (need {min_chunks}). Top score: {top_score:.3f}"
            ),
            details={
                "relevant_count": relevant_count,
                "top_score": top_score,
                "min_score": min_score,
                "all_scores": [round(s, 3) for s in scores],
            },
        )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        guardrail_name="retrieval_confidence",
        details={
            "relevant_count": relevant_count,
            "top_score": top_score,
        },
    )


# Output Guardrails - run after the LLM generates an answer

# 1. Citation / Hallucination check
CITATION_PATTERN = re.compile(r"\[\d+\]")
# Check if the answer content actually overlaps with the context. If the LLM says something that doesn't appear anywhere in the context, it's probably hallucinating.
MIN_OVERLAP_RATIO = 0.15


def check_hallucination(answer: str, context: str, chunks: list) -> GuardrailResult:
    """
    Check is the LLM's answer is grounded in the retrieved context
    Three checks:
    1. Does the answer contain citation markers like [1], [2]?
    2. Do the cited numbers actually correspond to real chunks?
    3. Does the answer's vocabulary overlap with the context?

    Parameters
    ----------
    answer : str
        The LLM-generated response.
    context : str
        The formatted context string that was sent to the LLM.
    chunks : list[RetrievedChunk]
        The retrieved chunks (to verify citation numbers).
    """
    # Special case: "I don't know" type answers are fine without citations
    idk_phrases = [
        "i don't have enough information",
        "the context doesn't contain",
        "based on the available sources",
        "i cannot answer",
        "not enough information",
        "i'm not able to answer",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in idk_phrases):
        return GuardrailResult(
            action=GuardrailAction.PASS,
            guardrail_name="hallucination_check",
            reason="Answer acknowledges uncertainty - this is good behavior",
            details={"answer_type": "uncertainty_acknowledgment"},
        )

    # Check 1: Are there ANY citations?
    citations = CITATION_PATTERN.findall(answer)
    if not citations:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="hallucination_check",
            reason="Answer contains no source citations - likely hallucinating",
            details={"citations_found": 0},
        )
    # Check 2: Do cited numbers correspond to actual chunks?
    cited_numbers = {int(c.strip("[]")) for c in citations}
    max_valid = len(chunks)
    invalid_citations = {n for n in cited_numbers if n < 1 or n > max_valid}
    if invalid_citations:
        logger.warning(
            f"Answer cites non-existent sources: {invalid_citations} (only {max_valid} chunks available)"
        )
        # Don't block (the LLM sometimes miscounts)

    # Check 3: Word overlap between answer and context
    answer_words = set(answer_lower.split())
    context_words = set(context.lower().split())

    # Remove any common words that would inflate overlap
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "and",
        "or",
        "but",
        "not",
        "no",
        "this",
        "that",
        "it",
        "its",
        "as",
        "if",
        "then",
        "than",
        "so",
        "up",
        "out",
        "about",
    }
    answer_content_words = answer_words - stop_words
    context_content_words = context_words - stop_words

    if not answer_content_words:
        return GuardrailResult(
            action=GuardrailAction.PASS,
            guardrail_name="hallucination_check",
            details={"note": "answer had no content words"},
        )
    overlap = answer_content_words & context_content_words
    overlap_ratio = len(overlap) / len(answer_content_words)

    if overlap_ratio < MIN_OVERLAP_RATIO:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="hallucination_check",
            reason=f"Low word overlap between answer and context ({overlap_ratio:.0%} < {MIN_OVERLAP_RATIO:.0%}). Answer may not be grounded in retrieved sources.",
            details={
                "overlap_ratio": round(overlap_ratio, 3),
                "threshold": MIN_OVERLAP_RATIO,
                "num_citations": len(citations),
            },
        )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        guardrail_name="hallucination_check",
        details={
            "num_citations": len(citations),
            "overlap_ratio": round(overlap_ratio, 3),
            "cited_sources": sorted(cited_numbers),
        },
    )


# 2. Response length check
MAX_RESPONSE_LENGTH = 3000  # characters — about 500-600 words
MIN_RESPONSE_LENGTH = 10  # anything shorter is suspicious


def check_response_length(answer: str) -> GuardrailResult:
    """
    Check if the response is within reasonable length bounds
    """
    length = len(answer)

    if length < MIN_RESPONSE_LENGTH:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="response_length",
            reason=f"Response too short ({length} chars). May be a generation failure.",
            details={"length": length, "min": MIN_RESPONSE_LENGTH},
        )

    if length > MAX_RESPONSE_LENGTH:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            guardrail_name="response_length",
            reason=f"Response too long ({length} chars, max {MAX_RESPONSE_LENGTH}). May be uncontrolled generation.",
            details={"length": length, "max": MAX_RESPONSE_LENGTH},
        )

    return GuardrailResult(
        action=GuardrailAction.PASS,
        guardrail_name="response_length",
        details={"length": length},
    )


# Combined output guardrail runner
def run_output_guardrails(answer: str, context: str, chunks: list) -> list[GuardrailResult]:
    """
    Run all output guardrails on an LLM response.

    Parameters
    ----------
    answer : str
        The LLM-generated response text.
    context : str
        The formatted context that was sent to the LLM.
    chunks : list[RetrievedChunk]
        The retrieved chunks used for context.

    Returns:
        List of GuardrailResult objects.
        If ANY has action=BLOCK, the pipeline should return a fallback message.
    """
    results = []
    # 1. Response length check (fast, catches obvious failures)
    length_result = check_response_length(answer)
    results.append(length_result)

    # 2. Hallucination / citation check
    hallucination_result = check_hallucination(answer, context, chunks)
    results.append(hallucination_result)

    return results

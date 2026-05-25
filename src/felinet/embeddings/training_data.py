"""
Generate synthetic (question, passage) training pairs for embedding fine-tuning.

1. Load all chunks from the corpus
2. For each chunk, ask GPT to generate 2-3 questions a cat owner would ask
3. Each (question, chunk_text) becomes a training pair
4. Save with checkpointing to keep progress is interrupted.
"""
from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configuration
class TrainingDataConfig:
    """
    Setting for the synthetic question generation pipeline.
    """
    # How many questions to generate per chunk
    questions_per_chunk: int = 3
    # Which LLM generates the questions
    model: str = "gpt-4o-mini"
    # Low tem - more focused, deterministic questions
    # High tem - more creative but sometimes off topic
    temperature: float = 0.7
    
    output_path: str = "data/training/raw_pairs.jsonl"
    # Tracks which chunks are already processed for resuming if the script partway through
    checkpoint_path: str = "data/training/checkpoint.json"

    delay: float = 0.1

# Prompt
# Goal produce realistic & standalone questions with diversity, specificity, and JSON format (easy to parse, no regex)

QUESTION_PROMPT = """You are helping create training data for a feline health search engine.

Given the following passage about cats, generate {n} questions that a cat owner, veterinary student, or shelter volunteer would realistically ask that this passage answers.

Requirements:
- Questions must be SPECIFIC to the information in the passage (not generic)
- Questions should be standalone (make sense without seeing the passage)
- Each question should be 1-2 sentences
- Vary the question style: some direct ("What causes..."), some situational ("My cat is...")
- Do NOT include answers, only questions

Passage:
---
{passage}
---

Respond with ONLY a JSON array of strings, no other text. Example:
["Question 1 here?", "Question 2 here?", "Question 3 here?"]"""

# Core Functions
def generate_questions_for_chunk(
        chunk_text: str,
        config: TrainingDataConfig,
        api_key: str
) -> list[str]:
    """
    Send one chunk to GPT-4o-mini and get back a list of questions.
    Parameters
    ----------
    chunk_text : str
        The text content of a single chunk from your corpus.
    config : TrainingDataConfig
        Generation settings (model, temperature, questions_per_chunk).
    api_key : str
        Your OpenAI API key.

    Returns
    -------
    list[str]
        A list of generated questions. May be fewer than requested if
        the LLM returns invalid JSON (handle that gracefully).
    """
    prompt = QUESTION_PROMPT.format(
        n=config.questions_per_chunk,
        passage=chunk_text
    )
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": 512,
        },
        timeout=30,
    )

    if response.status_code != 200:
        logger.warning(f"API error {response.status_code}: {response.text[:200]}")
        return []
    
    content = response.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON array from LLM's response.
    content = content.strip("`")
    if content.startswith("json"):
        content = content[4:].strip()

    try:
        questions = json.loads(content)
        if isinstance(questions, list):
            return [q for q in questions if isinstance(q, str) and len(q.strip()) > 10]
        return []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM response as JSON: {content[:100]}...")
        return []
    
def load_checkpoint(path: str) -> set[str]:
    """
    Load the set of chunk IDs that have already been processed
    """
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("completed_chunk_ids", []))
    return set()

def save_checkpoint(path: str, completed_ids: set[str]) -> None:
    """
    Save the set of completed chunk IDs to disk.
    """ 
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"completed_chunk_ids": sorted(completed_ids)}, f)

def run_generation(
        chunks: list[dict],
        config: TrainingDataConfig | None = None
) -> Path:
    """
    Generate synthetic questions for all chunks in the corpus.
    1. Loads any existing checkpoint
    2. Loops through each chunk, calling GPT
    3. Saves each (question, passage) pair to a JSON file
    4. Updates the checkpoint after each chunk
    5. Prints progress
    Parameters
    ----------
    chunks : list[dict]
        Each dict needs at minimum: {"id": str, "content": str}
        Additional metadata (source, title, etc.) is preserved in the output.
    config : TrainingDataConfig, optional
        Generation settings. Uses defaults if not provided.

    Returns
    -------
    Path
        Path to the output JSONL file containing all pairs.
    """
    if config is None:
        config = TrainingDataConfig()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env file")
    
    # Create output directory
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    completed = load_checkpoint(config.checkpoint_path)
    logger.info(f"Checkpoint: {len(completed)} chunks already processed")

    # Filter to only unprocessed chunks
    remaining = [c for c in chunks if c["id"] not in completed]
    logger.info(f"Remaining: {len(remaining)} chunks to process")

    if not remaining:
        print("All chunks already processed.")
        return output_path
    
    # Open output file in append mode
    total_pairs = 0
    errors = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for i, chunk in enumerate(remaining):
            chunk_id = chunk["id"]
            chunk_text = chunk["content"]

            # Generate questions for chunk
            try:
                questions = generate_questions_for_chunk(chunk_text, config, api_key)
            except Exception as e:
                logger.error(f"Error on chunk {chunk_id}: {e}")
                errors += 1
                continue

            # Write each (question, passage) pair as a JSON line
            for q in questions:
                pair = {
                    "query": q,
                    "positive": chunk_text,
                    "chunk_id": chunk_id,
                    "source": chunk.get("source", ""),
                    "title": chunk.get("title", "")
                }
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
                total_pairs += 1

            # Update checkpoint
            completed.add(chunk_id)
            save_checkpoint(config.checkpoint_path, completed)

            # Progress update every 50 chunks
            if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
                print(
                    f"  Progress: {i + 1}/{len(remaining)} chunks | "
                    f"{total_pairs} pairs generated | {errors} errors"
                )

            # Small delay to be nice to the API
            time.sleep(config.delay)

    print(f"\n{'=' * 60}")
    print(f"DONE: Generated {total_pairs} training pairs")
    print(f"  Output: {output_path}")
    print(f"  Errors: {errors}")
    print(f"  Total chunks processed: {len(completed)}")
    print(f"{'=' * 60}")

    return output_path

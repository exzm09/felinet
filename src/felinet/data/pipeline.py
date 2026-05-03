"""
FelinNet Data Ingestion Pipeline
Orchestrated with Perfect: runs all scrapers, validates the combined corpus, and saves the final dataset as DVC-tracked JSON.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from prefect import flow, task
from prefect.logging import get_run_logger

from felinet.schemas import DataSource, SourceDocument

# Tasks - each @task is one step Perfect tracks indenpendently with its own retry logic, timing, state tracking, and logging

@task(retries=2, retry_delay_seconds=30, name="scrape-cornell")
def run_cornell_scaper(max_articles: int | None = None) -> list[SourceDocument]:
    """
    Task: Run the Cornell Feline Health Center scraper.
    If the scraper crashes, wait 30 secs and try again, up to 2 extra times.
    """
    logger = get_run_logger()
    logger.info("Starting Cornell scraper...")
    # Import here to avoid circular imports and keep startup fast
    from felinet.data.cornell_scraper import scrape_cornell
    docs = scrape_cornell(max_articles=max_articles)
    logger.info(f"Cornell scraper finished: {len(docs)} documents")
    return docs

@task(retries=2, retry_delay_seconds=30, name="scrape-wikipedia")
def run_wikipedia_scraper(max_breeds: int | None = None) -> list[SourceDocument]:
    """
    Task: Run the Wikipedia cat breeds scraper.
    """
    logger = get_run_logger()
    logger.info("Starting Wikipedia scraper...")

    from felinet.data.wikipedia_scraper import scrape_wikipedia_breeds
    docs = scrape_wikipedia_breeds(max_breeds=max_breeds)
    logger.info(f"Wikipedia scraper finished: {len(docs)} documents")
    return docs

@task(retries=2, retry_delay_seconds=30, name="scrape-cfa")
def run_cfa_scraper(max_breeds: int | None = None) -> list[SourceDocument]:
    """
    Task: Run the CFA breed profiles scraper.
    """
    logger = get_run_logger()
    logger.info("Starting CFA scraper...")

    from felinet.data.cfa_scraper import scrape_cfa_breeds
    docs = scrape_cfa_breeds(max_breeds=max_breeds)
    logger.info(f"CFA scraper finished: {len(docs)} documents")
    return docs

@task(name="validate-corpus")
def validate_corpus(documents: list[SourceDocument]) -> dict:
    """
    Task: Run data quality checks on the combined corpus
    Returns a quality dict with pass/fail status and details
    """
    logger = get_run_logger()
    logger.info(f"Validating corpus of {len(documents)} documents...")

    report = {
        "total_documents": len(documents),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {},
        "passed": True
    }

    # Check 1: Row count in expected range
    # Way too few documents -> something broke in scraping.
    # Way too many documents -> might have duplication or garbage.
    min_expected = 50
    max_expected = 500
    count_ok = min_expected <= len(documents) <= max_expected
    report["checks"]["row_count_range"] = {
        "passed": count_ok,
        "detail": f"{len(documents)} docs (expected {min_expected}~{max_expected})"
    }
    if not count_ok:
        report["passed"] = False
        logger.warning(f"Row count check FAILED: {len(documents)} documents")

    # Check 2: No empty content
    # Pydantic enforces min_length=50, but double-check in case something bypassed the model
    empty_docs = [d for d in documents if not d.content or len(d.content.strip()) < 50]
    no_empty = len(empty_docs) == 0
    report["checks"]["no_empty_content"] = {
        "passed": no_empty,
        "detail": f"{len(empty_docs)} documents with empty/short content"
    }
    if not no_empty:
        report["passed"] = False
        logger.warning(f"Empty content check FAILED: {len(empty_docs)} bad docs")

    # Check 3: All sources are form known DataScource enum
    # Catch typos or unexpected sources sneaking in
    valid_sources = {s.value for s in DataSource}
    actual_sources = {d.source.value if isinstance(d.source, DataSource) else d.source for d in documents}
    unknown_sources = actual_sources - valid_sources
    sources_ok = len(unknown_sources) == 0
    report["checks"]["sources_in_expected_set"] = {
        "passed": sources_ok,
        "detail": f"Sources found: {sorted(actual_sources)}. Unknown: {sorted(unknown_sources) if unknown_sources else 'None'}"
    }
    if not sources_ok:
        report["passed"] = False
        logger.warning(f"Source check FAILED: unknown sources {unknown_sources}")

    # Check 4: Content length within bounds
    # Extremely short docs are probably extraction failures.
    # Extremely long docs might be entire websites scraped by accident.
    min_chars = 100
    max_chars = 100_100
    out_of_bounds = [
        d.id for d in documents if len(d.content) < min_chars or len(d.content) > max_chars
    ]
    length_ok = len(out_of_bounds) == 0
    report["checks"]["content_length_in_bounds"] = {
        "passed": length_ok,
        "detail": f"{len(out_of_bounds)} docs outside {min_chars}~{max_chars} chars"
    }
    if not length_ok:
        report["passed"] = False
        logger.warning(f"Content length check FAILED: {out_of_bounds[:5]}...")

    # Summary
    # Per-source breakdown to see distribution
    source_counts = {}
    for doc in documents:
        src = doc.source.value if isinstance(doc.source, DataSource) else doc.source
        source_counts[src] = source_counts.get(src, 0) + 1
    report["source_breakdown"] = source_counts

    passed_count = sum(1 for c in report["checks"].values() if c["passed"])
    total_checks = len(report["checks"])

    if report["passed"]:
        logger.info(f"All {total_checks} quality checks PASSED")
    else:
        logger.warning(f"Quality checks: {passed_count} / {total_checks} passed - review the report for details")
    return report

@task(name="save-corpus")
def save_combined_corpus(
    documents: list[SourceDocument],
    quality_report: dict,
    output_dir: str = "data/processed"
) -> Path:
    """
    Task: Save the validated corpus as a single combined JSON file.
    """
    logger = get_run_logger()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save combined corpus
    docs_data = [doc.model_dump(mode="json") for doc in documents]
    corpus_file = output_path / "felinet_corpus.json"
    with open(corpus_file, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Corpus saved: {len(documents)} documents -> {corpus_file}")

    # Save quality report alongside the corpus
    report_file = output_path / "quality_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Quality report saved: {report_file}")

    return corpus_file

# FLOW - the whole pipeline that orchestrates the tasks
@flow(name="felinet-data-ingestion", log_prints=True)
def ingest_corpus(
    test_mode: bool = False,
    source: str | None = None
) -> dict:
    """
    Main data ingestion flow.
    Runs all scrapers, validates the combined corpus, and saves everything to disk.
    Args:
        test_mode: if True, only scrape 5 articles per source (fast testing)
        source: if set, only run one scraper ("cornell", "wikipedia", or "cfa")
    Returns:
        Summary dict with document counts and quality report.
    """
    print("=" * 60)
    print("  FeliNet Data Ingestion Pipeline")
    print(f"  Mode: {'TEST (5 per source)' if test_mode else 'FULL'}")
    print(f"  Source filter: {source or 'all'}")
    print("=" * 60)

    limit = 5 if test_mode else None
    all_documents: list[SourceDocument] = []

    # Run scrapers
    # Each scraper is a @task

    if source is None or source == "cornell":
        cornell_docs = run_cornell_scaper(max_articles=limit)
        all_documents.extend(cornell_docs)
        print(f"  Cornell: {len(cornell_docs)} documents")

    if source is None or source == "wikipedia":
        wiki_docs = run_wikipedia_scraper(max_breeds=limit)
        all_documents.extend(wiki_docs)
        print(f"  Wikipedia: {len(wiki_docs)} documents")

    if source is None or source == "cfa":
        cfa_docs = run_cfa_scraper(max_breeds=limit)
        all_documents.extend(cfa_docs)
        print(f"  CFA: {len(cfa_docs)} documents")

    print(f"\n  Total documents collected: {len(all_documents)}")

    # Validate
    quality_report = validate_corpus(all_documents)

    if not quality_report["passed"]:
        print("\n   Some quality checks failed - review quality_report.json")
        print("     Pipeline continues (data is saved) but check the issues.")
    else:
        print("\n   All quality checks passed!")

    # Save
    output_file = save_combined_corpus(all_documents, quality_report)

    # Summary
    summary = {
        "total_documents": len(all_documents),
        "source_breakdown": quality_report.get("source_breakdown", {}),
        "quality_passed": quality_report["passed"],
        "output_file": str(output_file)
    }

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print(f"  Documents: {summary['total_documents']}")
    print(f"  Sources: {summary['source_breakdown']}")
    print(f"  Quality: {'PASSED' if summary['quality_passed'] else 'ISSUES FOUND'}")
    print(f"  Output: {summary['output_file']}")
    print("=" * 60)

    return summary

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FeliNet data ingestion pipeline")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only scrape 5 articles per source"
    )

    parser.add_argument(
        "--source",
        choices=["cornell", "wikipedia", "cfa"],
        default=None,
        help="Run only one scraper (default: all)"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    result = ingest_corpus(test_mode=args.test, source=args.source)


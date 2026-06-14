"""
DIY drift detection, NumPy only with an Evidently HTML report
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CENTROID_PATH = Path("data/monitoring/corpus_centroid.npy")


# The corpus centroid is the center of mass of everything FeliNet knows - average of all chunk embeddings (average cat topic in knowledge base)
def compute_corpus_centroid(
    collection_name: str = "felinet_chunks",
    url: str = "http://localhost:6333",
    cache: bool = True,
) -> np.ndarray:
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url)
    points, _ = client.scroll(
        collection_name=collection_name,
        with_vectors=True,
        with_payload=False,
        limit=100_000,  # one scroll gets them all
    )
    raw = [p.vector for p in points]
    if raw and isinstance(raw[0], dict):
        key = next(iter(raw[0]))
        raw = [v[key] for v in raw]

    vectors = np.asarray(raw, dtype=np.float32)
    centroid = vectors.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    if cache:
        CENTROID_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CENTROID_PATH, centroid)
    logger.info(f"Corpus centroid computed from {len(vectors)} chunks")
    return centroid


def load_corpus_centroid() -> np.ndarray:
    return np.load(CENTROID_PATH) if CENTROID_PATH.exists() else compute_corpus_centroid()


# Cosine similarity: how aligned two vectors are (1 -> identical direction, 0 -> unrelated). Since it's already normalized -> just dot product
def cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def query_centroid_similarity(query_embedding, centroid: np.ndarray | None = None) -> float:
    """
    Real time signal: how on topic is the query vs the corpus center
    """
    if centroid is None:
        centroid = load_corpus_centroid()
    return cosine_sim(query_embedding, centroid)


# PSI (Population Stability Index): a standard number for "how much did a 1-D distribution shift"
# <0.1 - stable; 0.1-0.25 - moderate drift; >0.25 - major drift.
# Apply it to the distribution of similarity centroid score: reference queries vs. recent ones.
def population_stability_index(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference)
    current = np.asarray(current)

    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)

    cur_pct = np.histogram(current, bins=edges)[0] / max(len(current), 1)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# Batch drift: reference query embeddings vs current query embeddings
def batch_drift_report(reference_embeddings, current_embeddings) -> dict:
    centoid = load_corpus_centroid()
    ref = np.asarray(reference_embeddings, dtype=np.float32)
    cur = np.asarray(current_embeddings, dtype=np.float32)

    # How far apart are the two question-clouds
    # Cosine distance of thier means.
    centroid_drift = 1.0 - cosine_sim(ref.mean(axis=0), cur.mean(axis=0))

    # Did the on-topic-ness dist shift?
    # PSI on similarity to corpus
    ref_sims = [cosine_sim(v, centoid) for v in ref]
    cur_sims = [cosine_sim(v, centoid) for v in cur]

    psi = population_stability_index(ref_sims, cur_sims)

    return {
        "centroid_shift_cosine_distance": round(centroid_drift, 4),
        "psi_on_corpus_similarity": round(psi, 4),
        "ref_mean_corpus_similarity": round(float(np.mean(ref_sims)), 4),
        "cur_mean_corpus_similarity": round(float(np.mean(cur_sims)), 4),
        "n_reference": len(ref),
        "n_current": len(cur),
        "drift_detected": bool(centroid_drift > 0.1 or psi > 0.25),
    }


def evidently_html_report(
    reference_embeddings,
    current_embeddings,
    out_path: str = "data/monitoring/drift_report.html",
) -> str:
    """
    Generate an HTML drift report. Written for Evidently 0.7.x (current).
    """
    import pandas as pd

    ref = np.asarray(reference_embeddings, dtype=np.float32)
    cur = np.asarray(current_embeddings, dtype=np.float32)
    cols = [f"dim_{i}" for i in range(ref.shape[1])]
    ref_df = pd.DataFrame(ref, columns=cols)
    cur_df = pd.DataFrame(cur, columns=cols)

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    # (current, reference)
    result = report.run(cur_df, ref_df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.save_html(out_path)
    logger.info(f"Saved Evidently drift report to {out_path}")
    return out_path

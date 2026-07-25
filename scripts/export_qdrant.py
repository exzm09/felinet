# scripts/export_qdrant.py
import json

import pandas as pd
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")
COLLECTION = "felinet_chunks"

points, offset = [], None
while True:  # scroll = paginate through every point
    batch, offset = client.scroll(
        collection_name=COLLECTION,
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )
    points.extend(batch)
    if offset is None:
        break

rows = [
    {"id": p.id, "vector": [float(x) for x in p.vector], "payload": json.dumps(p.payload)}
    for p in points
]
pd.DataFrame(rows).to_parquet("data/qdrant_export.parquet")
print(f"Exported {len(rows)} points -> data/qdrant_export.parquet")

import os
import time

import paths  # noqa: F401
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from knowledgebase import load_knowledge_base

# ── Pinecone setup ────────────────────────────────────────────────────────────

api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

spec = ServerlessSpec(cloud='aws', region='us-east-1')

index_name = 'langgraph-research-agent'
dims = 384  # all-MiniLM-L6-v2 output dimension

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=dims,
        metric='cosine',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
print(index.describe_index_stats())

# ── Load pre-built knowledge base (no re-chunking or re-embedding) ────────────

print("Loading knowledge base from disk...")
data = load_knowledge_base()

# ── Upsert in batches ─────────────────────────────────────────────────────────

batch_size = 64

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i + batch_size)
    batch = data[i:i_end].to_dict(orient='records')

    ids = [r['id'] for r in batch]
    embeds = [r['embedding'] for r in batch]  # already computed in knowledgebase.py
    metadata = [{
        'arxiv_id': r['arxiv_id'],
        'title': r['title'],
        'chunk': r['chunk'],
    } for r in batch]

    index.upsert(vectors=list(zip(ids, embeds, metadata)))

print("Done. Index stats:")
print(index.describe_index_stats())
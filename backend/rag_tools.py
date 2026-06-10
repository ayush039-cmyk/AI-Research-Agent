import os
import requests
import numpy as np
import paths  # noqa: F401
from pinecone import Pinecone
from langchain_core.tools import tool

# ── Pinecone & HuggingFace setup ─────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('langgraph-research-agent')

HF_TOKEN = os.getenv('HF_TOKEN')
HF_API_URL = 'https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2'

def encode(texts: list) -> list:
    headers = {'Authorization': f'Bearer {HF_TOKEN}'}
    response = requests.post(
        HF_API_URL,
        headers=headers,
        json={'inputs': texts, 'options': {'wait_for_model': True}}
    )
    embeddings = np.array(response.json())
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / norms).tolist()

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_rag_contexts(matches: list) -> str:
    formatted_results = []
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Chunk: {x['metadata']['chunk']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
        )
        formatted_results.append(text)
    return '\n---\n'.join(formatted_results)

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool('rag_search')
def rag_search(query: str) -> str:
    '''Finds specialist information on AI using a natural language query.'''
    xq = encode([query])
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    return format_rag_contexts(xc['matches'])

@tool('rag_search_filter')
def rag_search_filter(query: str, arxiv_id: str) -> str:
    '''Finds information from ArXiv using a natural language query and a specific ArXiv ID.'''
    xq = encode([query])
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={'arxiv_id': arxiv_id})
    return format_rag_contexts(xc['matches'])

if __name__ == '__main__':
    output = rag_search.invoke(input={'query': 'transformer attention mechanism'})
    print(output)

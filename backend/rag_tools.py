import os

import paths  # noqa: F401
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

# ── Pinecone & encoder setup ──────────────────────────────────────────────────

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('langgraph-research-agent')
encoder = SentenceTransformer('all-MiniLM-L6-v2')


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_rag_contexts(matches: list) -> str:
    '''Formats the retrieved context matches into a readable string format.

    Args:
        matches (list): A list of matched documents with metadata.

    Returns:
        str: A formatted string of document titles, chunks, and ArXiv IDs.
    '''
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
    '''Finds specialist information on AI using a natural language query.

    Args:
        query (str): The search query in natural language.

    Returns:
        str: A formatted string of relevant document contexts.
    '''
    xq = encoder.encode([query], normalize_embeddings=True).tolist()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    return format_rag_contexts(xc['matches'])


@tool('rag_search_filter')
def rag_search_filter(query: str, arxiv_id: str) -> str:
    '''Finds information from the ArXiv database using a natural language query and a specific ArXiv ID.

    Args:
        query (str): The search query in natural language.
        arxiv_id (str): The ArXiv ID of the specific paper to filter by.

    Returns:
        str: A formatted string of relevant document contexts.
    '''
    xq = encoder.encode([query], normalize_embeddings=True).tolist()
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={'arxiv_id': arxiv_id})
    return format_rag_contexts(xc['matches'])


if __name__ == '__main__':
    output = rag_search.invoke(input={'query': 'transformer attention mechanism'})
    print(output)

    


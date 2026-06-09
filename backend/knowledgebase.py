import json
import os

import paths  # noqa: F401
import pandas as pd
from sentence_transformers import SentenceTransformer

from chunker import expand_df
from pdf import download_pdfs, extract_from_arxiv
from paths import FILES_DIR

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DATASET_PATH = os.path.join(FILES_DIR, 'arxiv_dataset.json')
KNOWLEDGE_BASE_PATH = os.path.join(FILES_DIR, 'knowledge_base.pkl')


# ── Model initialisation ──────────────────────────────────────────────────────

def get_embedding_model():
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (downloads once on first use)...")
    return SentenceTransformer(EMBEDDING_MODEL)


# ── Embeddings ────────────────────────────────────────────────────────────────

def create_embedding(model, text):
    return model.encode(text, normalize_embeddings=True).tolist()


def create_embeddings(model, texts, batch_size=32):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True).tolist()


# ── Paper loading / preparation ───────────────────────────────────────────────

def load_papers_df(json_path=DATASET_PATH):
    with open(json_path, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))

    df["pdf_file_name"] = df["pdf_link"].apply(
        lambda link: os.path.join(FILES_DIR, link.rstrip("/").split("/")[-1] + ".pdf")
    )
    return df


def prepare_papers(max_results=20, search_query="cat:cs.AI", refresh=False):
    if refresh or not os.path.isfile(DATASET_PATH):
        df = extract_from_arxiv(search_query=search_query, max_results=max_results)
        df = download_pdfs(df)
        return df

    df = load_papers_df()
    missing_pdfs = df["pdf_file_name"].isna() | ~df["pdf_file_name"].apply(os.path.isfile)
    if missing_pdfs.any():
        df = download_pdfs(df)
    return df


def build_chunked_df(df):
    df = df[df["pdf_file_name"].notna() & df["pdf_file_name"].apply(os.path.isfile)].copy()
    df = df.reset_index(drop=True)
    return expand_df(df)


# ── Embedding pipeline ────────────────────────────────────────────────────────

def embed_chunks(model, chunked_df):
    texts = chunked_df["chunk"].tolist()
    print(f"Creating embeddings for {len(texts)} chunks...")
    chunked_df = chunked_df.copy()
    chunked_df["embedding"] = create_embeddings(model, texts)
    return chunked_df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_knowledge_base(knowledge_df, output_path=KNOWLEDGE_BASE_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    knowledge_df.to_pickle(output_path)
    print(f"Knowledge base saved to {output_path}")


def load_knowledge_base(path=KNOWLEDGE_BASE_PATH):
    return pd.read_pickle(path)


# ── High-level builder ────────────────────────────────────────────────────────

def build_knowledge_base(max_results=20, refresh=False, save=True):
    if not refresh and os.path.isfile(KNOWLEDGE_BASE_PATH):
        print(f"Loading existing knowledge base from {KNOWLEDGE_BASE_PATH}...")
        return load_knowledge_base()
    df = prepare_papers(max_results=max_results, refresh=refresh)
    chunked_df = build_chunked_df(df)

    if chunked_df.empty:
        raise ValueError("No chunks were created. Download PDFs first with pdf.py.")

    model = get_embedding_model()
    knowledge_df = embed_chunks(model, chunked_df)

    if save:
        save_knowledge_base(knowledge_df)

    return knowledge_df


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = get_embedding_model()

    test_text = "hello hallo hola salut"
    test_embedding = create_embedding(model, test_text)
    print(f"Model          : {EMBEDDING_MODEL}")
    print(f"Embedding dims : {len(test_embedding)}")

    knowledge_df = build_knowledge_base(refresh=False)
    print(knowledge_df[["id", "title", "arxiv_id"]].head())
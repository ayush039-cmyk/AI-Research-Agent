import json
import os

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from paths import FILES_DIR

def load_and_chunk_pdf(pdf_file_name, chunk_size=512):
    """
    Processes PDF documents into semantically meaningful text chunks for AI analysis.
    
    This function handles the extraction of text from PDFs and implements intelligent
    text splitting to preserve context and meaning. It uses LangChain's document 
    processing capabilities for robust PDF handling.

    Parameters
    ----------
    pdf_file_name : str
        Path to the target PDF file for processing
    chunk_size : int, optional
        Maximum character length for each text chunk (default: 512)
        Chosen to optimize for transformer model context windows

    Returns
    -------
    List[Document]
        Collection of LangChain Document objects, each containing:
        - Chunk text content
        - Metadata from the original PDF
        - Page numbers and positions
    
    Notes
    -----
    The chunking process includes a 64-character overlap between segments
    to maintain context and prevent splitting of important phrases or concepts.
    """
    print(f'Loading and splitting into chunks: {pdf_file_name}')
    reader = PdfReader(pdf_file_name)
    documents = [
        Document(page_content=page.extract_text(), metadata={"source": pdf_file_name, "page": i})
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=64)
    return text_splitter.split_documents(documents)


def expand_df(df):
    """
    Expands a DataFrame of PDF metadata into a structured collection of text chunks
    with preserved relationships and context.
    
    This function processes each PDF document into chunks while maintaining the
    relationships between segments and their associated metadata. It creates a
    traceable chain of text segments that preserves the document's logical flow.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing document metadata with required columns:
        - pdf_file_name: Path to PDF file
        - arxiv_id: Unique identifier for the paper
        - title: Paper title
        - summary: Paper abstract
        - authors: List of authors
        - url: Source URL

    Returns
    -------
    pandas.DataFrame
        Expanded DataFrame where each row represents a document chunk with:
        - Unique chunk identifiers
        - Complete paper metadata
        - Chunk content
        - References to adjacent chunks (previous/next)
    
    Notes
    -----
    The expansion process creates bidirectional links between chunks,
    enabling reconstruction of the original document flow and context-aware
    processing in the AI pipeline.
    """
    expanded_rows = []

    for idx, row in df.iterrows():
        try:
            chunks = load_and_chunk_pdf(row['pdf_file_name'])
        except Exception as e:
            print(f"Error processing {row['pdf_file_name']}: {e}")
            continue

        for i, chunk in enumerate(chunks):
            expanded_rows.append({
                'id': f"{row['arxiv_id']}#{i}",
                'title': row['title'],
                'summary': row['summary'],
                'authors': row['authors'],
                'arxiv_id': row['arxiv_id'],
                'url': row['url'],
                'chunk': chunk.page_content,
                'prechunk_id': '' if i == 0 else f"{row['arxiv_id']}#{i-1}",
                'postchunk_id': '' if i == len(chunks) - 1 else f"{row['arxiv_id']}#{i+1}"
            })

    return pd.DataFrame(expanded_rows)


if __name__ == '__main__':
    dataset_path = os.path.join(FILES_DIR, 'arxiv_dataset.json')
    with open(dataset_path, encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['pdf_file_name'] = df['pdf_link'].apply(
        lambda link: os.path.join(FILES_DIR, link.rstrip('/').split('/')[-1] + '.pdf')
    )
    df = df[df['pdf_file_name'].apply(os.path.isfile)].reset_index(drop=True)

    expanded_df = expand_df(df)
    print(expanded_df.head())
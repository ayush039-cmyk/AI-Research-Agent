# AI BASED RESEARCH ASSISTANT AGENT 

## An intelligent research assistant that ingests PDF documents, indexes them in a Pinecone vector database, and uses a Groq-powered agentic pipeline to answer research queries with deep accuracy. Features ArXiv fetching, web search fallback, RAG-based retrieval, and a full-stack web frontend with authentication.

## How it works:
```mermaid
flowchart TD
    A([PDF Sources]) --> B["pdf.py - Download & Parse PDFs"]
    B --> C["chunker.py - Split into Semantic Chunks"]
    C --> D["knowledgebase.py - Generate Embeddings"]
    D --> E["vector_store.py - Upsert to Pinecone Vector DB"]
    E --> F["agent.py + graph.py - Groq LLM + LangGraph Agent"]
    F --> G["rag_tools.py - RAG Search"]
    F --> H["web_search.py / fetch.py - Web Search + ArXiv"]
    G --> I["research_service.py - Unified Response"]
    H --> I
    I --> J["app.py - Flask REST API"]
    J --> K(["Frontend - chat.html"])
```


## Features
PDF Ingestion — Downloads and parses PDFs from URLs or local files

Semantic Chunking — Splits documents into meaningful overlapping chunks for better retrieval

Vector Embeddings — Converts chunks into dense embeddings and stores them in Pinecone

RAG Search — Retrieves the most relevant chunks based on semantic similarity to a query

ArXiv Integration — Directly fetches academic papers from ArXiv by query or paper ID

Web Search Fallback — Falls back to live web search when the knowledge base lacks relevant data

Groq LLM Agent — Fast, accurate responses powered by Groq's inference API 

LangGraph Orchestration — Structured agent execution graph for reliable multi-step reasoning

Auth System — JWT-based login/signup with protected routes

Chat UI — Clean, responsive multi-turn chat interface

## Agent Execution Flow 
```mermaid
flowchart TD
    A([User Query]) --> B["run_navigator(state)"]
    B --> C{Tool calls found?}
    C -->|No| D["fallback_final_answer()"]
    C -->|Yes| E["Extract tool_name + tool_args"]
    E --> F["tool_str_to_func[tool_name](tool_args)"]
    F --> G["rag_search / rag_search_filter"]
    F --> H["fetch_arxiv"]
    F --> I["web_search"]
    G --> J["Update intermediate_steps"]
    H --> J
    I --> J
    J --> K{final_answer called?}
    K -->|No| B
    K -->|Yes| L(["Return Final Answer to User"])
```

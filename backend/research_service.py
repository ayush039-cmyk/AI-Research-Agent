import json
import os
import re

import paths  # noqa: F401
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from fetch import fetch_arxiv
from rag_tools import rag_search, rag_search_filter
from web_search import web_search

ARXIV_ID_PATTERN = re.compile(r'\b(\d{4}\.\d{4,5})(?:v\d+)?\b')
FAILED_TOOL_PATTERN = re.compile(
    r'<function=(\w+)(\{.*?\})</function>',
    re.DOTALL,
)


def extract_arxiv_id(query: str) -> str | None:
    match = ARXIV_ID_PATTERN.search(query)
    return match.group(1) if match else None


def _run_tool(tool, payload: dict) -> str:
    try:
        result = tool.invoke(payload)
        text = str(result).strip()
        return text if text else 'No results returned.'
    except Exception as exc:
        return f'Tool error: {exc}'


def collect_research(query: str) -> tuple[str, list[str]]:
    sections: list[str] = []
    sources: list[str] = []

    rag_output = _run_tool(rag_search, {'query': query})
    if rag_output and 'No results' not in rag_output:
        sections.append(f'## RAG search (ArXiv knowledge base)\n{rag_output[:6000]}')
        sources.append('Pinecone RAG index (embedded ArXiv papers)')

    arxiv_id = extract_arxiv_id(query)
    if arxiv_id:
        abstract = _run_tool(fetch_arxiv, {'arxiv_id': arxiv_id})
        sections.append(f'## ArXiv abstract ({arxiv_id})\n{abstract[:4000]}')
        sources.append(f'https://arxiv.org/abs/{arxiv_id}')

        filtered = _run_tool(rag_search_filter, {'query': query, 'arxiv_id': arxiv_id})
        if filtered and 'No results' not in filtered:
            sections.append(f'## RAG chunks for paper {arxiv_id}\n{filtered[:4000]}')
            if f'https://arxiv.org/abs/{arxiv_id}' not in sources:
                sources.append(f'https://arxiv.org/abs/{arxiv_id}')

    web_output = _run_tool(web_search, {'query': query})
    if web_output and 'No results' not in web_output:
        sections.append(f'## Web search\n{web_output[:5000]}')
        sources.append('Google search (SerpAPI)')

    if not sections:
        sections.append('No external research results were retrieved. Answer using general knowledge carefully.')

    return '\n\n'.join(sections), sources


def _to_langchain_messages(chat_history: list[BaseMessage] | None) -> list:
    if not chat_history:
        return []
    return list(chat_history[-6:])


def synthesize_report(
    query: str,
    research_context: str,
    sources: list[str],
    chat_history: list[BaseMessage] | None = None,
) -> str:
    llm = ChatGroq(
        model='llama-3.3-70b-versatile',
        api_key=os.environ['GROQ_API_KEY'],
        temperature=0.2,
    )

    system = SystemMessage(content=(
        'You are an AI research agent. Write a clear, detailed research report for the user '
        'using ONLY the collected research below. Include concrete facts, paper titles, ArXiv IDs, '
        'and URLs where available. Structure your answer with these sections:\n'
        'INTRODUCTION\nRESEARCH STEPS\nREPORT\nCONCLUSION\nSOURCES\n\n'
        'The REPORT section must be 3-4 substantive paragraphs. Do not leave sections empty. '
        'Do not say you lack information if research data is provided.'
    ))

    user = HumanMessage(content=(
        f'User question:\n{query}\n\n'
        f'Collected research:\n{research_context}\n\n'
        f'Source list hint:\n' + '\n'.join(f'- {s}' for s in sources)
    ))

    messages = [system, *_to_langchain_messages(chat_history), user]
    response = llm.invoke(messages)
    content = (response.content or '').strip()

    if content:
        return content

    return _format_fallback_report(query, research_context, sources)


def _format_fallback_report(query: str, research_context: str, sources: list[str]) -> str:
    intro = f'Research summary for: {query}'
    steps = '- Queried RAG knowledge base\n- Ran web search'
    if extract_arxiv_id(query):
        steps += '\n- Fetched ArXiv abstract and filtered RAG chunks'
    body = research_context[:8000]
    conclusion = 'See the collected research above for details and references.'
    source_lines = '\n'.join(f'- {s}' for s in sources) or '- Research tools'

    return f"""INTRODUCTION
------------
{intro}

RESEARCH STEPS
--------------
{steps}

REPORT
------
{body}

CONCLUSION
----------
{conclusion}

SOURCES
-------
{source_lines}
"""


def run_research(query: str, chat_history: list[BaseMessage] | None = None) -> str:
    research_context, sources = collect_research(query.strip())
    return synthesize_report(query.strip(), research_context, sources, chat_history)

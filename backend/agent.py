import os

import paths  # noqa: F401
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolCall, ToolMessage
from langchain_groq import ChatGroq

from rag_tools import rag_search, rag_search_filter
from fetch import fetch_arxiv
from web_search import web_search
from final_answer import final_answer

# ── Prompt ────────────────────────────────────────────────────────────────────

system_prompt = (
    '''You are the Navigator research agent.
    Choose tools to gather information before answering.

    Rules:
    1. Use rag_search for AI/research topics.
    2. Use fetch_arxiv + rag_search_filter when an ArXiv ID is mentioned.
    3. Use web_search for broader or recent context.
    4. Use at least TWO different research tools before final_answer.
    5. Do NOT repeat the same tool with the same arguments.
    6. When enough evidence is in the scratchpad, call final_answer with a full report
       (introduction, research_steps, main_body, conclusion, sources).'''
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
    ('assistant', 'scratchpad: {scratchpad}'),
])

# ── LLM (Groq) ────────────────────────────────────────────────────────────────

llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.environ['GROQ_API_KEY'],
    temperature=0
)

# ── Tools ─────────────────────────────────────────────────────────────────────

tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer
]

# ── Scratchpad ────────────────────────────────────────────────────────────────

def create_scratchpad(intermediate_steps: list[ToolCall]) -> str:
    research_steps = []

    for action in intermediate_steps:
        if action.log != 'TBD':
            research_steps.append(
                f'Tool: {action.tool}, input: {action.tool_input}\n'
                f'Output: {action.log}'
            )

    return '\n---\n'.join(research_steps)

# ── Navigator chain ───────────────────────────────────────────────────────────

navigator = (
    {
        'input': lambda x: x['input'],
        'chat_history': lambda x: x['chat_history'],
        'scratchpad': lambda x: create_scratchpad(intermediate_steps=x['intermediate_steps']),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice='auto')
)
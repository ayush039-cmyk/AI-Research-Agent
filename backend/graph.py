import json
import operator
import re
from typing import Annotated, List, TypedDict

import paths  # noqa: F401
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from agent import navigator
from rag_tools import rag_search, rag_search_filter
from fetch import fetch_arxiv
from web_search import web_search
from final_answer import final_answer


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    '''Represents the state of the research agent.'''
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[AgentAction], operator.add]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _extract_failed_final_answer(error: Exception) -> dict | None:
    message = str(error)
    if 'tool_use_failed' not in message:
        return None

    match = re.search(r'<function=final_answer>(\{.*?\})</function>', message, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _fallback_final_answer(content: str, intermediate_steps: list) -> dict:
    sources = []
    for step in intermediate_steps:
        if step.log and step.log != 'TBD':
            if step.tool == 'fetch_arxiv':
                sources.append(f"ArXiv {step.tool_input.get('arxiv_id', '')}")
            elif step.tool in {'rag_search', 'rag_search_filter'}:
                sources.append('Vector knowledge base (RAG)')
            elif step.tool == 'web_search':
                sources.append('Web search results')

    body = content.strip() or 'Research completed using the collected sources.'
    return {
        'introduction': body.split('\n', 1)[0][:500],
        'research_steps': [f'Used {step.tool}' for step in intermediate_steps if step.log != 'TBD'],
        'main_body': body,
        'conclusion': 'See main report above.',
        'sources': sources or ['Research agent tools'],
    }


def run_navigator(state: dict) -> dict:
    print('run_navigator')
    print(f'intermediate_steps: {state["intermediate_steps"]}')

    out = None
    last_error = None

    for attempt in range(3):
        try:
            out = navigator.invoke(state)
            break
        except Exception as exc:
            last_error = exc
            recovered = _extract_failed_final_answer(exc)
            if recovered:
                action_out = AgentAction(tool='final_answer', tool_input=recovered, log='TBD')
                return {'intermediate_steps': [action_out]}
            print(f'Navigator attempt {attempt + 1} failed: {exc}')

    if out is None:
        fallback = _fallback_final_answer('', state['intermediate_steps'])
        action_out = AgentAction(tool='final_answer', tool_input=fallback, log='TBD')
        return {'intermediate_steps': [action_out]}

    if not out.tool_calls:
        action_out = AgentAction(
            tool='final_answer',
            tool_input=_fallback_final_answer(out.content or '', state['intermediate_steps']),
            log='TBD',
        )
    else:
        tool_name = out.tool_calls[0]['name']
        tool_args = out.tool_calls[0]['args']
        action_out = AgentAction(tool=tool_name, tool_input=tool_args, log='TBD')

    return {'intermediate_steps': [action_out]}


tool_str_to_func = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_arxiv': fetch_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}


def run_tool(state: dict) -> dict:
    '''Executes the tool specified in the last intermediate step.'''
    tool_name = state['intermediate_steps'][-1].tool
    tool_args = state['intermediate_steps'][-1].tool_input

    print(f'{tool_name}.invoke(input={tool_args})')

    out = tool_str_to_func[tool_name].invoke(input=tool_args)

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )

    return {'intermediate_steps': [action_out]}


# ── Router ────────────────────────────────────────────────────────────────────

def router(state: dict) -> str:
    '''Routes to the next tool based on the navigator's decision.'''
    if isinstance(state['intermediate_steps'], list):
        return state['intermediate_steps'][-1].tool
    print('Router invalid format')
    return 'final_answer'


# ── Graph ─────────────────────────────────────────────────────────────────────

tools = [rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer]

graph = StateGraph(AgentState)

graph.add_node('navigator', run_navigator)
graph.add_node('rag_search_filter', run_tool)
graph.add_node('rag_search', run_tool)
graph.add_node('fetch_arxiv', run_tool)
graph.add_node('web_search', run_tool)
graph.add_node('final_answer', run_tool)

graph.set_entry_point('navigator')
graph.add_conditional_edges(
    source='navigator',
    path=router,
    path_map={tool_obj.name: tool_obj.name for tool_obj in tools},
)

for tool_obj in tools:
    if tool_obj.name != 'final_answer':
        graph.add_edge(tool_obj.name, 'navigator')

graph.add_edge('final_answer', END)

runnable = graph.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

def build_report(output: dict) -> str:
    '''Builds a formatted report based on the navigator's output.

    Args:
        output (dict): A dictionary containing the various sections of the report.

    Returns:
        str: A formatted string containing the full research report.
    '''
    research_steps = output['research_steps']
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])

    sources = output['sources']
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])

    return f"""
INTRODUCTION
------------
{output['introduction']}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output['main_body']}

CONCLUSION
----------
{output['conclusion']}

SOURCES
-------
{sources}
"""


def run_research_graph(query: str, chat_history: List[BaseMessage] | None = None) -> str:
    '''Run the LangGraph agent loop (legacy). Prefer research_service.run_research.'''
    output = runnable.invoke({
        'input': query,
        'chat_history': chat_history or [],
        'intermediate_steps': [],
    })
    return build_report(output=output['intermediate_steps'][-1].tool_input)


def run_research(query: str, chat_history: List[BaseMessage] | None = None) -> str:
    '''Run the reliable research pipeline and return a formatted report.'''
    from research_service import run_research as run_research_pipeline

    return run_research_pipeline(query, chat_history=chat_history)


if __name__ == '__main__':
    report = run_research('Create a summary about this ArXiv paper with the ID 2502.20384')
    print(report)
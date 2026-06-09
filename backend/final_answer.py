from langchain_core.tools import tool


@tool
def final_answer(
    introduction: str,
    research_steps: str | list,
    main_body: str,
    conclusion: str,
    sources: str | list
) -> str:
    '''Returns a natural language response in the form of a research report.

    Args:
        introduction (str): A short paragraph introducing the user's question and the topic.
        research_steps (str | list): Bullet points or text explaining the steps taken for research.
        main_body (str): The bulk of the answer, 3-4 paragraphs long, providing high-quality information.
        conclusion (str): A short paragraph summarizing the findings.
        sources (str | list): A list or text providing the sources referenced during the research.

    Returns:
        str: A formatted research report string.
    '''
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])

    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])

    return (
        f"{introduction}\n\n"
        f"Research Steps:\n{research_steps}\n\n"
        f"Main Body:\n{main_body}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Sources:\n{sources}"
    )
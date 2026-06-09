import re
import requests
from langchain_core.tools import tool

abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)


@tool('fetch_arxiv')
def fetch_arxiv(arxiv_id: str) -> str:
    '''Fetches the abstract from an ArXiv paper given its ArXiv ID.

    Args:
        arxiv_id (str): The ArXiv paper ID.

    Returns:
        str: The extracted abstract text from the ArXiv paper.
    '''
    res = requests.get(f'https://arxiv.org/abs/{arxiv_id}')
    re_match = abstract_pattern.search(res.text)
    return re_match.group(1) if re_match else 'Abstract not found.'


if __name__ == '__main__':
    arxiv_id = '2502.20384'
    output = fetch_arxiv.invoke(input={'arxiv_id': arxiv_id})
    print(output)
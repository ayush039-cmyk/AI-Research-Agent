import os

import paths  # noqa: F401
from serpapi import GoogleSearch
from langchain_core.tools import tool

serpapi_params = {
    'engine': 'google',
    'api_key': os.getenv('SERPAPI_KEY')
}


@tool('web_search')
def web_search(query: str) -> str:
    '''Finds general knowledge information using a Google search.

    Args:
        query (str): The search query string.

    Returns:
        str: A formatted string of the top search results, including title, snippet, and link.
    '''
    search = GoogleSearch({
        **serpapi_params,
        'q': query,
        'num': 5
    })

    results = search.get_dict().get('organic_results', [])
    formatted_results = '\n---\n'.join(
        ['\n'.join([x['title'], x['snippet'], x['link']]) for x in results]
    )

    return formatted_results if results else 'No results found.'


if __name__ == '__main__':
    output = web_search.invoke(input={'query': 'water on mars'})
    print(output)
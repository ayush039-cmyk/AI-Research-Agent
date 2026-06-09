import json
import os
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from paths import FILES_DIR

ARXIV_NAMESPACE = '{http://www.w3.org/2005/Atom}'


def extract_from_arxiv(
    search_query='cat:cs.AI',
    max_results=100,
    json_file_path=os.path.join(FILES_DIR, 'arxiv_dataset.json'),
):
    """
    Fetches papers from the ArXiv API based on a search query, saves them as JSON, 
    and returns a pandas DataFrame.

    Args:
        search_query (str): The search query for ArXiv (default is 'cat:cs.AI').
        max_results (int): The maximum number of results to retrieve (default is 100).
        json_file_path (str): File path where JSON data will be saved.

    Returns:
        pd.DataFrame: DataFrame containing the extracted paper information.
    """
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    
    url = f'http://export.arxiv.org/api/query?search_query={search_query}&max_results={max_results}'
    response = requests.get(url)
    root = ET.fromstring(response.content)
    
    papers = []
    for entry in root.findall(f'{ARXIV_NAMESPACE}entry'):
        title = entry.find(f'{ARXIV_NAMESPACE}title').text.strip()
        summary = entry.find(f'{ARXIV_NAMESPACE}summary').text.strip()
        author_elements = entry.findall(f'{ARXIV_NAMESPACE}author')
        authors = [author.find(f'{ARXIV_NAMESPACE}name').text for author in author_elements]
        paper_url = entry.find(f'{ARXIV_NAMESPACE}id').text
        arxiv_id = paper_url.split('/')[-1]
        pdf_link = next((link.attrib['href'] for link in entry.findall(f'{ARXIV_NAMESPACE}link')
                         if link.attrib.get('title') == 'pdf'), None)
        papers.append({
            'title': title,
            'summary': summary,
            'authors': authors,
            'arxiv_id': arxiv_id,
            'url': paper_url,
            'pdf_link': pdf_link
        })

    df = pd.DataFrame(papers)

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)
        print(f'Data saved to {json_file_path} ...')

    return df


def download_pdfs(df, download_folder=FILES_DIR):
    """
        Retrieves and stores academic papers from ArXiv as PDF files using URLs provided in a DataFrame.
        This function processes each paper systematically, handling potential download failures gracefully,
        and maintains a record of file locations for subsequent processing.
    
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing paper metadata with a required 'pdf_link' column 
            storing ArXiv PDF URLs.
        download_folder : str, optional
            Target directory for PDF storage (default: 'files'). Will be created if 
            it doesn't exist.
    
        Returns
        -------
        pandas.DataFrame
            Enhanced DataFrame with an additional 'pdf_file_name' column containing:
            - Full file paths for successfully downloaded PDFs
            - None values for failed downloads
            
        Notes
        -----
        The function implements error handling for network issues and invalid URLs,
        ensuring the process continues even if individual downloads fail.
      """

    os.makedirs(download_folder, exist_ok=True)
    pdf_file_names = []

    for index, row in df.iterrows():
        pdf_link = row['pdf_link']
        try:
            response = requests.get(pdf_link, timeout=10)
            response.raise_for_status()

            file_name = os.path.join(download_folder, pdf_link.split('/')[-1] + '.pdf')

            with open(file_name, 'wb') as f:
                f.write(response.content)

            pdf_file_names.append(file_name)
            print(f'[{index+1}/{len(df)}] Downloaded: {file_name}')

        except requests.exceptions.RequestException as e:
            print(f'[{index+1}/{len(df)}] Failed: {e}')
            pdf_file_names.append(None)

    df['pdf_file_name'] = pdf_file_names
    return df


if __name__ == '__main__':
    # Step 1: Fetch papers from ArXiv (creates files/arxiv_dataset.json)
    df = extract_from_arxiv(max_results=20)

    # Step 2: Download the PDFs
    df = download_pdfs(df)

    print(df.head())
import requests
from typing import List
from bs4 import BeautifulSoup
from llama_index.core import Document

def create_arxiv_endpoint(tags, max_results):
    return f"http://export.arxiv.org/api/query?search_query={tags}&start=0&max_results={max_results}&sortBy=relevance"


def search_research_tool(tags: str):
    MAX_RESULTS = 5
    response = requests.get(create_arxiv_endpoint(tags, MAX_RESULTS))
    soup = BeautifulSoup(response.text, "lxml-xml")

    summaries = []
    titles = []
    for entry in soup.find_all('entry'):
        summary = entry.summary.text
        summaries.append(summary)
        title = entry.title.text
        titles.append(title)

    return titles, summaries

# def update_vector_index(index, data):
#     documents = []
#     titles, summaries = data

#     for title, summary in zip(titles, summaries):
#         document = Document(text=summary, metadata = {"title": title})
#         documents.append(document)

#     index.add_documents(documents)
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
    
    # PERSIST_DIR = "/workspace/llama/agent_dir"
    # if not os.path.join(PERSIST_DIR):
    
# print(search_research_tool("transformers, attention is all you need"))
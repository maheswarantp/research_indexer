import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage, StorageContext, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import FunctionTool

from tools.search_research import search_research_tool
from tools.write_output import write_output_file

CONTEXT_QUERY_ENGINE_TEMPLATE = """Purpose: The primary role of this agent is to search the web and get information about research papers """

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm_llama3 = Ollama(model="llama3", request_timeout=60.0)

Settings.embed_model = embed_model
Settings.llm = llm_llama3


PERSIST_DIR = "/workspace/research_indexer/agent_dir"
if not os.path.exists(PERSIST_DIR):
    # index doesnt exist, create one for the documents
    print("Index not found, creating one...")
    documents = SimpleDirectoryReader("./data").load_data()
    vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    vector_index.storage_context.persist(PERSIST_DIR)
else:
    # index already exists, load that
    print(f"Index found, loading from directory: {PERSIST_DIR}")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    vector_index = load_index_from_storage(storage_context)

query_engine = vector_index.as_query_engine(llm=llm_llama3)

def update_vector_index(titles, summaries):
    documents = []
    # titles, summaries = data

    for title, summary in zip(titles, summaries):
        document = Document(text=summary, metadata = {"title": title})
        documents.append(document)


    [vector_index.insert(document) for document in documents]
    
    vector_index.refresh(documents)
    vector_index.storage_context.persist(persist_dir=PERSIST_DIR)

tools = [
    FunctionTool.from_defaults(
        fn=search_research_tool,
        name="search_research_tool",
        description="This tool can search the internet for getting summaries on research paper, will take tags as a variable and give top 5 results"
    ),
    FunctionTool.from_defaults(
        fn=update_vector_index,
        name="update_vector_index",
        description="Updates the existing vector index with new data information which can be obtained from a variety of sources, primarily the internet"
    ),
    QueryEngineTool(
        query_engine = query_engine,
        metadata = ToolMetadata(
            name="research_query_engine",
            description="this tool can query the vector index about information about research papers"
        )
    ),
    FunctionTool.from_defaults(
        fn=write_output_file,
        name="output_tool",
        description="this tool writes any information passed as a variable to a file present in output folder"
    )
]

agent = ReActAgent.from_tools(tools, llm=llm_llama3, verbose=True, context=CONTEXT_QUERY_ENGINE_TEMPLATE)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0

    while retries < 3:
        try:
            result = agent.query(prompt)
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}: ", e)
    
    if retries >= 3:
        print("Unable to process request, try again...")
        continue
    
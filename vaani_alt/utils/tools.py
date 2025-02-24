import os
import tempfile
import replicate
from typing import Dict, List, Optional, Union
import base64
from io import BytesIO
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize Tavily Search Tool
tavily_search_tool = TavilySearchResults(max_results=5)

def get_file_extension(file_path: str) -> str:
    """Get the file extension from a file path."""
    return Path(file_path).suffix.lower()

def process_document(file_path: str, file_content: Optional[Union[bytes, str]] = None) -> List[Document]:
    """Process a document based on its extension."""
    if not file_content:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
    
    ext = get_file_extension(file_path)
    documents = []
    
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
        if file_content:
            if isinstance(file_content, str):
                temp_file.write(file_content.encode('utf-8'))
            else:
                temp_file.write(file_content)
        else:
            with open(file_path, 'rb') as f:
                temp_file.write(f.read())
        temp_file_path = temp_file.name
    
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(temp_file_path)
            documents = loader.load()
        elif ext == '.txt' or not ext:
            loader = TextLoader(temp_file_path)
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

def create_qdrant_from_documents(documents: List[Document], collection_name: str = "vaani_docs") -> Qdrant:
    """Create a Qdrant vector store from documents."""
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Get Qdrant credentials from environment variables
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    # Create Qdrant instance
    vector_store = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        force_recreate=True  # Be careful with this in production
    )
    
    return vector_store

def query_qdrant(query: str, collection_name: str = "vaani_docs", k: int = 5) -> List[Document]:
    """Query Qdrant vector store."""
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Get Qdrant credentials from environment variables
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    
    # Connect to existing Qdrant collection
    vector_store = Qdrant(
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name
    )
    
    # Query for similar documents
    docs = vector_store.similarity_search(query, k=k)
    return docs

def generate_image(prompt: str) -> List[str]:
    """Generate an image using Replicate's Flux Schnell model."""
    input_data = {
        "prompt": prompt
    }
    
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input=input_data
    )
    
    # Save images to temporary files and return their paths
    image_paths = []
    for index, item in enumerate(output):
        temp_image_path = f"/tmp/output_{index}.webp"
        with open(temp_image_path, "wb") as file:
            file.write(item.read())
        image_paths.append(temp_image_path)
    
    return image_paths

@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return tavily_search_tool.invoke({"query": query})
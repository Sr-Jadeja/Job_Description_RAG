import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DOCS_DIR   = "docs"
INDEX_PATH = "rag/faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def build_index():
    """Load all .txt files from docs/, chunk and embed into FAISS index"""
    print("Loading documents...")
    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    docs   = loader.load()
    print(f"Loaded {len(docs)} documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks   = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings  = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}/")
    return vectorstore


def load_vectorstore():
    """Load saved FAISS index, build it first if not found"""
    if not os.path.exists(INDEX_PATH):
        print("Index not found — building now...")
        return build_index()
    embeddings = get_embeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)


def retrieve(query: str, k: int = 4) -> str:
    """Return top-k relevant chunks as a single string"""
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant content found."
    return "\n\n---\n\n".join([d.page_content for d in docs])

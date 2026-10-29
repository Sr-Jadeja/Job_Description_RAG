"""
Run this script whenever you add new .txt files to the docs/ folder.
It rebuilds the FAISS vector index from scratch.

Usage:
    python ingest.py
"""

import shutil
import os
from rag.rag_pipeline import build_index, INDEX_PATH

if os.path.exists(INDEX_PATH):
    shutil.rmtree(INDEX_PATH)
    print(f"Deleted old index at {INDEX_PATH}/")

build_index()
print("Done. You can now run the app.")

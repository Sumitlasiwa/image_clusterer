

"""
chroma_manager.py

Handles initialization and management of ChromaDB database.
Used to store and retrieve face embeddings for clustering/search.
"""

import os
import shutil
import chromadb
from chromadb.config import Settings
from pathlib import Path

def init_chroma(path: str):
    """
    Initializes a new ChromaDB database at the given path.

    Args:
        path: Path to store ChromaDB
    
    Returns:
        client: Chroma PersistentClient
    """
    RESET = False
    path = Path(path)
    if path.exists() and RESET:
        shutil.rmtree(path)

    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(
            allow_reset=True,
            anonymized_telemetry=False,
            persist_directory= str(path)
        )
    )
    return client

def get_collection(client):
    """
    Creates a new collection or resets existing one.

    Args:
        client: ChromaDB client
    
    Returns:
        collection: Collection for face embeddings
    """
    if "face_embeddings" in client.list_collections():
        client.delete_collection("face_embeddings")
    return client.get_or_create_collection(name="face_embeddings", metadata={"hnsw:space": "cosine"})

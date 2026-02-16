"""
J.A.S.O.N. Digital Consciousness - Memory Management using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from langchain_core.tools import Tool
from typing import List, Dict, Any
import json
import os
import time

class MemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./jason_memory")
        self.collection = self.client.get_or_create_collection(name="jason_knowledge")

        # Create tools
        self.recall_tool = Tool(
            name="Recall",
            func=self.recall,
            description="Recall relevant information from memory based on a query"
        )

        self.archive_tool = Tool(
            name="Archive",
            func=self.archive,
            description="Archive new information or task results to memory"
        )

    def recall(self, query: str) -> str:
        """Recall relevant information from memory"""
        results = self.collection.query(
            query_texts=[query],
            n_results=5
        )

        if results['documents']:
            memories = []
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                memories.append(f"Memory: {doc}\nContext: {metadata}")
            return "\n\n".join(memories)
        else:
            return "No relevant memories found."

    def archive(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Archive information to memory"""
        if metadata is None:
            metadata = {"timestamp": str(time.time()), "type": "general"}

        # Generate unique ID
        import uuid
        doc_id = str(uuid.uuid4())

        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

        return f"Archived: {content[:100]}..."

    def get_relevant_context(self, task: str) -> str:
        """Get context for a task before execution"""
        return self.recall(f"relevant to: {task}")

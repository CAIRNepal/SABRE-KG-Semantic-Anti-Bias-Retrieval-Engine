#!/usr/bin/env python3
"""
Vector Database Module for Bias Mitigation
Implements embedding-based retrieval using Chroma DB
"""

from .simple_embedding_retriever import SimpleEmbeddingBiasRetriever

# Optional: ChromaDB retriever (import-safe if Chroma not installed)
try:
    from .chroma_embedding_retriever import ChromaEmbeddingBiasRetriever
    _CHROMA_AVAILABLE = True
except Exception:
    ChromaEmbeddingBiasRetriever = None  # type: ignore
    _CHROMA_AVAILABLE = False

__all__ = ['SimpleEmbeddingBiasRetriever']

if _CHROMA_AVAILABLE:
    __all__.append('ChromaEmbeddingBiasRetriever')

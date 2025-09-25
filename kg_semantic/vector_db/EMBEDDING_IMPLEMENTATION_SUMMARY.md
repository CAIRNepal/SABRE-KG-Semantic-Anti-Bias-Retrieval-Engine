# Embedding Retrieval Implementation (Active Modules Only)

## Overview

This project now uses two embedding-based retrievers against `custom_kg/enhanced_linkml_data.yaml`:
- Simple in-memory retriever (no DB)
- ChromaDB-backed retriever (persistent vector store)

Archived modules (indexer, legacy vector retrievers, tests) have been removed from the active surface.

## Active Components

- `kg_semantic/vector_db/simple_embedding_retriever.py`
  - In-memory NumPy embeddings, cosine similarity via scikit-learn
  - Caches embeddings with pickle for fast startup
  - Dependencies: `sentence-transformers`, `numpy`, `scikit-learn`, `pyyaml`

- `kg_semantic/vector_db/chroma_embedding_retriever.py`
  - ChromaDB persistent collection with metadata filters
  - Uses Sentence Transformers for text embedding
  - Dependencies: above + `chromadb` stack (opentelemetry, onnxruntime, etc.)

- `kg_semantic/vector_db/__init__.py`
  - Exposes: `SimpleEmbeddingBiasRetriever`, `ChromaEmbeddingBiasRetriever`
  - Chroma imports are optional (guarded); simple retriever always available

## Data Source

- `custom_kg/enhanced_linkml_data.yaml` (≈54 person entries)
- Fields are converted into rich, searchable text and metadata

## Usage

### Toggle in notebook (experiment_1.ipynb)
```python
from kg_semantic.vector_db import SimpleEmbeddingBiasRetriever, ChromaEmbeddingBiasRetriever

USE_CHROMADB = True  # False => use SimpleEmbeddingBiasRetriever

if USE_CHROMADB:
    embedding_retriever = ChromaEmbeddingBiasRetriever()
else:
    embedding_retriever = SimpleEmbeddingBiasRetriever()

embedding_retriever.initialize()
examples = embedding_retriever.retrieve_counter_examples(question_data, max_results=5)
```

### Minimal script example
```python
from kg_semantic.vector_db import SimpleEmbeddingBiasRetriever

retriever = SimpleEmbeddingBiasRetriever()
retriever.initialize()
results = retriever.retrieve_counter_examples({
    'question': 'Who was the board chair?',
    'context': 'Board meeting discussed leadership changes.',
    'domain_info': {
        'stereotype_type': 'leadership_competence',
        'bias_direction': 'female_leadership_assumption',
        'context_type': 'corporate_leadership'
    }
}, max_results=5)
```

## Notes on ChromaDB

- First run will build a persistent collection at `./chroma_data`
- IDs are uniquified; filters use Chroma `where` format (e.g., `{ "gender": {"$eq": "female"} }`)
- Faster query latency once initialized; good for scaling beyond in-memory limits

## Results Snapshot (from recent run)

- Documents indexed: 54
- Gender distribution: ~65% female, ~35% male
- Common bias types: leadership_competence, technical_competence, professional_competence
- Example query times: ChromaDB ~0.15–0.25s; Simple ~2.0–2.6s (cached)

## Archived/Removed Items

The following legacy utilities were archived and are no longer part of the active API:
- `embedding_indexer.py`
- `vector_retriever.py`
- `pure_embedding_retriever.py`
- `compare_approaches.py`
- `test_embedding_approach.py`

If needed, recover from VCS history or `_archive/`.

## Recommendations

- Grow `enhanced_linkml_data.yaml` to improve coverage
- Consider alternative sentence-transformer models for domain fit
- For production-scale usage, prefer ChromaDB; for quick experiments, use Simple

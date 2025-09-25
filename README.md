
# SABRE-KG: Semantic Anti-Bias Retrieval Engine

This project presents **SABRE-KG (Semantic Anti-Bias Retrieval Engine)**, an experimental pipeline for detecting and mitigating **gender bias** in Large Language Models (LLMs) using prompt-based evaluation and Retrieval-Augmented Generation. The system supports two retrieval modes to fetch counter-stereotypical examples for bias mitigation:

- Semantic RAG over a knowledge graph (SPARQL-based)
- Embedding-based retrieval over LinkML data (Simple in-memory or ChromaDB)

## Project Overview

- **Bias Detection:** Evaluates LLM responses to identify potential gender biases using carefully crafted prompt variations from the BBQ dataset
- **Knowledge Graph Construction:** Creates structured bias mitigation knowledge graphs from WinoBias and targeted examples
- **Semantic RAG:** Implements pure semantic retrieval using SPARQL queries and ontology-based reasoning
- **Embedding RAG:** Implements embedding-based retrieval over `enhanced_linkml_data.yaml` using either a Simple in-memory retriever or a ChromaDB-backed retriever
- **Bias Mitigation:** Applies KG-augmented intervention strategies to reduce or eliminate detected gender bias
- **Multi-Model Evaluation:** Tests bias mitigation across GPT-4o, Claude, Gemini, and Mistral models

## Architecture

### Knowledge Graph Pipeline
- **LinkML Schema:** Structured data model for person entities with bias mitigation attributes
- **WinoBias Integration:** 332 counter-stereotypical examples from WinoBias dataset
- **Targeted Examples:** 7 additional examples covering missing stereotype types
- **RDF Conversion:** Turtle format with 2,382 triples for semantic querying

### Semantic Retrieval System
- **SPARQL Engine:** Query-based retrieval using semantic patterns
- **Ontology Integration:** Bias mitigation ontology with stereotype classifications

### Embedding Retrieval System
- **Simple Embedding Retriever:** In-memory sentence embeddings with cosine similarity; zero external DB dependencies.
- **ChromaDB Retriever:** Persistent vector store with metadata filters; better scalability and query latency.
- **Toggle in Notebook:** In `experiment_1.ipynb`, set `USE_CHROMADB = True/False` to switch.

### RAG Intervention Pipeline
- **Counter-Example Retrieval:** Relevant bias-challenging examples
- **Prompt Engineering:** Context-aware intervention prompts
- **Improvement Assessment:** Pre/post response comparison

## Project Structure

```
Project Folder/
├── custom_kg/                               # Knowledge base YAML files
│   ├── enhanced_linkml_data.yaml            # 54 person entities (embedding source)
│   ├── enhanced_winobias_kg.yaml            # WinoBias examples
│   └── linkml_schema.yaml                   # Data model schema
├── kg_semantic/
│   ├── data/                                # RDF data and conversion scripts
│   │   ├── enhanced_persons.ttl
│   │   ├── persons.ttl
│   │   └── convert_data.py
│   ├── ontology/                            # Bias mitigation ontology
│   │   ├── bias_mitigation_ontology.owl
│   │   └── generate_ontology.py
│   ├── integration/                         # (SPARQL) semantic retriever
│   │   └── semantic_retriever.py
│   ├── queries/                             # SPARQL query templates
│   │   ├── query_engine.py
│   │   ├── retrieval_queries.sparql
│   │   └── sample_queries.sparql
│   └── vector_db/                           # Embedding-based retrieval
│       ├── __init__.py                      # Exports retrievers
│       ├── simple_embedding_retriever.py    # In-memory retriever (active)
│       ├── chroma_embedding_retriever.py    # ChromaDB retriever (active)
│       └── EMBEDDING_IMPLEMENTATION_SUMMARY.md
├── dataset/
├── initial_LLM_results/
├── rag_results/
└── experiment_1.ipynb                       # Complete pipeline notebook
```

## How to Run

### 1. Environment Setup
```bash
# Activate your preferred Python environment (conda, venv, etc.)
# For conda users:
conda activate your_environment_name
# OR for venv users:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Knowledge Graph Generation
```bash
# Convert YAML to RDF
cd kg_semantic/data
python convert_data.py

# Generate ontology
cd ../ontology
python generate_ontology.py
```

### 3. Run Experiments
```bash
# Launch Jupyter
jupyter notebook experiment_1.ipynb

# Follow the pipeline:
# 1. LLM evaluation on BBQ dataset
# 2. Bias classification and domain analysis
# 3. Semantic RAG intervention
# 4. Results analysis and comparison
```

### 4. Embedding-based Retrieval (Simple or Chroma)

This repo includes two embedding retrievers against `custom_kg/enhanced_linkml_data.yaml`:

- SimpleEmbeddingBiasRetriever: in-memory embeddings (no vector DB)
- ChromaEmbeddingBiasRetriever: persistent vector store via ChromaDB

In `experiment_1.ipynb` you can toggle which retriever to use:

```python
from kg_semantic.vector_db import SimpleEmbeddingBiasRetriever, ChromaEmbeddingBiasRetriever

USE_CHROMADB = True  # False => use SimpleEmbeddingBiasRetriever

if USE_CHROMADB:
    embedding_retriever = ChromaEmbeddingBiasRetriever()
else:
    embedding_retriever = SimpleEmbeddingBiasRetriever()

embedding_retriever.initialize()
```

Optional: install ChromaDB dependencies if using Chroma retriever:

```bash
pip install chromadb opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc \
            importlib-resources kubernetes mmh3 onnxruntime "posthog>=2.4.0,<6.0.0"
```

## Technologies Used

- **Python 3.x** with Jupyter Notebook
- **RDF/SPARQL** for semantic knowledge representation
- **LinkML** for structured data modeling
- **OpenAI API** (GPT-4o), Anthropic (Claude), Google (Gemini), DeepSeek (Mistral)
- **Pandas, NumPy, Matplotlib** for analysis and visualization
- **rdflib** for RDF processing and SPARQL queries

## License

MIT

## How to Reference This Work

If you use or reference this project in your research, please cite it as:
```bibtex
@misc{shrestha2025sabrekg,
  title={SABRE-KG: Semantic Anti-Bias Retrieval Engine for Large Language Models},
  author={Shrestha, Abhash and Chhetri, Tek Raj},
  year={2025},
  howpublished={\url{https://github.com/CAIRNepal/SABRE-KG-Semantic-Anti-Bias-Retrieval-Engine}},
  note={Semantic KG-based bias mitigation}
}
```

*Feel free to contribute or raise issues for improvements.*

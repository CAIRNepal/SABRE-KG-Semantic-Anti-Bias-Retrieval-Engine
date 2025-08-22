
# SABRE-KG: Semantic Anti-Bias Retrieval Engine

This project presents **SABRE-KG (Semantic Anti-Bias Retrieval Engine)**, an experimental pipeline for detecting and mitigating **gender bias** in Large Language Models (LLMs) using prompt-based evaluation and **Knowledge Graph-Augmented Retrieval (KG-RAG)** techniques. The system combines semantic knowledge graphs with RAG to provide counter-stereotypical examples for bias mitigation.

## Project Overview

- **Bias Detection:** Evaluates LLM responses to identify potential gender biases using carefully crafted prompt variations from the BBQ dataset
- **Knowledge Graph Construction:** Creates structured bias mitigation knowledge graphs from WinoBias and targeted examples
- **Semantic RAG:** Implements pure semantic retrieval using SPARQL queries and ontology-based reasoning
- **Bias Mitigation:** Applies KG-augmented intervention strategies to reduce or eliminate detected gender bias
- **Multi-Model Evaluation:** Tests bias mitigation across GPT-4o, Claude, Gemini, and Mistral models

## Architecture

### Knowledge Graph Pipeline
- **LinkML Schema:** Structured data model for person entities with bias mitigation attributes
- **WinoBias Integration:** 332 counter-stereotypical examples from WinoBias dataset
- **Targeted Examples:** 7 additional examples covering missing stereotype types
- **RDF Conversion:** Turtle format with 1505+ triples for semantic querying

### Semantic Retrieval System
- **SPARQL Engine:** Query-based retrieval using semantic patterns
- **Ontology Integration:** Bias mitigation ontology with stereotype classifications

### RAG Intervention Pipeline
- **Counter-Example Retrieval:** Relevant bias-challenging examples
- **Prompt Engineering:** Context-aware intervention prompts
- **Improvement Assessment:** Pre/post response comparison

## Project Structure

```
Project Folder/
├── custom_kg/                          # Knowledge Graph YAML files
│   ├── enhanced_linkml_data.yaml      # 54 person entities
│   ├── enhanced_winobias_kg.yaml      # WinoBias examples
│   └── linkml_schema.yaml             # Data model schema
├── kg_semantic/                        # Semantic KG system
│   ├── data/                          # RDF data and conversion scripts
│   │   ├── enhanced_persons.ttl       # 1505 triples
│   │   └── convert_data.py            # YAML to RDF converter
│   ├── ontology/                      # Bias mitigation ontology
│   ├── integration/                    # Semantic retriever
│   └── queries/                       # SPARQL query templates
├── dataset/                           # BBQ dataset (1000 samples)
├── initial_LLM_results/               # Raw LLM responses
├── rag_results/                       # RAG intervention results
└── experiment_1.ipynb                 # Complete pipeline notebook
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

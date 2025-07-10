
# Gender Bias Detection and Mitigation in Large Language Models

This project presents an experimental pipeline for detecting and mitigating **gender bias** in Large Language Models (LLMs) using prompt-based evaluation and bias intervention techniques. All code, analysis, and visualizations are contained in a single Jupyter Notebook: `experiment_1.ipynb`.

##  Project Overview

- **Bias Detection:** Evaluates LLM responses to identify potential gender biases using carefully crafted prompt variations.
- **Bias Mitigation:** Applies intervention strategies to reduce or eliminate detected gender bias in model outputs.
- **Visualization & Analysis:** Provides charts and tables demonstrating the effectiveness of mitigation techniques.

##  Notebook

- `experiment_1.ipynb` — the complete pipeline from start to finish:
  - Prompt design for gender bias evaluation
  - Automated querying of language models
  - Bias scoring and visualization
  - Mitigation and post-analysis

##  How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Abhash297/RAG-Bias-Mitigation.git
   cd RAG-Bias-Mitigation
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook experiment_1.ipynb
   ```

4. Follow the steps inside the notebook to reproduce the bias detection and mitigation experiments.

##  Technologies Used

- Python 3.x
- Jupyter Notebook
- OpenAI GPT API (or equivalent)
- Pandas, Numpy, Matplotlib

##  Results

The notebook visualizes the presence of gender bias and evaluates the reduction of bias after applying mitigation strategies.

##  Future Work

- Expand to multi-dimensional bias detection (race, age, etc.)
- Integrate knowledge graphs for context-aware mitigation
- Automate benchmarking with fairness metrics

##  License

This project is licensed under the MIT License.

---

*Feel free to contribute or raise issues for improvements.*

# Cross-Domain Patent Transfer Analysis & Prediction

An academically rigorous framework for predicting **Negative Transfer** in NLP models across diverse patent domains. This project uses a supervised meta-learning approach to determine if a model trained on a source domain (e.g., Software) will perform reliably when transferred to a target domain (e.g., Electronics) before any deployment occurs.

## 🚀 Key Features

*   **Meta-Learning Engine:** A Random Forest meta-classifier that predicts "Safe" vs. "Unsafe" transfers using pre-transfer features (Embedding Similarity, Label Shift, Entropy, etc.).
*   **Statistical Rigor:** Implements **Leave-One-Domain-Out (LODO)** cross-validation with 5-seed variance analysis and **Paired T-Tests** for statistical significance.
*   **Interactive Glassmorphism Dashboard:** A premium web-based visualization tool to explore transfer risks, feature correlations, and per-pair decision reports.
*   **Explainability:** Integrated ablation studies and vocabulary mismatch analysis to explain *why* a transfer is predicted to fail.

## 🛠️ Technical Stack

*   **Logic:** Python 3.11+ (Scikit-Learn, Pandas, NumPy, SciPy)
*   **NLP:** TF-IDF Vectorization, Cosine Similarity, JSD Label Shift
*   **UI:** Vanilla HTML5, CSS3 (Glassmorphism), JavaScript (ES6)

## 📂 Project Structure

*   `main.py`: Entry point for the full research pipeline.
*   `06_analysis.py`: Core meta-modeling logic and LODO evaluation.
*   `dashboard/`: Interactive web interface.
*   `export_json.py`: Bridge script to convert Python results for the web UI.
*   `final_project_report.md`: Complete academic manuscript of the findings.

## 🚦 How to Run

### 1. Environment Setup
```powershell
pip install -r requirements.txt
```

### 2. Execute Pipeline
Run the full analysis on the 20 Newsgroups or Patent dataset:
```powershell
python main.py --dataset newsgroups
```

### 3. Launch Dashboard
Export the data and start a local server:
```powershell
python export_json.py
cd dashboard
python -m http.server 8000
```
Then visit: `http://localhost:8000/index.html`

## 📊 Results Summary

Our Meta-Model achieves an **ROC-AUC of 0.82 ± 0.01**, significantly outperforming traditional similarity-based heuristics ($p < 0.001$). By prioritizing Safe-class precision, the framework effectively eliminates catastrophic transfer errors while maintaining high utility.

---
**Author:** yamini-5  
**Project:** Final Research Project - Negative Transfer Prediction

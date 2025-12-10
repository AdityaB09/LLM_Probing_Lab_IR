# LLM Probing Lab  
### Layerwise Probing • Interpretability • Multi-Dataset Transformer Analysis

---

## 1. Overview

**LLM Probing Lab** is a complete interactive platform for analyzing transformer models layer-by-layer across multiple NLP tasks. It provides:

- Layerwise probing for accuracy, weighted F1, and ECE  
- Dataset exploration and visualization  
- Cross-model comparison  
- Token-level interpretability (Activation Norm, Attention Rollout, Grad × Input)  
- Automated insights generation (Insights AI)  
- Full experiment history and run reproducibility  
- Flask backend + SQLite storage + modern UI  

This system allows researchers to understand **what each transformer layer learns**, compare encoder architectures, and interpret decision-making in depth.

---

## 2. Project Structure

```
llm_probing_lab/
│
├── app.py                 # Flask backend (core API)
├── attribution.py         # Interpretability utilities
├── storage.py             # SQLite database layer
├── summarizer.py          # Insights AI narrative generator
├── config.py              # Dataset + model configuration
├── requirements.txt
│
├── static/                # Frontend JS/CSS
├── templates/             # UI pages for all modules
├── datasets/              # Optional local datasets
│
└── README.md
```

---

## 3. Installation & Environment Setup

### Step 1 — Unzip the Project

Download/unzip:

```
llm_probing_lab-main.zip
```

Navigate into folder:

```bash
cd llm_probing_lab
```

All datasets required for probing are already included in the datasets/ directory.
No Kaggle account, API token, or external download is required.
Simply unzip the project and run the backend + frontend as described.


---

### Step 2 — Create a Python Virtual Environment

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:

- Flask  
- Transformers (HuggingFace)  
- Torch  
- Scikit-learn  
- Tokenizers  
- SQLite integration  
- Summarization dependencies  

---

## 4. Dataset Setup

The platform includes built-in loaders for:

| Dataset | Task |
|--------|------|
| Sarcasm | Sarcasm detection |
| Fake News | Fake vs Real news classification |
| Amazon Reviews | Sentiment analysis |
| Hate Speech | Toxicity detection |

Datasets are automatically fetched or loaded locally based on `config.py`.

No manual setup is needed.

To add a custom dataset, extend:

```
config.py
```

---

## 5. Running the Application

### Start the Backend Server

```bash
python app.py
```

You will see:

```
 * Running on http://127.0.0.1:5000/
```

Open this URL in your browser to load the UI.

---

## 6. Using the Platform

Navigation bar includes:

- **Overview**
- **Explorer**
- **Run Probes**
- **History**
- **Compare**
- **Interpret**
- **Insights AI**

Below is a detailed explanation of every module.

---

#  7. Explorer — Dataset Browser

Navigate:

```
/explore
```

You can:

- Choose any dataset  
- Preview random examples  
- View label distribution  
- Understand dataset size, tasks, and semantics  

Useful for verifying dataset quality before probing.

---

# 8. Run Probes — Layerwise Probing Engine

Navigate:

```
/probe
```

Steps:

1. Select **dataset**  
2. Select **sample size**  
3. Select **model** (DistilBERT / BERT / RoBERTa)  
4. Click **Launch Probe Run**

The system performs:

- Hidden state extraction for all transformer layers  
- Trains logistic regression probes  
- Computes:  
  - Accuracy  
  - Weighted F1  
  - Expected Calibration Error (ECE)  

You will see charts:

- Accuracy vs Layer  
- Weighted F1 vs Layer  
- Calibration vs Layer  

All results are saved automatically.

---

# 📜 9. History — All Past Runs

Navigate:

```
/history
```

Features:

- View all probe runs (ID, dataset, model, timestamp)  
- Re-open results  
- Compare past runs  
- Feed past runs into Insights AI  

Reproducibility is built-in.

---

# ⚖️ 10. Compare Models — Instant Side-by-Side Evaluation

Navigate:

```
/compare
```

Steps:

1. Select dataset  
2. Choose Model A  
3. Choose Model B  
4. Click **Launch Comparison Experiment**

Outputs:

- Accuracy curves for both models  
- Weighted F1 curves  
- Calibration (ECE) curves  
- Auto-generated narrative comparison  

Example:

> “distilbert-base-uncased peaks at layer 6 while bert-base-uncased peaks at layer 10, suggesting deeper layers capture task-specific semantics.”

---

# 🎨 11. Interpret — Token-Level Heatmap Visualization

Navigate:

```
/interpret
```

Capabilities:

- Choose dataset or paste custom text  
- Choose model  
- Select attribution method:  
  - Activation Norm  
  - Attention Rollout  
  - Grad × Input  
- Choose layer  
- Generate Token Heatmap  

Outputs:

- Original text  
- Prediction  
- Token-level importance visualization  

This allows you to understand **exactly which words influenced the model’s decision**.

---

# 12. Insights AI — Automatic Research Report Generation

Navigate:

```
/insights-ai
```

This module analyzes all stored runs and produces:

### ✓ Dataset-level insights  
### ✓ Best-performing models  
### ✓ Layer signatures (universal sweet spots)  
### ✓ Calibration behaviors  
### ✓ Trends across multiple probing experiments  

You also receive:

- Chart: Best Accuracy per Model/Dataset  
- Chart: Universal Layer Signature (Mean Accuracy)  

This transforms your probing logs into a scientific narrative.

---

# 13. Database Architecture

SQLite database: `runs.db`

### Table: `runs`

| Column | Description |
|--------|-------------|
| id | Identifier |
| models | JSON of model names |
| dataset_key | Dataset used |
| summary | Auto-generated insight |
| created_at | Timestamp |

---

### Table: `layer_metrics`

| Column | Description |
|--------|-------------|
| run_id | Foreign key |
| model_name | Example: distilbert-base-uncased |
| layer_index | Transformer layer number |
| accuracy | Probe accuracy |
| f1 | Weighted F1 |
| ece | Calibration error |

---

# 14. Troubleshooting

```bash
pip install -r requirements.txt
```

### Torch/Tokenizers issue?
```
pip install tokenizers==0.13.3
pip install torch --upgrade
```


---

# 🧪 15. Example End-to-End Workflow

### Goal: Compare DistilBERT vs BERT on Fake News Detection

#### Step 1 — Run probing:

- Dataset: Fake News  
- Model: DistilBERT  
- Samples: 1000  

Run again with BERT.

#### Step 2 — Compare models:

Navigate to `/compare`  
Select:

- Model A: DistilBERT  
- Model B: BERT  

#### Step 3 — Interpret:

Navigate to `/interpret`  
Paste:

> “According to experts, this claim is fabricated and lacks evidence.”

View token heatmap.

#### Step 4 — Generate insights:

Navigate to `/insights-ai`  
View:

- Peak layers  
- Best model per dataset  
- Universal sweet spot  
- Narrative research report  

---

# 🚀 16. Extending the System

### Add new transformer models in `config.py`:

```python
MODELS = {
  "distilbert-base-uncased": "...",
  "bert-base-uncased": "...",
  "roberta-base": "...",
  "new-model-here": "..."
}
```




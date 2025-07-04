# Academic Paper Search Engine

A **hybrid academic paper search engine** that combines **BM25** (a traditional keyword-based ranking model) with **MiniLM-based semantic re-ranking**. It supports both keyword and natural language queries and improves precision by **26%** over BM25-only methods.

## Features

* **Hybrid Search (BM25 + MiniLM)** for improved semantic understanding

* **FastAPI** backend for high-speed search query handling

* **React + HTML/CSS/JS GUI** for interactive user experience

* **Natural language query support**

* **Evaluation metrics**: Precision, Recall, F1

## Architecture

![Architecture Diagram](images/architecture.png)

## Dataset

* Source: [arXiv.org](https://arxiv.org/)
* Domain: Computer Vision (`cs.CV`)
* Size: \~150,000 academic papers
* Metadata includes: `ID`, `Title`, `Abstract`, `Authors`, `Categories`, `DOI`, etc.

## Core Components

1. **BM25**: First-stage retrieval for initial candidate selection

* Probabilistic ranking based on term frequency and inverse document frequency
* High recall for keyword-based matching
* Computational efficiency for large-scale retrieval

2. **MiniLM**: Second-stage semantic re-ranking

* Lightweight sentence-transformer model (6x smaller than BERT)
* Deep semantic representations for contextual understanding
* Maintains 95% of BERT's performance with significantly reduced computational cost

## Running the Application

1. **Start Backend (FastAPI)**

```bash
uvicorn search_api:app --reload
```

Open `http://127.0.0.1:8000/docs` to test the API.

2. **Start Frontend**

Open `frontend/index.html` in your browser (static HTML/JS-based).

## Evaluation

* **Expanded Precision**: Measures relevance across query variants
* **Expanded Recall**: Measures completeness across variants
* **F1 Score**: Harmonic mean of precision and recall
* Evaluation done via `evaluation.py` with LLM-generated query expansions

| Model  | Precision | F1 Score |
| ------ | --------- | -------- |
| BM25   | 0.47      | 0.33     |
| Hybrid | **0.73**  | **0.42** |

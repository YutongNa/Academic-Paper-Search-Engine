# README

### File3_Evaluation

This is the third of three submitted zip files, File3_Evaluation.

This project evaluates the performance of two document retrieval models:
- BM25 
- Hybrid (BM25 + MiniLM)

The evaluation is based on manually annotated relevance labels across multiple query variants.

---
#### How to run `evaluation.py` 
The script can be executed either **locally** or in **Google Colab.**

**- Option 1: Running Locally**
1. Place the `evaluation.py` script and the `Data`folder in the same directory. 
2. From your terminal, run 

```python
python evaluation.py
```

**- Option 2: Running on Google Colab**
1. Upload the `evaluation.py` script and the entire `Data`folder in the same directory. 
2. In Colab code cell, run:

```python 
!python evaluation.py
```

**Important Note**
The script loads all CSV files using relative paths like:

```python
pd.read_csv("Data/Filename.csv")
```

This means:
- Do not use absolute paths like /content/your_file.csv.
- The Data/ folder must be uploaded or placed in the same directory as the script.
- This setup works for both local and Colab environments, as long as the folder structure is preserved.


---

#### Data Folder Overview
The Data/ folder contains all necessary CSV files for evaluation:

1. `queries.csv`
- Contains five test queries. Each query includes five manually created variants (generated using an LLM through a web interface, not through code).

2. `bm25_retrieval_results.csv` and `hybrid_retrieval_results.csv`
- Contain ranked retrieval results for all test queries and their variants.
- Results were generated using the code in File1_Preprocessing_and_Retrieval, specifically under Section 3: Retrieval and Ranking.
- For simplicity, results were saved manually into CSV files instead of displaying large DataFrames in the notebook.
- These results reflect the document rankings produced by both models.

3. `bm25_relevance_labels.csv` and `hybrid_relevance_labels.csv`
- Contain binary relevance labels:
    - 1 for relevant documents
    - 0 for irrelevant documents

- Labels were manually annotated by reading the abstracts of each retrieved document.
- Used as the ground truth for evaluation.


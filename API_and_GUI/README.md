# README

### File2_API_and_GUI

This is the second of three submitted zip files, File2_API_and_GUI.

This file implements a semantic search engine using a hybrid approach combining **BM25** and **MiniLM**. It exposes the functionality through a FastAPI web interface for easy query testing and result inspection.

---

#### Compatibility Notice

> This project is currently only compatible with **Windows**.  
> It **does not run on macOS** due to GPU-related compatibility issues with `MiniLM` on the Metal backend (MPS). 

---

#### Dataset Notice

The dataset included in this project (`2_lemmatized_data_subset.csv`) is only a **10% subset** of the original full dataset. This was done to comply with file size restrictions (under 50MB) for submission.

If you would like to use the complete dataset (~100,000KB), you can download it from the following Google Docs link:

**Download Full Dataset (From Google Drive)**: https://drive.google.com/uc?export=download&id=1QHjuTZprI-tkLLSQtyE-Fg0L2mpDjsPJ


**Note:** Since this is only a portion of the full dataset, the search results will **not match the results shown in the original demo**.

---

#### Switching to the Full Dataset

Once you’ve downloaded the full dataset, you can easily switch to using it by modifying a line in the `search_api.py` file. In `search_api.py`, you'll find:

```python
# Loading Preprocessed Data
df = pd.read_csv("2_lemmatized_data_subset.csv")  # This is 10% subset of the original file 

# Uncomment the line below if you have the original file downloaded from the OneDrive link.
# Make sure that the original file is in the same folder as this .py file.
# df = pd.read_csv("2_lemmatized_data.csv")
```
---

#### Setup Instructions (Windows Only)

Ensure all project files are in the **same directory**, and run the following commands **from that same directory**.

1. Create a Virtual environment (python -m venv venv)​

2. Activate the Virtual Environment ​(call venv\Scripts\activate​)

3. Install requirements (pip install -r requirements.txt)​

4. Run API (uvicorn search_api:app --reload)​

5. Run locally on your web browser ​(http://127.0.0.1:8000/docs)

**Note:** If the query search took more than 30s to return results, try to refresh the page a few times. Ideally, the results will returned within 20 seconds.
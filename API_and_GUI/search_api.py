from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import json
from fastapi.responses import FileResponse
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

#Initializing FastAPI
app = FastAPI()


#Using CORS so the API can be accessible using REACT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))

#Loading Preprocessed Data
df = pd.read_csv("2_lemmatized_data_subset.csv") # This is 10% subset of the original file 

#Uncomment the line below if have the original file downloaded from the one drive link 
#Make sure that the original file is under the same folder of this .py file 
#df = pd.read_csv("2_lemmatized_data_.csv")
tokenized_corpus = [f"{t} {a}".split() for t, a in zip(df['title_lemmatized'], df['abstract_lemmatized'])]
doc_ids = df['id'].astype(str).tolist()
doc_texts = [" ".join(tokens) for tokens in tokenized_corpus]



#BM25 and MiniLM
bm25 = BM25Okapi(tokenized_corpus)
minilm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


#Hybrid Search

def hybrid_search(query, doc_ids, doc_texts, bm25, alpha, top_k):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Expand candidate pool for MiniLM reranking
    bm25_limit = max(top_k * 3, 100)
    #bm25_limit = min(max(top_k * 2, 50), 100)
    top_n = sorted(zip(doc_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:bm25_limit]

    top_doc_ids = [doc_id for doc_id, _ in top_n]
    top_doc_texts = [doc_texts[doc_ids.index(doc_id)] for doc_id in top_doc_ids]

    query_emb = minilm_model.encode(query, convert_to_tensor=True)
    doc_embs = minilm_model.encode(top_doc_texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, doc_embs)[0]

    def min_max_norm(tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)

    bm25_tensor = torch.tensor([score for _, score in top_n])
    bm25_norm = min_max_norm(bm25_tensor)
    cosine_norm = min_max_norm(cosine_scores)

    hybrid_scores = alpha * bm25_norm + (1 - alpha) * cosine_norm

    full_ranking = [
        (doc_id, float(bm25_score), float(hybrid_score))
        for (doc_id, bm25_score), hybrid_score in zip(top_n, hybrid_scores.tolist())
    ]

    # Now sort by hybrid score and return only the top_k final results
    final_ranking = sorted(full_ranking, key=lambda x: x[2], reverse=True)[:top_k]

    return final_ranking


# API Endpoint 
@app.get("/search")
def search(
    query: str = Query(..., description="Search query string"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight between BM25 and MiniLM (0 to 1)"),
    top_k: int = Query(100, ge=1, le=100, description="Number of results to return"),
    mode: str = Query("hybrid", enum=["bm25", "minilm", "hybrid"], description="Search mode: BM25, MiniLM, or Hybrid")
):
    if mode == "bm25":
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_n = sorted(zip(doc_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(doc_id, score, score) for doc_id, score in top_n]  # dummy hybrid score = bm25
    
    elif mode == "minilm":
        query_emb = minilm_model.encode(query, convert_to_tensor=True)
        doc_embs = minilm_model.encode(doc_texts, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_emb, doc_embs)[0]
        top_n = sorted(zip(doc_ids, cosine_scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(doc_id, score, score) for doc_id, score in top_n]  # dummy bm25 score = minilm
    
    else:
    
        results = hybrid_search(query, doc_ids, doc_texts, bm25, alpha, top_k)

    
    if not results:
        print(f"[!] No results found for query: '{query}'")
        return {
            "query": query,
            "alpha": alpha,
            "top_k": top_k,
            "results": [],
            "message": "No matching documents found. Try simplifying your query."
        }
    
    output = []

    print("Top search results:", results)
    print("Sample doc ID:", results[0][0] if results else "No results")

    for rank, (doc_id, bm25_score, hybrid_score) in enumerate(results, start=1):
        row_match = df[df["id"].astype(str) == doc_id]
        if row_match.empty:
            print(f"[!] Document with ID {doc_id} not found in DataFrame.")
            continue
        
        row = row_match.iloc[0]
        output.append({
            "rank": rank,
            "id": doc_id,
            "title": row["title"],
            "authors": row.get("authors", "Unknown"),
            "abstract": row["abstract"],
            "bm25_score": bm25_score,
            "hybrid_score": hybrid_score
        })

    return {
        "query": query,
        "alpha": alpha,
        "top_k": top_k,
        "results": output
    }
@app.get("/autocomplete")
def autocomplete(prefix: str = Query(..., min_length=1)):
    prefix_lower = prefix.lower()
   
    titles = df["title"].dropna().tolist()
   
    matches = [title for title in titles if title.lower().startswith(prefix_lower)]
   
    return JSONResponse(content={"suggestions": matches[:10]})
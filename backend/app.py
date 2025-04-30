import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
#from sqlalchemy import text
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import BertTokenizerFast, BertModel
from sentence_transformers import SentenceTransformer
import csv
import json
from pathlib import Path
# from convokit import Corpus
import faiss
import re
from safetensors.torch import load_file, save_file

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
# LOCAL_MYSQL_USER = "root"
# LOCAL_MYSQL_USER_PASSWORD = "Lukeshao2022" # Fill with personal password for MySQL
# # TODO: Delegate these values to env. vars
# LOCAL_MYSQL_PORT = 3306
# LOCAL_MYSQL_DATABASE = "FitMyVibe"

# mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
# mysql_engine.load_file_into_db('dump.sql')

app = Flask(__name__)
CORS(app)

embs_reddit = np.load("social-component/reddit/reddit_embeddings.npy")
print(f"Reddit embedding shape: {embs_reddit.shape}")

embs_prods = np.load("social-component/reddit/prod_embeddings-2.npy")
print(f"Product embedding shape: {embs_prods.shape}")

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

def vectorize_query(query):
    """
    Vectorizes the ad-hoc query using pre-trained BERT embeddings.
    """

    merge_state_dict = {}
    files = ["tensor_pack/chunk_1_1.safetensors",
             "tensor_pack/chunk_1_2.safetensors",
             "tensor_pack/chunk_1_3.safetensors",
             "tensor_pack/chunk_1_4.safetensors",
             "tensor_pack/chunk_2.safetensors",
             "tensor_pack/chunk_3.safetensors",
             "tensor_pack/chunk_4.safetensors",
             "tensor_pack/chunk_5.safetensors"]
    merged_file = "fashion-bert-output-v4/model.safetensors"

    def merge_files(files):
        for file in files:
            load_files_dict = load_file(file)
            merge_state_dict.update(load_files_dict)
    
    merge_files(files)

    save_file(merge_state_dict, merged_file)
    del merge_state_dict

    model = SentenceTransformer('fashion-bert-output-v2')
    encoded_query = model.encode([query], convert_to_numpy=True) #tokenizer(query, return_tensors='pt', padding=True, truncation=True)
    encoded_query = encoded_query / np.linalg.norm(encoded_query, axis=1, keepdims=True)

    # with torch.no_grad():
    #     outputs = model(**encoded_query)
        
    # query_embeddings = outputs.last_hidden_state[:, 0, :] should be an npdarray
    # encoded_query = encoded_query.reshape(1, -1)
    encoded_query = encoded_query.astype("float32")
    print(f"Query embedding shape: {encoded_query.shape}")
    return encoded_query # query_embeddings

def vector_from_id(article_id):
    """
    Returns the associated embedding vector to a certain article ID.
    Article IDs are specified in the database, where each article is associated with
    an ID as its primary key.
    Format of the return value is a tuple, where the first value is the product ID
    and the second is the embedding represented as a list of decimals.
    """
    # csv_path = 'FINAL-EMBEDDINGS.csv' # Replace with path to file (conditioned on gender and price)
    # with open(csv_path, newline="", encoding="utf-8") as f:
    #     reader = csv.reader(f)
    #     header = next(reader)
    #     for row in reader:
    #         # row[0] is the id column
    #         if row[0] == str(article_id):
    #             # parse the remaining 3xx dimensions into floats
    #             vals = [float(x) for x in row[1:]]
    #             return (np.array(vals, dtype=float))
    
    # embs_prods = np.load("social-component/reddit/prod_embeddings-2.npy")
    return embs_prods[article_id]

def order_articles(query_embeddings, filtered_ids, article_vectors):
    """
    Orders the articles in the database based on a cosine similarity metric.
    Takes the embedding of the query and the vectors of the searched articles as
    input, and returns a ranked list of article IDs based on similarity scores.
    """
    k_corpus = 10
    k_prod = min(len(filtered_ids), 30)
    alpha, beta = 1.0, 0.75

    # embs_reddit = np.load("social-component/reddit/reddit_embeddings.npy") 
    dim_reddit   = embs_reddit.shape[1]
    index_reddit = faiss.IndexFlatIP(dim_reddit)
    index_reddit.add(embs_reddit.astype("float32"))
    query_embeddings = query_embeddings.astype("float32")

    # print("REDDIT EMBEDDING SHAPE: " + query_embeddings.shape)

    _, idxs_u = index_reddit.search(query_embeddings, k_corpus)
    top_u_embs = embs_reddit[idxs_u[0]]  # shape (k_corpus, dim)

    # embs_prods = np.loadtxt("FINAL-EMBEDDINGS.csv")
    # embs_prods = np.loadtxt("FINAL-EMBEDDINGS.csv", delimiter=",", skiprows=1)
    # embs_prods = np.load("social-component/reddit/prod_embeddings-2.npy")
    # embs_prods = embs_prods[:, 1:]
    # embs_prods = embs_prods[:500]

    def build_search_dict(filtered_ids):
        count = 0
        search_dict = {}
        for id in filtered_ids:
            search_dict[count] = id
            count += 1
        return search_dict
    
    search_dict = build_search_dict(filtered_ids)

    dim_prods = article_vectors.shape[1]

    index_prods = faiss.IndexFlatIP(dim_prods)  
    index_prods.add(article_vectors.astype("float32"))

    expanded_q = alpha * query_embeddings + beta * top_u_embs.mean(axis=0, keepdims=True)
    expanded_q /= np.linalg.norm(expanded_q, axis=1, keepdims=True)

    sim_p, idxs_p = index_prods.search(expanded_q.astype("float32"), k_prod)
    print("IDXS_P PRINTING")
    print(idxs_p)
    return [search_dict[search_id] for search_id in idxs_p[0]]

    # after table
    sim_scores = []
    article_ids = []
    for article_id, article_vector in article_vectors:
        tensor = torch.Tensor(article_vector)
        sim = torch.cosine_similarity(expanded_q, tensor)
        print(f"article id: {article_id} (sim score = {sim})")
        sim_scores.append(sim)
        article_ids.append(article_id)

    articles_scores = list(zip(article_ids, sim_scores))
    articles_scores.sort(key=lambda x : x[1], reverse=True)

    ranked_articles_ids = []
    for article_id, _ in articles_scores:
        ranked_articles_ids.append(article_id)

    return ranked_articles_ids


def table_lookup(indices):
    """
    Looks up the relevant data about a set of articles given their article IDs.
    In its current form, this lookup returns information about the product name,
    its regular price, and a link to the image.
    """
    
    items_path = Path("COMBINED-FINAL-DEDUPED.json")
    with items_path.open("r", encoding="utf-8") as f:
        items_data = json.load(f)

    items_by_id = {item["ID"]: item for item in items_data}

    def truncate_description(description, word_limit=20):
        """Truncates description to specified word limit and adds '...' if truncated"""
        if not description:
            return ""
        words = description.split()
        if len(words) <= word_limit:
            return description
        return " ".join(words[:word_limit]) + "..."

    ranked_results = []
    for idx in indices:
        rec = items_by_id.get(idx)
        if (rec):
            img_link = rec.get("prodImgLink") or "static/images/clothing-icon.png"
            prod_link = rec.get("prodLink", "") or "https://www.mercari.com/jp/"
            ranked_results.append({
                "prodName": rec.get("name"),
                "prodPrice": rec.get("price"),
                "prodImgLink": img_link,
                "prodLink": prod_link,
                "prodDesc": truncate_description(rec.get("description"))
            })
    return ranked_results

@app.route("/articles")
def episodes_search():
    query = request.args.get("inspirationDesc")
    gender = request.args.get("gender", default=None)
    if gender == "men":
        gender = "m"
    elif gender == "women":
        gender = "f"
    else:
        gender = None

    budget = request.args.get("budget", default=None)
    if budget == "":
        budget == None
    else:
        budget = float(budget)
    article = request.args.get("article", default=None)
    if article == "T":
        article = "Tops"
    elif article == "B":
        article = "Bottoms"
    elif article == "S":
        article = "Shoes"
    elif article == "A":
        article = "Accessories"
    else:
        article = None

    query_embeddings = vectorize_query(query)

    items_path = Path("COMBINED-FINAL-DEDUPED.json")
    with items_path.open("r", encoding="utf-8") as f:
        items_data = json.load(f)

    items_by_id = {item["ID"]: item for item in items_data}

    
    def check_filters():
        filter_ids = []
        for id in range(0, 1332):
            rec = items_by_id.get(id)
            if rec == None:
                continue

            gender_filter = True if gender == "" or gender == None else rec['gender'] == gender
            if not(isinstance(budget, float)) or (rec["price"] == ""):
                budget_filter = True
            else:
                budget_low = budget - 24
                budget_high = budget + 25
                budget_filter = float(rec['price']) >= budget_low and float(rec['price']) <= budget_high
            article_filter = True if article == "" or article == None else rec['category'] == article

            # print(f"Filters for article {id}: G={gender_filter}, B={budget_filter}, A={article_filter}")

            if (gender_filter and budget_filter and article_filter):
                filter_ids.append(id)
                # print(f"ADDED IDX {id} TO CANDIDATES")
        
        return filter_ids
    
    filter_ids = check_filters()
    article_vectors = embs_prods[filter_ids]
    
    # print(f"ARTICLE VECTORS: {article_vectors}")

    # Articles that pass the filter are stored in article_vectors
    # Make order articles use the article vectors as the set of articles to query
    ranked_idx = order_articles(query_embeddings, filter_ids, article_vectors)

    final_idx = []
    for idx in ranked_idx:
        if idx in filter_ids:
            final_idx.append(idx)

    print("RANKED INDICES" + str(ranked_idx))

    ranked_results = table_lookup(final_idx)

    print("DONE RANKING")
    print(ranked_results)

    return json.dumps(ranked_results, default=str)
import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
from sqlalchemy import text
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
    files = ["fashion-bert-output-v4/chunk_1_1.safetensors",
             "fashion-bert-output-v4/chunk_1_2.safetensors",
             "fashion-bert-output-v4/chunk_1_3.safetensors",
             "fashion-bert-output-v4/chunk_1_4.safetensors",
             "fashion-bert-output-v4/chunk_2.safetensors",
             "fashion-bert-output-v4/chunk_3.safetensors",
             "fashion-bert-output-v4/chunk_4.safetensors",
             "fashion-bert-output-v4/chunk_5.safetensors"]
    merged_file = "fashion-bert-output-v4/model.safetensors"

    def merge_files(files):
        for file in files:
            load_files_dict = load_file(file)
            merge_state_dict.update(load_files_dict)
    
    merge_files(files)

    save_file(merge_state_dict, merged_file)
    del merge_state_dict

    model = SentenceTransformer('fashion-bert-output-v4')
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

    # raise ValueError(f"Article ID {article_id} not found in {csv_path}")

    # query_sql = f"SELECT vecPos, vecVal FROM prodvec WHERE prodID = {article_id}"
    # data = mysql_engine.query_selector(text(query_sql))
    # vector = np.zeros(768)
    # for i in data:
    #     idx = i[0]
    #     val = i[1]
    #     if (idx != 0):
    #         vector[idx - 1] = val
    # return (article_id, vector)

def order_articles(query_embeddings, article_vectors):
    """
    Orders the articles in the database based on a cosine similarity metric.
    Takes the embedding of the query and the vectors of the searched articles as
    input, and returns a ranked list of article IDs based on similarity scores.
    """
    k_corpus, k_prod = 10, 20
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
    dim_prods   = embs_prods.shape[1]

    # print(dim_prods)

    index_prods = faiss.IndexFlatIP(dim_prods)  
    index_prods.add(embs_prods.astype("float32"))

    expanded_q = alpha * query_embeddings + beta * top_u_embs.mean(axis=0, keepdims=True)
    expanded_q /= np.linalg.norm(expanded_q, axis=1, keepdims=True)

    sim_p, idxs_p = index_prods.search(expanded_q.astype("float32"), k_prod)
    # print(idxs_p)
    return idxs_p[0]

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

def table_lookup(indices, gender, budget, article):
    """
    Looks up the relevant data about a set of articles given their article IDs.
    In its current form, this lookup returns information about the product name,
    its regular price, and a link to the image.
    Only returns indices for articles that meet the proper criteria:
    - Gender is specified argument gender
    - Cost is between budget - 24 and budget + 25
    - Article type is consistent
    """
    items_path = Path("COMBINED-FINAL.json")
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
        rec_gender = rec['gender']
        rec_cost = float(rec['price'])
        budget_low = budget - 24
        budget_high = budget + 25
        rec_article = rec['category']

        if (rec_gender == gender and
            rec_cost >= budget_low and
            rec_cost <= budget_high and
            rec_article == article):
            print("MATCHED ON")
            print(rec)
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

    # ranked_results = []
    # for idx in indices:
    #     lookup_query = f"""SELECT proddesc.prodName, prodprice.prodRegPrice, prodlink.prodImageLink, prodlink.prodLink
    #         FROM proddesc
    #         JOIN prodprice
    #             ON proddesc.prodID = prodprice.prodID
    #         JOIN prodlink
    #             ON proddesc.prodID = prodlink.prodID
    #         WHERE proddesc.prodID = {idx}"""
    #     lookup_data = mysql_engine.query_selector(text(lookup_query))
    #     ranked_results += lookup_data
    
    # keys = ["prodName", "prodPrice", "prodImgLink", "prodLink"]
    # dict_results = [dict(zip(keys, res)) for res in ranked_results]
    # print(dict_results)
    # return dict_results
        

@app.route("/articles")
def episodes_search():
    query = request.args.get("inspirationDesc")
    gender = request.args.get("gender")
    budget = request.args.get("budget")
    article = request.args.get("article")
    # style = request.args.get("style")
    # brand = request.args.get("brand")

    # print("QUERY: " + query)
    query_embeddings = vectorize_query(query)

    article_vectors = []
    for id in range(1, 500):
        article_vectors.append(vector_from_id(id))

    ranked_idx = order_articles(query_embeddings, article_vectors)[:20]
    print("RANKED INDICES" + str(ranked_idx))
    ranked_results = table_lookup(ranked_idx, gender, budget, article)
    print("DONE RANKING")

    return json.dumps(ranked_results, default=str)

# if 'DB_NAME' not in os.environ:
#     app.run(debug=True,host="0.0.0.0",port=5000)


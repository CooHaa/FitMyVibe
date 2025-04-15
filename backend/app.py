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
from transformers import AutoModel, BertTokenizerFast, BertModel

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "" # Fill with personal password for MySQL
# TODO: Delegate these values to env. vars
LOCAL_MYSQL_PORT = 3306
LOCAL_MYSQL_DATABASE = "FitMyVibe"

mysql_engine = MySQLDatabaseHandler(LOCAL_MYSQL_USER,LOCAL_MYSQL_USER_PASSWORD,LOCAL_MYSQL_PORT,LOCAL_MYSQL_DATABASE)

# Path to init.sql file. This file can be replaced with your own file for testing on localhost, but do NOT move the init.sql file
mysql_engine.load_file_into_db('dump.sql')

app = Flask(__name__)
CORS(app)

#####
# Article types
topwear_articles = ['shirt',
 'tshirt',
 'top',
 'sweatshirt',
 'kurta',
 'waistcoat',
 'rain jacket',
 'blazer',
 'shrug',
 'dupatta',
 'tunic',
 'jacket',
 'sweater',
 'kurti',
 'suspender',
 'nehru jacket',
 'romper',
 'dresse',
 'lehenga choli',
 'belt',
 'suit']

bottomwear_articles = ['jean',
 'track pant',
 'short',
 'skirt',
 'trouser',
 'capri',
 'tracksuit',
 'swimwear',
 'legging',
 'salwar and dupatta',
 'patiala',
 'stocking',
 'tight',
 'churidar',
 'salwar',
 'jegging',
 'rain trouser']

footwear_articles = ['shoe',
 'casual shoe',
 'flip flop',
 'sandal',
 'formal shoe',
 'flat',
 'sports shoe',
 'heel',
 'sports sandal']
# If a keyword is not one of these, we put it under accessories
#####


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

def vectorize_input(style, category, budget):
    """
    Vectorizes (np.array) the input based on the input style keyword, article type, and budget.
    The vector takes on the following form:

    `<budget, isTopwear, isBottomwear, isShoes, isAccessory, isCasual, isFormal, isAthletic>`

    Keyword arguments:

    style -- A style keyword, which is one of ['Casual', 'Formal', 'Athletic', 
    'Athleisure', 'Business-casual']

    category -- An article category that will be referenced with `topwear_articles`, 
    `bottomwear_articles` and `footwear_articles` to categorize the input item as
    either topwear, bottomwear, footwear, or an accessory. Any item that does not
    fit into topwear, bottomwear, or footwear will automatically be categorized as
    an accessory.

    budget -- The budget of the user, in dollars.
    """
    budget_comp = [int(budget)]
    category_comp = [0, 0, 0, 0]
    style_comp = [0, 0, 0]

    # For flexibility, all plurals are turned into singular forms
    # TODO: Change to edit distance calculation
    # if (category[-1] == 's'):
    #     category = category[:-1]

    if (category in topwear_articles):
        category_comp[0] = 1
    elif (category in bottomwear_articles):
        category_comp[1] = 1
    elif (category in footwear_articles):
        category_comp[2] = 1
    else:
        category_comp[3] = 1

    if (style == "Casual"):
        style_comp[0] = 1
    elif (style == "Formal"):
        style_comp[1] = 1
    elif (style == "Athletic"):
        style_comp[2] = 1
    elif (style == "Athleisure"):
        style_comp[0] = 1
        style_comp[2] = 1
    else: # Business casual, catch all for now
        style_comp[0] = 1
        style_comp[1] = 1
    
    vector = budget_comp + category_comp + style_comp
    return np.array(vector)

def sql_search(gender):
    """
    Returns a list of JSON entries where the gender equals the parameter gender.
    """
    if (gender == "men"):
        gender = "Men"
    else:
        gender = "Women"
    query_sql = "SELECT * FROM articles WHERE gender = \"{gender}\""
    keys = ["id","budget","gender", "mCat", "sCat", "articleType", "color", "season", "year", "usage", "name"]
    data = mysql_engine.query_selector(text(query_sql))
    return [dict(zip(keys,i)) for i in data]

def vectorize_query(query):
    """
    Vectorizes the ad-hoc query using pre-trained BERT embeddings.
    """
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    encoded_query = tokenizer(query, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**encoded_query)
        
    query_embeddings = outputs.last_hidden_state[:, 0, :]
    return query_embeddings

def vector_from_id(id):
    query_sql = "SELECT vecPos, vecVal FROM prodvec WHERE prodID = '{id}'"
    data = mysql_engine.query_selector(text(query_sql))
    vector = []
    for i in data:
        vector[i[0]] = i[1]
    return vector

def order_articles(query_embeddings, article_vectors, article_ids):
    # after table lookups

    sim_scores = []

    for article_vector, article_id in article_vectors:
        tensor = torch.Tensor(article_vector)
        sim = torch.cosine_similarity(query_embeddings.mean(axis=0), tensor.mean(axis=0))
        sim_scores.append(sim)
    
    articles_scores = list(zip(article_ids, sim_scores))
    articles_scores.sort(key=lambda x : x[1], reverse=True)

    ranked_articles_ids = []
    for article_id, _ in articles_scores:
        ranked_articles_ids.append(article_id)

    return ranked_articles_ids

def table_lookup(indices):
    ranked_results = []
    for idx in indices:
        lookup_query = """SELECT proddesc.prodName, prodprice.prodRegPrice, prodlink.prodImageLink FROM proddesc
            JOIN prodprice
                ON proddesc.prodID = prodpice.prodID
            JOIN prodlink
                ON proddesc.prodID = prodlink.prodID
            WHERE proddesc.prodID = {idx}"""
        lookup_data = mysql_engine.query_selector(text(lookup_query))
        ranked_results += lookup_data
    
    return ranked_results
        

@app.route("/articles")
def episodes_search():
    query = request.args.get("inspirationDesc")
    query_embeddings = vectorize_query(query)
    article_vectors = []
    for id in range(1, 166):
        article_vectors.append(vector_from_id(id))
    ranked_idx = order_articles(query_embeddings, article_vectors, list(range(1, 166)))
    ranked_results = table_lookup(ranked_idx)

    return json.dumps(ranked_results)
    
    

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)



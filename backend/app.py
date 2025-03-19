import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# These are the DB credentials for your OWN MySQL
# Don't worry about the deployment credentials, those are fixed
# You can use a different DB name if you want to
LOCAL_MYSQL_USER = "root"
LOCAL_MYSQL_USER_PASSWORD = "pwd" # Fill with personal password for MySQL
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
topwear_articles = ['shirts',
 'tshirts',
 'tops',
 'sweatshirts',
 'kurtas',
 'waistcoat',
 'rain jacket',
 'blazers',
 'shrug',
 'dupatta',
 'tunics',
 'jackets',
 'sweaters',
 'kurtis',
 'suspenders',
 'nehru jackets',
 'rompers',
 'dresses',
 'lehenga choli',
 'belts',
 'suits']

bottomwear_articles = ['jeans',
 'track pants',
 'shorts',
 'skirts',
 'trousers',
 'capris',
 'tracksuits',
 'swimwear',
 'leggings',
 'salwar and dupatta',
 'patiala',
 'stockings',
 'tights',
 'churidar',
 'salwar',
 'jeggings',
 'rain trousers']

footwear_articles = ['casual shoes',
 'flip flops',
 'sandals',
 'formal shoes',
 'flats',
 'sports shoes',
 'heels',
 'sports sandals']
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
    budget_comp = [budget]
    category_comp = [0, 0, 0, 0]
    style_comp = [0, 0, 0]

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
    query_sql = f"""SELECT * FROM articles WHERE gender = {gender}"""
    keys = ["id","budget","gender", "mCat", "sCat", "articleType", "color", "season", "year", "usage", "name"]
    data = mysql_engine.query_selector(query_sql)
    return json.dumps([dict(zip(keys,i)) for i in data])

def order_articles(style, category, budget, article_jsons):
    """
    Returns an ordered ranking of JSONs representing articles of clothing based on
    what is most similar to the input `style`, `category` and `budget` parameters.
    
    Similarity is calculated using cosine similarity.
    """

    cosine_scores = []

    input_vector = vectorize_input(style, category, budget)
    for article in article_jsons:
        article_style = article['usage']
        article_category = article['articleType']
        article_cost = article['budget']
        article_vector = vectorize_input(article_style, article_category, article_cost)
        cosine_numerator = np.dot(input_vector, article_vector)
        cosine_denominator = np.norm(input_vector) * np.norm(article_vector)
        cosine_score = cosine_numerator / cosine_denominator
        print(cosine_score)
        cosine_scores.append(cosine_score)
    
    articles_scores = list(zip(article_jsons, cosine_scores))
    articles_scores.sort(key=lambda x : x[1], reverse=True)

    ranked_articles = []
    for article, _ in articles_scores:
        ranked_articles.append(article)

    return ranked_articles


@app.route("/articles")
def episodes_search():
    style = request.args.get("style")
    category = request.args.get("category")
    budget = request.args.get("budget")
    gender = request.args.get("gender")
    result_json = sql_search(gender)
    ranked_results = order_articles(style, category, budget, result_json)
    return ranked_results
    

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)
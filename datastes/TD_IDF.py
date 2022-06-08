import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

def item(id):
  return reviews.loc[reviews['app_id'] == id]['app_name'].tolist()[0]  #itemid to name
#Recommander les jeux les plus similaires
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    idx = reviews[reviews["app_id"]== item_id].index.tolist()[0] #itemid to index
    similar_indices = cosine_similarities[idx].argsort()[::-1]  #sort the array in ascending order and return the index
    for i in range(num):
       print("Recommended: " + item(reviews["app_id"][similar_indices[i+1]]) + " (score:" +      str(cosine_similarities[idx][similar_indices[i+1]]) + ")")

reviews = pd.read_csv("https://raw.githubusercontent.com/alyang666/DS50/main/datastes/steam_reviews_cum.csv")

cosine_similarities=pd.read_csv("cosine_similarities.csv",index_col=0)

cosine_similarities=cosine_similarities.values

recommend(item_id=10, num=5)








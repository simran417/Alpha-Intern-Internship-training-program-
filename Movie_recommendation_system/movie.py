import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie=pd.read_csv("C:/Users/hp/Downloads/dataset.csv")

movie['tag'] = movie['genre']+movie['overview']
movie['l_title']=movie['title'].astype(str).str.lower() #add column for movie name in small letters

dataset1=movie[['id','l_title','tag']]

cv=CountVectorizer(max_features=10000, stop_words='english')

vec=cv.fit_transform(dataset1['tag'].values.astype('U')).toarray()

vec.shape
sim=cosine_similarity(vec)
dist=sorted(list(enumerate(sim[0])),reverse=True, key =lambda vec:vec[1])

def recommend(l_title):
    try:
        index = dataset1[dataset1['l_title'] == l_title].index[0]
        dist = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vec: vec[1])
        for i in dist[1:6]:  # skip the first one, which is the movie itself
            print(dataset1.iloc[i[0]].l_title)
    except IndexError:
        print("Movie not found in the dataset")

Movie_name=input("enter movie: ")
recommend(Movie_name)

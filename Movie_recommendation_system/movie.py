import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movie=pd.read_csv("C:/Users/hp/Downloads/dataset.csv")
# print(movie.head())
movie['tag'] = movie['genre']+movie['overview']
# print(movie.head())
dataset1=movie[['id','title','tag']]
# print(dataset1.head())
cv=CountVectorizer(max_features=10000, stop_words='english')
# print(cv)
vec=cv.fit_transform(dataset1['tag'].values.astype('U')).toarray()
# print(vec)
vec.shape
sim=cosine_similarity(vec)
dist=sorted(list(enumerate(sim[0])),reverse=True, key =lambda vec:vec[1])

def recommend(title):
    try:
        index = dataset1[dataset1['title'] == title].index[0]
        dist = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vec: vec[1])
        for i in dist[1:6]:  # skip the first one, which is the movie itself
            print(dataset1.iloc[i[0]].title)
    except IndexError:
        print("Movie not found in the dataset")

abc=input("entor movie: ")
# recommend("Dilwale Dulhania Le Jayenge")   
recommend(abc)     
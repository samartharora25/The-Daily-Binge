#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import ast


# In[64]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[66]:


movies.head(1)


# In[67]:


credits.head(1)


# In[68]:


movies=movies.merge(credits,on='title')


# In[69]:


movies.head(1)


# In[70]:


#INCLUDED
#genres
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[71]:


movies.info()


# In[72]:


movies.head()


# In[73]:


movies.isnull().sum()


# In[74]:


movies.dropna(inplace=True)


# In[75]:


movies.duplicated().sum()


# In[76]:


movies.iloc[0].genres


# In[77]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[78]:


movies['genres']=movies['genres'].apply(convert)


# In[79]:


movies.head()


# In[80]:


movies['keywords']=movies['keywords'].apply(convert)


# In[81]:


movies.head()


# In[82]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
        else:
            break
    return L


# In[83]:


movies['cast']=movies['cast'].apply(convert3)


# In[84]:


movies.head()


# In[86]:


movies['crew'][0]


# In[89]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L 


# In[91]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[92]:


movies.head()


# In[94]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[95]:


movies.head()


# In[97]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[98]:


movies.head()


# In[99]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[100]:


movies.head()


# In[123]:


new_df=movies[['movie_id','title','tags']]


# In[124]:


new_df.head()


# In[127]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[128]:


new_df.head()


# In[130]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[131]:


new_df.head()


# In[142]:


get_ipython().system('pip install nltk')


# In[143]:


import nltk


# In[144]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[145]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)
    


# In[147]:


new_df['tags']=new_df['tags'].apply(stem)


# In[148]:


#using vectorisation 
#Bag of words method 
#dimensional space of 5000 words


# In[149]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[150]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[151]:


vectors


# In[152]:


vectors[0]


# In[153]:


cv.get_feature_names()


# In[154]:


#calculating the distance between the movies
#using cosine distance
#greater the angle, more dis-similar the movie
#distance is inversly proportional to similarity


# In[155]:


from sklearn.metrics.pairwise import cosine_similarity


# In[157]:


similarity=cosine_similarity(vectors)


# In[166]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[174]:


def recommend (movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[176]:


recommend('Batman Begins')


# In[173]:


new_df.iloc[1216].title


# In[177]:


import pickle


# In[178]:


pickle.dump(new_df,open('movies.pkl','wb'))


# In[179]:


new_df['title'].values


# In[181]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[182]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





import streamlit as st
import pickle
import pandas as pd
import requests
import time  # For delay

# API headers (not used in fetch_poster below, can be removed if unused)
url = "https://api.themoviedb.org/3/account/22011294"
headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJhM2RhOTIxYTBjOTZiMGE0MmNmMmY0NmJjOGMwN2RlMSIsIm5iZiI6MTc0NzI2MjgwNS4wNTMsInN1YiI6IjY4MjUxZDU1OWRjOWQ0NDI5NjVhMzgyMCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.1wNOjj5t7IQyXi_EYJbVV224D4HAP_fBNddHPHJvqq8"
}

# Function to fetch movie poster using TMDB API
def fetch_poster(movie_id):
    try:
        response = requests.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=a3da921a0c96b0a42cf2f46bc8c07de1&language=en-US'
        )
        data = response.json()
        if 'poster_path' in data and data['poster_path']:
            return f"https://image.tmdb.org/t/p/w500/{data['poster_path']}"
        else:
            print(f"No poster found for movie ID: {movie_id}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching poster for movie ID {movie_id}: {e}")
        return None

# Function to recommend similar movies
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)

        # Delay to avoid overloading API
        time.sleep(1)

        poster = fetch_poster(movie_id)
        if poster is None:
            poster = "https://via.placeholder.com/500x750?text=No+Image"
        recommended_movies_posters.append(poster)

    return recommended_movies, recommended_movies_posters

# Load data
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Streamlit UI
st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    "Select a movie to get recommendations:",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])

st.write("Created by: Samarth Arora")
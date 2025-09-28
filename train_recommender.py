import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib
import os

# 1. Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# 2. Clean data
# Fill NaN in genres with empty string
movies['genres'] = movies['genres'].fillna('')

# 3. Convert genres into TF-IDF matrix
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# 4. Compute cosine similarity between movies
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 5. Build a mapping from movie title to index
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# 6. Function to recommend movies
def recommend(title, num_recommendations=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # exclude itself
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# 7. Save models
os.makedirs("models", exist_ok=True)
joblib.dump(cosine_sim, "models/cosine_sim.pkl")
joblib.dump(indices, "models/indices.pkl")
movies.to_csv("models/movies.csv", index=False)

print("Model training complete. Files saved in 'models/' folder.")

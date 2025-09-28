import joblib
import pandas as pd

# Load saved files
cosine_sim = joblib.load("models/cosine_sim.pkl")
indices = joblib.load("models/indices.pkl")
movies = pd.read_csv("models/movies.csv")

def recommend(title, num_recommendations=5):
    if title not in indices:
        return [f"Movie '{title}' not found in database."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # skip the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

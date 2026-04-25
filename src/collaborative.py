import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeRecommender:
    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame):
        self.ratings = ratings.copy()
        self.movies = movies.copy()
        self.user_movie_matrix = None
        self.user_similarity = None

    def fit(self):
        self.user_movie_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        self.user_similarity = cosine_similarity(self.user_movie_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )

    def recommend_for_user(self, user_id: int, top_n: int = 10, min_rating: float = 3.5) -> pd.DataFrame:
        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"User {user_id} not found.")

        similar_users = self.user_similarity[user_id].sort_values(ascending=False)
        similar_users = similar_users.drop(user_id)

        user_seen_movies = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )

        weighted_scores = {}

        for sim_user, similarity_score in similar_users.items():
            sim_user_ratings = self.ratings[
                (self.ratings["userId"] == sim_user) &
                (self.ratings["rating"] >= min_rating)
            ]

            for _, row in sim_user_ratings.iterrows():
                movie_id = row["movieId"]
                if movie_id in user_seen_movies:
                    continue

                if movie_id not in weighted_scores:
                    weighted_scores[movie_id] = {"weighted_sum": 0.0, "sim_sum": 0.0}

                weighted_scores[movie_id]["weighted_sum"] += row["rating"] * similarity_score
                weighted_scores[movie_id]["sim_sum"] += similarity_score

        recommendations = []
        for movie_id, vals in weighted_scores.items():
            if vals["sim_sum"] == 0:
                continue
            score = vals["weighted_sum"] / vals["sim_sum"]
            recommendations.append((movie_id, score))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

        rec_df = pd.DataFrame(recommendations, columns=["movieId", "collab_score"])
        rec_df = rec_df.merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left")

        return rec_df[["movieId", "title", "genres", "collab_score"]]
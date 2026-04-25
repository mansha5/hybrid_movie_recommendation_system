import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies: pd.DataFrame):
        self.movies = movies.copy()
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_index = None
        self.movieid_to_idx = None
        self.idx_to_movieid = None

    def fit(self):
        self.movies["content"] = (self.movies["genres"].fillna("") + " " + self.movies["title"].fillna(""))
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["content"])
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        self.movie_index = pd.Series(self.movies.index, index=self.movies["title"]).drop_duplicates()
        self.movieid_to_idx = pd.Series(self.movies.index, index=self.movies["movieId"]).to_dict()
        self.idx_to_movieid = pd.Series(self.movies["movieId"].values, index=self.movies.index).to_dict()

    def recommend_by_title(self, title: str, top_n: int = 10) -> pd.DataFrame:
        matches = [t for t in self.movie_index.index if title.lower() in t.lower()]
        if not matches:
            raise ValueError(f"Movie '{title}' not found in dataset.")
        title = matches[0]   # pick best match

        idx = self.movie_index[title]
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        sim_scores = sim_scores[1: top_n + 1]
        movie_indices = [i[0] for i in sim_scores]

        result = self.movies.iloc[movie_indices][["movieId", "title", "genres"]].copy()
        result["content_score"] = [score for _, score in sim_scores]
        return result.reset_index(drop=True)

    def get_similar_movies_for_user_profile(self, liked_movie_ids: list, top_n: int = 20) -> pd.DataFrame:
        """
        Build a content profile from movies the user rated highly.
        """
        valid_indices = [self.movieid_to_idx[mid] for mid in liked_movie_ids if mid in self.movieid_to_idx]
        if not valid_indices:
            return pd.DataFrame(columns=["movieId", "title", "genres", "content_score"])

        score_vector = self.similarity_matrix[valid_indices].mean(axis=0)
        scored = list(enumerate(score_vector))
        scored = sorted(scored, key=lambda x: x[1], reverse=True)

        seen_movie_ids = set(liked_movie_ids)
        recommendations = []

        for idx, score in scored:
            movie_id = self.idx_to_movieid[idx]
            if movie_id in seen_movie_ids:
                continue
            row = self.movies.iloc[idx]
            recommendations.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "genres": row["genres"],
                "content_score": float(score)
            })
            if len(recommendations) >= top_n:
                break

        return pd.DataFrame(recommendations)
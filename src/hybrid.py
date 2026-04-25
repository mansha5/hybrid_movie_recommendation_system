import pandas as pd


class HybridRecommender:
    def __init__(self, content_model, collab_model, ratings_df: pd.DataFrame):
        self.content_model = content_model
        self.collab_model = collab_model
        self.ratings_df = ratings_df.copy()

    def recommend_for_user(self, user_id: int, top_n: int = 10, alpha: float = 0.5) -> pd.DataFrame:
        """
        alpha = weight for collaborative score
        (1 - alpha) = weight for content score
        """
        user_high_rated = self.ratings_df[
            (self.ratings_df["userId"] == user_id) &
            (self.ratings_df["rating"] >= 3.5)
        ]["movieId"].tolist()

        content_recs = self.content_model.get_similar_movies_for_user_profile(
            liked_movie_ids=user_high_rated,
            top_n=200
        )

        collab_recs = self.collab_model.recommend_for_user(user_id=user_id, top_n=200)

        collab_recs = collab_recs[["movieId", "collab_score"]]

        hybrid = pd.merge(
            content_recs,
            collab_recs,
            on="movieId",
            how="outer"
        )

        hybrid["content_score"] = hybrid["content_score"].fillna(0)
        hybrid["collab_score"] = hybrid["collab_score"].fillna(0)

        hybrid = hybrid[["movieId", "content_score", "collab_score"]]

        hybrid = hybrid.merge(
        self.content_model.movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
        )

        if hybrid["content_score"].max() > 0:
            hybrid["content_score"] /= hybrid["content_score"].max()

        # normalize collab score
        if hybrid["collab_score"].max() > 0:
            hybrid["collab_score"] /= hybrid["collab_score"].max()

        if hybrid.empty:
            return hybrid

        max_content = hybrid["content_score"].max()
        max_collab = hybrid["collab_score"].max()
        
        hybrid["hybrid_score"] = (
            alpha * hybrid["collab_score"] + (1 - alpha) * hybrid["content_score"]
        )

        hybrid = hybrid.sort_values(by="hybrid_score", ascending=False)

        user_seen_movies = set(
            self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()
        )
        hybrid = hybrid[~hybrid["movieId"].isin(user_seen_movies)]

        hybrid = hybrid.sort_values("hybrid_score", ascending=False).head(top_n)
        return hybrid[["movieId", "title", "genres", "content_score", "collab_score", "hybrid_score"]].reset_index(drop=True)
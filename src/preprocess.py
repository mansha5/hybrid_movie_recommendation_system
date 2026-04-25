import pandas as pd

def load_data(movies_path: str, ratings_path: str):
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    return movies, ratings

def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    movies = movies.copy()
    movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["title"]=movies["title"].str.replace(r"\(\d{4}\)","",regex=False).str.strip()
    return movies

def get_user_movie_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Creates user-item matrix:
    rows = users, columns = movieIds, values = ratings
    """
    user_movie_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )
    return user_movie_matrix
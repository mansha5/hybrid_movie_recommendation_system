from src.preprocess import load_data, preprocess_movies
from src.content_based import ContentBasedRecommender
from src.collaborative import CollaborativeRecommender
from src.hybrid import HybridRecommender
from src.evaluate import simple_train_test_split, precision_at_k


def main():
    movies_path = "data/movies.csv"
    ratings_path = "data/ratings.csv"

    movies, ratings = load_data(movies_path, ratings_path)
    movies = preprocess_movies(movies)

    print("Movies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)

    # Split ratings for simple evaluation
    train_ratings, test_ratings = simple_train_test_split(ratings)

    # Build models on training data
    content_model = ContentBasedRecommender(movies)
    content_model.fit()

    collab_model = CollaborativeRecommender(train_ratings, movies)
    collab_model.fit()

    hybrid_model = HybridRecommender(content_model, collab_model, train_ratings)

    # Example: recommend by movie title
    title2= input("Please enter the movie:")
    print(f"\n--- Content-based recommendations for {title2}' ---")
    try:
        print(content_model.recommend_by_title(title2, top_n=5))
    except ValueError as e:
        print(e)

    # Example: recommend for a user
    sample_user_id = 1
    print(f"\n--- Hybrid recommendations for user {sample_user_id} ---")
    try:
        recs = hybrid_model.recommend_for_user(user_id=sample_user_id, top_n=20, alpha=0.3)
        print(recs)
    except ValueError as e:
        print(e)
        return

    # Simple evaluation: Precision@10 on users in test set
    precisions = []

    test_users = test_ratings["userId"].unique()
    for user_id in test_users[:50]:  # keep it fast
        user_test_movies = test_ratings[
            (test_ratings["userId"] == user_id) &
            (test_ratings["rating"] >= 3.5)
        ]["movieId"].tolist()

        if not user_test_movies:
            continue

        try:
            user_recs = hybrid_model.recommend_for_user(user_id=user_id, top_n=20, alpha=0.3)
            recommended_ids = user_recs["movieId"].tolist()
            '''print("\nUser:", user_id)
            print("Recommended:", recommended_ids)
            print("Actual:", user_test_movies)'''
            p_at_10 = precision_at_k(recommended_ids, user_test_movies, k=10)
            precisions.append(p_at_10)
        except ValueError:
            continue

    if precisions:
        avg_precision = sum(precisions) / len(precisions)
        print(f"\nAverage Precision@10: {avg_precision:.4f}")
    else:
        print("\nNot enough users for evaluation.")


if __name__ == "__main__":
    main()
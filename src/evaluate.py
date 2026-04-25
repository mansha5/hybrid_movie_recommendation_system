import pandas as pd


def precision_at_k(recommended_movie_ids, relevant_movie_ids, k=10):
    recommended_at_k = recommended_movie_ids[:k]
    if k == 0:
        return 0.0

    hits = len(set(recommended_at_k) & set(relevant_movie_ids))
    return hits / k


def simple_train_test_split(ratings: pd.DataFrame):
    """
    For each user:
    - Keep the last highly-rated movie as test
    - Rest as train
    """
    ratings = ratings.sort_values(["userId", "timestamp"])
    train_rows = []
    test_rows = []

    for user_id, group in ratings.groupby("userId"):
        if len(group) < 5:
            train_rows.extend(group.to_dict("records"))
            continue

        high_rated = group[group["rating"] >= 4.0]
        if len(high_rated) == 0:
            train_rows.extend(group.to_dict("records"))
            continue

        test_idx = high_rated.index[-1]

        for idx, row in group.iterrows():
            if idx == test_idx:
                test_rows.append(row.to_dict())
            else:
                train_rows.append(row.to_dict())

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    return train_df, test_df
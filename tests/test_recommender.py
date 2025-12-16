import pandas as pd
from src.recommender import MovieRecommender


def test_recommendation_not_empty():
    df = pd.DataFrame({
        "title": ["A", "B"],
        "metadata_text": ["action hero", "romantic drama"]
    })

    model = MovieRecommender(df)
    result = model.recommend(["A"], top_n=1)
    assert len(result) == 1

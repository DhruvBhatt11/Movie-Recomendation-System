import os
import pandas as pd
from ast import literal_eval
import kagglehub


def download_dataset():
    return kagglehub.dataset_download("rounakbanik/the-movies-dataset")


def safe_eval(val):
    if pd.isna(val):
        return []
    try:
        return literal_eval(val)
    except:
        return []


def load_movies(path):
    movies = pd.read_csv(os.path.join(path, "movies_metadata.csv"), low_memory=False)
    keywords = pd.read_csv(os.path.join(path, "keywords.csv"))

    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")

    movies = movies.dropna(subset=["id"])
    keywords = keywords.dropna(subset=["id"])

    movies = movies.merge(keywords, on="id", how="left")

    for col in ["genres", "keywords"]:
        movies[col] = movies[col].apply(safe_eval)

    movies["metadata_text"] = (
        movies["title"].fillna("") + " " +
        movies["overview"].fillna("")
    )

    return movies[["title", "metadata_text"]].drop_duplicates()

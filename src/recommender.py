import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(df["metadata_text"])
        self.index = {t.lower(): i for i, t in enumerate(df["title"])}

    def recommend(self, liked, top_n=10):
        indices = [self.index[t.lower()] for t in liked if t.lower() in self.index]
        if not indices:
            raise ValueError("No valid movie titles found.")

        user_vec = self.matrix[indices].mean(axis=0)
        scores = cosine_similarity(user_vec, self.matrix).flatten()

        for i in indices:
            scores[i] = -1

        top = np.argsort(scores)[-top_n:][::-1]
        return self.df.iloc[top][["title"]]

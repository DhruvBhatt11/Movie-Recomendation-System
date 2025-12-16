from data_loader import download_dataset, load_movies
from recommender import MovieRecommender

if __name__ == "__main__":
    path = download_dataset()
    movies = load_movies(path)

    model = MovieRecommender(movies)

    liked = input("Enter movies you like (comma separated): ").split(",")
    recs = model.recommend([m.strip() for m in liked])

    print("\nRecommended Movies:")
    for title in recs["title"]:
        print("-", title)

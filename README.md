# Movie-Recomendation-System
A content-based movie recommendation system built using Python, TF-IDF, and cosine similarity.
The system recommends movies based on the metadata of films you like, such as genres, keywords, overview, and tagline.

Features
  Content-based filtering 
  Uses TF-IDF Vectorization
  Similarity computed using Cosine Similarity
  Supports multiple liked movies as input
  Dataset downloaded automatically from Kaggle
  CLI-based interaction

Tech Stack
  Python
  Pandas, NumPy
  Scikit-learn
  KaggleHub
  Jupyter Notebook / Python Script


Dataset
  Source: Kaggle – The Movies Dataset
  Automatically downloaded using kagglehub
  Includes:
    >Movie titles
    >Genres
    >Keywords
    >Overview


Installation
1. Clone the repository
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender

2. Install dependencies
pip install pandas numpy scikit-learn kagglehub

3. Kaggle API Setup
Make sure your Kaggle API credentials are configured:
~/.kaggle/kaggle.json


Usage
Run using Python script
python movie_recomender.py

Example Input
The Matrix, Inception, Toy Story

Example Output
You might also like:
- The Matrix Reloaded
- Interstellar
- Minority Report



How It Works
  Movie metadata is merged and cleaned
  Metadata is converted into a single text feature
  TF-IDF vectorization is applied
  User profile is created by averaging liked movie vectors
  Cosine similarity finds the closest movies


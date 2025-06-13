import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

movies = pd.read_csv("data/movielens/movies.csv")

def load_data(path="data/movielens/"):
    ratings = pd.read_csv(f"{path}ratings.csv")
    return ratings

def build_sparse_matrix(ratings):
    # reindex ids to 0…N−1
    user_ids = ratings.userId.unique()
    movie_ids = ratings.movieId.unique()
    user2idx = {u:i for i,u in enumerate(user_ids)}
    movie2idx = {m:i for i,m in enumerate(movie_ids)}

    # map to indices
    row = ratings.userId.map(user2idx)
    col = ratings.movieId.map(movie2idx)
    data = ratings.rating.values

    # build sparse matrix
    mat = coo_matrix((data, (row, col)),
                     shape=(len(user_ids), len(movie_ids)))
    return mat, user2idx, movie2idx

def train_svd(mat, n_components=50):
    """
    Train SVD on the user×item rating matrix.
    Returns the fitted TruncatedSVD model, plus:
      user_factors = U @ Σ
      item_factors = V  (note: sklearn returns Vᵀ, so we’ll transpose)
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(mat)   # shape (n_users, n_components)
    item_factors = svd.components_.T        # shape (n_items, n_components)
    return svd, user_factors, item_factors
def recommend_cf(user1, user2, user2idx, movie2idx,
                 user_factors, item_factors, top_n=10):
    # look up each user’s factor vector
    u1_vec = user_factors[user2idx[user1]]
    u2_vec = user_factors[user2idx[user2]]

    # joint score = elementwise minimum (so both like it)
    # or average, or any blending you like
    joint_user = np.minimum(u1_vec, u2_vec)

    # score every movie as dot(joint_user, item_factors[i])
    scores = item_factors.dot(joint_user)

    # pick top N indices
    top_idx = np.argsort(scores)[::-1][:top_n]

    # invert movie2idx map
    idx2movie = {idx:m for m,idx in movie2idx.items()}
    recommendations = [(idx2movie[i], scores[i]) for i in top_idx]
    return recommendations

if __name__ == "__main__":
    # 1. Load & build matrix
    ratings = load_data()
    mat, user2idx, movie2idx = build_sparse_matrix(ratings)

    # 2. Train SVD CF model
    svd, user_factors, item_factors = train_svd(mat, n_components=50)
    print("SVD trained:", user_factors.shape, item_factors.shape)

    # 3. Demo: recommend for users 1 & 2
    recs = recommend_cf(1, 2, user2idx, movie2idx, user_factors, item_factors, top_n=5)

    print("Top 5 recommendations for users 1 & 2:")
    for movie_id, score in recs:
        # look up the title
        title = movies.loc[movies.movieId == movie_id, "title"].values[0]
        print(f"{title} (ID {movie_id}) — score {score:.2f}")

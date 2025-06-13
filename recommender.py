import pandas as pd
from lightfm.data import Dataset

# 1. Load the ratings CSV
ratings = pd.read_csv("data/movielens/ratings.csv")

# 2. Build ID → index mappings
unique_users  = ratings["userId"].unique()
unique_movies = ratings["movieId"].unique()
user2idx = {uid: i for i, uid in enumerate(unique_users)}
movie2idx = {mid: i for i, mid in enumerate(unique_movies)}

# 3. Initialize the LightFM Dataset
ds = Dataset()
ds.fit(users=unique_users, items=unique_movies)

# 4. Build the interactions and weights
#    This consumes triples of (userId, movieId, rating)
interactions, weights = ds.build_interactions(
    [(row.userId, row.movieId, row.rating)
     for row in ratings.itertuples()]
)

# 5. Inspect the result
print(f"Interactions matrix shape: {interactions.shape}")
print(f"Non-zero interactions:    {interactions.getnnz()}")

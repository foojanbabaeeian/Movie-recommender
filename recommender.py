#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
import faiss
import openai

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
DATA_DIR         = "data/movielens"
MOVIES_CSV       = f"{DATA_DIR}/movies.csv"
MOVIES_ONEHOT    = f"{DATA_DIR}/movies_onehot.csv"
RATINGS_CSV      = f"{DATA_DIR}/ratings.csv"
EMB_FILE         = f"{DATA_DIR}/embeddings.npy"
FAISS_INDEX_FILE = f"{DATA_DIR}/faiss_content.index"

openai.api_key   = os.getenv("OPENAI_API_KEY")
EMBED_MODEL      = "text-embedding-ada-002"

# ──────────────────────────────────────────────────────────────────────────────
# 1) LOAD & PREPARE DATA
print("Loading data…")
movies        = pd.read_csv(MOVIES_CSV)
movies_onehot = pd.read_csv(MOVIES_ONEHOT)
ratings       = pd.read_csv(RATINGS_CSV)

# Build “summary” = title + genres
genre_cols = [c for c in movies_onehot.columns if c not in ("movieId","title")]
def make_summary(r):
    gs = [g for g in genre_cols if r[g] == 1]
    return f"{r['title']}: " + ", ".join(gs)

movies_onehot["summary_text"] = movies_onehot.apply(make_summary, axis=1)

# ID → index maps
user_ids   = ratings.userId.unique()
movie_ids  = ratings.movieId.unique()
user2idx   = {u:i for i,u in enumerate(user_ids)}
movie2idx  = {m:i for i,m in enumerate(movie_ids)}

# ──────────────────────────────────────────────────────────────────────────────
# 2) COLLABORATIVE FILTERING (SVD)
def build_sparse(rdf):
    rows = rdf.userId.map(user2idx)
    cols = rdf.movieId.map(movie2idx)
    data = rdf.rating.values
    return coo_matrix((data,(rows,cols)),
                      shape=(len(user2idx),len(movie2idx)))

print("Training CF SVD…")
cf_mat        = build_sparse(ratings)
svd_model     = TruncatedSVD(n_components=50, random_state=42)
user_factors  = svd_model.fit_transform(cf_mat)   # (n_users×50)
item_factors  = svd_model.components_.T            # (n_items×50)

# ──────────────────────────────────────────────────────────────────────────────
# 3) CONTENT EMBEDDINGS + FAISS INDEX
def ensure_embeddings():
    if not os.path.exists(EMB_FILE):
        print("Embedding summaries…")
        texts, embeds = movies_onehot.summary_text.tolist(), []
        batch = 100
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            resp  = openai.Embedding.create(model=EMBED_MODEL, input=chunk)
            embeds.extend([d["embedding"] for d in resp["data"]])
        arr = np.array(embeds, dtype="float32")
        os.makedirs(os.path.dirname(EMB_FILE), exist_ok=True)
        np.save(EMB_FILE, arr)
        return arr
    else:
        print("Loading embeddings…")
        return np.load(EMB_FILE)

def ensure_faiss_index(embs):
    dim = embs.shape[1]
    if not os.path.exists(FAISS_INDEX_FILE):
        print("Building FAISS index…")
        faiss.normalize_L2(embs)
        idx = faiss.IndexFlatIP(dim)
        idx.add(embs)
        faiss.write_index(idx, FAISS_INDEX_FILE)
        return idx
    else:
        print("Loading FAISS index…")
        return faiss.read_index(FAISS_INDEX_FILE)

embs  = ensure_embeddings()
index = ensure_faiss_index(embs)

# ──────────────────────────────────────────────────────────────────────────────
# 4) RECOMMENDATION FUNCTIONS

def recommend_cf_genre(u1, u2, genre="Any", top_n=200):
    """
    Joint CF + optional genre filter, filtering out unrated movies.
    """
    # get the two users' factor vectors
    v1 = user_factors[user2idx[u1]]
    v2 = user_factors[user2idx[u2]]
    joint = np.minimum(v1, v2)

    # pick mids matching the genre (or all)
    if genre != "Any":
        mids = movies_onehot.loc[movies_onehot[genre] == 1, "movieId"]
    else:
        mids = movies_onehot["movieId"]

    # **filter out movies that never appeared in ratings**:
    mids = [mid for mid in mids if mid in movie2idx]

    # score each remaining movie
    scores = {
        mid: item_factors[movie2idx[mid]].dot(joint)
        for mid in mids
    }

    # return top_n by score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


def recommend_mood(mood, top_k=200):
    q = f"I'm feeling {mood}. Suggest a movie that fits this vibe."
    resp = openai.Embedding.create(model=EMBED_MODEL, input=[q])
    m_emb = np.array(resp["data"][0]["embedding"], dtype="float32").reshape(1,-1)
    faiss.normalize_L2(m_emb)
    _, I = index.search(m_emb, top_k)
    return movies_onehot.iloc[I[0]]["movieId"].tolist()

def recommend_joint(u1, u2, mood="Any", genre="Any", final_n=10):
    cf       = recommend_cf_genre(u1, u2, genre)
    mood_list= recommend_mood(mood) if mood!="Any" else [m for m,_ in cf]
    cf_dict  = dict(cf)
    inter    = [(m, cf_dict[m]) for m in mood_list if m in cf_dict]
    topn     = sorted(inter, key=lambda x: x[1], reverse=True)[:final_n]
    return [
        {
          "movieId": m,
          "title": movies.loc[movies.movieId==m,"title"].iloc[0],
          "score": s
        }
        for m,s in topn
    ]

# ──────────────────────────────────────────────────────────────────────────────
# 5) DEMO CLI
if __name__ == "__main__":
    uA    = int(input("User A ID (e.g. 1): ") or "1")
    uB    = int(input("User B ID (e.g. 2): ") or "2")
    mood  = input("Mood (cozy, nostalgic, Any): ") or "Any"
    genre = input("Genre (Comedy, Action, Any): ")   or "Any"

    recs  = recommend_joint(uA, uB, mood=mood, genre=genre, final_n=10)
    print(f"\nRecommendations for users {uA} & {uB}:\n")
    for r in recs:
        print(f"{r['title']} — score {r['score']:.2f}")

import tmdbsimple as tmdb
import pandas as pd
import os
from time import sleep

tmdb.API_KEY = os.getenv("TMDB_API_KEY")  # set this in your environment

def fetch_tmdb_metadata(movies_csv="data/movielens/movies.csv",
                        out_csv="data/movielens/plots.csv"):
    """
    Reads movies.csv, looks up each title on TMDb,
    and writes a CSV with columns: movieId, title, summary, poster_path.
    """
    df = pd.read_csv(movies_csv)
    results = []
    for _, row in df.iterrows():
        title, mid = row["title"], row["movieId"]
        search = tmdb.Search()
        resp = search.movie(query=title)
        if search.results:
            best = search.results[0]
            summary = best.get("overview", "")
            poster = best.get("poster_path", "")
        else:
            summary, poster = "", ""
        results.append({
            "movieId": mid,
            "title": title,
            "summary": summary,
            "poster_path": poster
        })
        sleep(0.25)  # be kind to the API

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"TMDb metadata saved to {out_csv}")

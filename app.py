import streamlit as st
import pandas as pd
from recommender import recommend_joint, user2idx, movie2idx, movies, movies_onehot

st.set_page_config(page_title="Mood-Genre Movie Recommender", layout="wide")

st.title("🎬 Mood- & Genre-Based Movie Recommender")

# Sidebar for inputs
with st.sidebar:
    st.header("Your Settings")
    userA = st.number_input("User A ID", min_value=1, max_value=max(user2idx), value=1, step=1)
    userB = st.number_input("User B ID", min_value=1, max_value=max(user2idx), value=2, step=1)
    mood  = st.selectbox("Mood", options=["Any","cozy","nostalgic","happy","thrilling","romantic"])
    genres = ["Any"] + [c for c in movies_onehot.columns if c not in ("movieId","title","summary_text")]
    genre = st.selectbox("Genre", options=genres)

    st.markdown("---")
    if st.button("Recommend"):
        st.session_state.run = True

# Main panel
if st.session_state.get("run", False):
    with st.spinner("Finding great movies…"):
        results = recommend_joint(userA, userB, mood=mood, genre=genre, final_n=10)

    st.subheader("Top Recommendations")
    df = pd.DataFrame(results)
    df = df[["title","score"]]
    st.table(df.style.format({"score":"{:.2f}"}))

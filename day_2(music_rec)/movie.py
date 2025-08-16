# movie.py — full CSV, TF-IDF (1–2 grams), title/artist weighting, relevance tweaks

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # faster cosine for tf-idf
from scipy.sparse import csr_matrix


# ----------------------------
# Text utilities (no NLTK)
# ----------------------------
_WHITESPACE = re.compile(r"\s+")
_NONALPHA = re.compile(r"[^a-z\s]")

def _clean(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = _NONALPHA.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s

def _weighted_blob(title: str, artist: str, text: str,
                   w_title: int = 3, w_artist: int = 2, w_text: int = 1) -> str:
    """
    Weight title and artist higher by repeating tokens.
    This influences TF-IDF similarly to field boosts in search engines.
    """
    t = _clean(title)
    a = _clean(artist)
    x = _clean(text)

    # Repeat tokens to simulate field boosts
    title_boost = " ".join([t] * max(1, w_title)) if t else ""
    artist_boost = " ".join([a] * max(1, w_artist)) if a else ""
    text_boost = " ".join([x] * max(1, w_text)) if x else ""

    return " ".join(part for part in (title_boost, artist_boost, text_boost) if part)


# ----------------------------
# Model loader
# ----------------------------
def load_model(repo_path: str = ".") -> Dict[str, Any]:
    """
    Load ALL rows from spotify_millsongdata.csv, build TF-IDF vectors.
    Returns a state dict used by the app.
    """
    repo = Path(repo_path)
    csv_path = repo / "spotify_millsongdata.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    # Load entire CSV
    df = pd.read_csv(csv_path)  # full read; ensure enough RAM

    # Normalize expected columns
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    # Common column names in the dataset: 'song', 'artist', 'text'
    if "title" not in df.columns and "song" in df.columns:
        df = df.rename(columns={"song": "title"})
    if "artist" not in df.columns:
        df["artist"] = "Unknown"
    if "text" not in df.columns:
        # Try to locate any text-like column
        text_like = [c for c in df.columns if c in ("lyrics", "content", "description")]
        df["text"] = df[text_like[0]].astype(str) if text_like else ""

    # Drop rows with completely missing title/artist/text
    df["title"] = df["title"].astype(str).fillna("")
    df["artist"] = df["artist"].astype(str).fillna("")
    df["text"] = df["text"].astype(str).fillna("")

    # Deduplicate (optional but helps quality)
    df = df.drop_duplicates(subset=["title", "artist", "text"]).reset_index(drop=True)

    # Build boosted blob for vectorization
    df["__blob__"] = _weighted_blob_series(df["title"], df["artist"], df["text"])

    # Prepare convenience columns
    df["__title_lc__"] = df["title"].str.lower()
    df["__artist_lc__"] = df["artist"].str.lower()

    # Vectorizer: 1–2 grams, sublinear tf, stopwords, sensible df bounds
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_df=0.5,        # ignore overly common terms
        min_df=2,          # ignore rare noise
        max_features=None  # use full vocabulary (can set e.g. 100_000 if RAM tight)
    )

    tfidf = vectorizer.fit_transform(df["__blob__"])  # sparse CSR matrix

    state = {
        "df": df,
        "vectorizer": vectorizer,
        "tfidf": tfidf,  # csr_matrix [n_songs x vocab]
    }
    return state


def _weighted_blob_series(titles: pd.Series, artists: pd.Series, texts: pd.Series) -> pd.Series:
    # Vectorized helper to avoid Python loops
    # Build strings with repetition multipliers
    t = titles.astype(str).str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    a = artists.astype(str).str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    x = texts.astype(str).str.lower().str.replace(r"[^a-z\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()

    return (
        (t + " " + t + " " + t).str.strip()   # title x3
        + " "
        + (a + " " + a).str.strip()           # artist x2
        + " "
        + x                                   # lyrics/text x1
    ).str.strip()


# ----------------------------
# Recommendation
# ----------------------------
def recommend_songs(query: str, top_k: int = 10, state: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    If query matches a song title (exact or contains), return songs similar to that song.
    Otherwise, treat query as free-text (title/artist/lyrics).
    Returns DataFrame with at least ['title','artist','score'].
    """
    if state is None:
        raise ValueError("State is None. Did you call load_model() first?")

    df: pd.DataFrame = state["df"]
    tfidf: csr_matrix = state["tfidf"]
    vectorizer: TfidfVectorizer = state["vectorizer"]

    q = (query or "").strip().lower()
    if not q:
        out = df.head(top_k).copy()
        out["score"] = 1.0
        return out[["title", "artist", "score"]]

    # 1) Title match path (closest to Spotify “this track” intent)
    exact = df.index[df["__title_lc__"] == q].tolist()
    contains = df.index[df["__title_lc__"].str.contains(re.escape(q), na=False)].tolist() if not exact else []

    cand_idx: Optional[int] = None
    if exact:
        cand_idx = exact[0]
    elif contains:
        cand_idx = contains[0]

    if cand_idx is not None:
        sims = linear_kernel(tfidf[cand_idx], tfidf).ravel()  # cosine similarities
        sims[cand_idx] = -1.0  # exclude itself
        top_idx = _topk_indices(sims, top_k)
        out = df.loc[top_idx, ["title", "artist"]].copy()
        out["score"] = sims[top_idx]

        # mild re-rank to prefer same-artist and near-title matches
        out = _rerank(out, df.iloc[cand_idx]["title"], df.iloc[cand_idx]["artist"])
        return out.reset_index(drop=True)

    # 2) Free-text search path
    q_vec = vectorizer.transform([_clean(query)])
    sims = linear_kernel(q_vec, tfidf).ravel()
    top_idx = _topk_indices(sims, top_k)
    out = df.loc[top_idx, ["title", "artist"]].copy()
    out["score"] = sims[top_idx]
    # Re-rank to slightly prefer entries containing query terms in title/artist
    out = _rerank(out, query_title_hint=query, query_artist_hint=query)
    return out.reset_index(drop=True)


# ----------------------------
# Helpers
# ----------------------------
def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Efficient top-k without full sort."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]

def _rerank(df_out: pd.DataFrame, query_title_hint: str, query_artist_hint: str) -> pd.DataFrame:
    """
    Add small boosts for same-artist and title matches to feel more Spotify-like.
    This keeps the main similarity but nudges intuitive contenders up.
    """
    qt = str(query_title_hint).lower()
    qa = str(query_artist_hint).lower()

    bonus = np.zeros(len(df_out), dtype=float)
    # prefer matching artist/title substrings a bit
    if "artist" in df_out.columns:
        m = df_out["artist"].astype(str).str.lower().str.contains(re.escape(qa), na=False)
        bonus += m.to_numpy() * 0.02
    if "title" in df_out.columns:
        m = df_out["title"].astype(str).str.lower().str.contains(re.escape(qt), na=False)
        bonus += m.to_numpy() * 0.02

    # Apply bonus and re-sort
    df_out = df_out.copy()
    df_out["score"] = df_out["score"].astype(float) + bonus
    return df_out.sort_values("score", ascending=False)

# app.py — Streamlit Music Recommender UI (Spotify-like feel)

from __future__ import annotations
import os
from pathlib import Path
import importlib.util
import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import json
import time

# =========================
# Page config + THEME CSS
# =========================
st.set_page_config(
    page_title="🎧 Music Recommender",
    page_icon="🎵",
    layout="wide",
)

# Custom CSS for style + hover cards
st.markdown("""
<style>
:root {
  --card-bg: #0f172a;
  --card-border: #1f2937;
  --chip-bg: rgba(255,255,255,0.06);
  --chip-border: rgba(255,255,255,0.15);
}

.gradient-title {
  font-size: 2.2rem;
  font-weight: 800;
  background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

@keyframes floaty {
  0% { transform: translateY(0px); }
  50% { transform: translateY(-3px); }
  100% { transform: translateY(0px); }
}

.header-card {
  border: 1px solid var(--card-border);
  border-radius: 18px;
  padding: 18px;
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(4px);
  animation: floaty 4s ease-in-out infinite;
}

.result-card {
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 12px;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
  background: rgba(255,255,255,0.03);
}
.result-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 28px rgba(0,0,0,0.15);
  border-color: #34d399;
}

.chip {
  display: inline-block;
  padding: 6px 10px;
  margin: 4px 6px 0 0;
  border-radius: 999px;
  border: 1px solid var(--chip-border);
  background: var(--chip-bg);
  font-size: 0.85rem;
  cursor: pointer;
}
.chip:hover { border-color: #60a5fa; }

.small-cap { opacity: 0.7; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# =========================
# Helpers
# =========================
def load_lottie(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def normalize_path(p: str) -> str:
    return str(Path(p).expanduser().resolve())

@st.cache_resource(show_spinner=True)
def load_module_from_path(py_file_path: str):
    if not os.path.isfile(py_file_path):
        raise FileNotFoundError(f"No file found at: {py_file_path}")
    spec = importlib.util.spec_from_file_location("music_mod", py_file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

@st.cache_resource(show_spinner=True)
def init_recommender(music_module, repo_path: str):
    if hasattr(music_module, "load_model"):
        return music_module.load_model(repo_path)
    return None

def try_recommend(music_module, query: str, k: int, state=None) -> pd.DataFrame:
    if not hasattr(music_module, "recommend_songs"):
        raise AttributeError("The module does not define recommend_songs(query, top_k, state).")
    df = music_module.recommend_songs(query=query, top_k=k, state=state)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    df.columns = df.columns.str.lower()
    return df

def add_to_history(q: str):
    if q and q.strip():
        hist = st.session_state.setdefault("history", [])
        if q not in hist:
            hist.insert(0, q)
            st.session_state["history"] = hist[:10]


# =========================
# Sidebar: Setup + Controls
# =========================
BASE = Path(__file__).parent.resolve()
with st.sidebar:
    st.header("⚙️ Setup")
    repo_path = st.text_input("Project folder", value=str(BASE))
    module_path_default = str(BASE / "movie.py")  # change to music.py if renamed
    py_file = st.text_input("Recommender module", value=module_path_default)

    k = st.slider("Number of recommendations", 5, 50, 10, step=1)
    load_btn = st.button("Load / Reload recommender", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Your module should implement:")
    st.code("load_model(repo_path='.')\nrecommend_songs(query, top_k=10, state=None)")

# =========================
# Header + Animation
# =========================
with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        lottie = load_lottie(str(BASE / "assets" / "music.json"))  # optional animation
        if lottie:
            st_lottie(lottie, height=140, speed=1.0, loop=True)
    with c2:
        st.markdown('<div class="header-card">', unsafe_allow_html=True)
        st.markdown('<div class="gradient-title">Music Recommender</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="small-cap">Type a song title, artist, or vibe '
            '(e.g., "acoustic heartbreak indie") and get recommendations.</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Normalize paths
repo_path = normalize_path(repo_path)
py_file = normalize_path(py_file)

# Load module/state
music_module = st.session_state.get("music_module")
state = st.session_state.get("state")

if load_btn:
    with st.spinner("Loading recommender…"):
        try:
            music_module = load_module_from_path(py_file)
            state = init_recommender(music_module, repo_path)
            st.session_state["music_module"] = music_module
            st.session_state["state"] = state
            st.toast("Recommender ready!", icon="🎧")
            time.sleep(0.2)
            st.success("Loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load recommender: {e}")


# =========================
# Quick prompts + History
# =========================
st.subheader("🔎 Find songs you’ll love")

qcol1, qcol2 = st.columns([3, 1])
with qcol1:
    query = st.text_input("Search by title / artist / vibes",
                          placeholder="e.g., wish you were here  •  lo-fi study  •  summer roadtrip")
with qcol2:
    go = st.button("Recommend", type="primary", use_container_width=True)

quick_prompts = [
    "acoustic heartbreak indie", "lofi chill beats", "energetic workout pop",
    "jazz evening instrumental", "melancholic piano", "happy summer roadtrip",
    "90s alternative rock", "deep house club", "nepali acoustic"
]
st.caption("Try a quick vibe:")
for p in quick_prompts:
    if st.button(p, key=f"qp_{p}", use_container_width=False):
        query = p
        st.session_state["__preset_query__"] = query
        st.rerun()

history = st.session_state.get("history", [])
if history:
    st.caption("Recent searches:")
    for h in history:
        if st.button(h, key=f"hist_{h}", use_container_width=False):
            query = h
            st.session_state["__preset_query__"] = query
            st.rerun()
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.toast("History cleared.", icon="🧹")

# Pre-fill query if chip was clicked
if "__preset_query__" in st.session_state and not query:
    query = st.session_state["__preset_query__"]

# =========================
# Results area
# =========================
if go:
    if not music_module or state is None:
        st.error("Load your recommender first from the sidebar.")
    elif not query.strip():
        st.warning("Please enter a query.")
    else:
        add_to_history(query)
        with st.spinner("Finding your tracks…"):
            try:
                recs = try_recommend(music_module, query, k, state=state)
            except Exception as e:
                st.error(f"Error during recommendation: {e}")
                st.stop()

        if recs is None or recs.empty:
            st.info("No results. Try another title/artist or vibe.")
        else:
            t1, t2 = st.tabs(["🎴 Cards", "📋 Table"])
            with t1:
                for i, row in recs.reset_index(drop=True).iterrows():
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    cimg, cmain, cside = st.columns([1, 4, 2])

                    with cimg:
                        cover = row.get("cover_url") or row.get("image") or ""
                        if isinstance(cover, str) and cover.startswith(("http://", "https://")):
                            st.image(cover, use_container_width=True)

                    with cmain:
                        title = str(row.get("title", "Unknown Title"))
                        artist = str(row.get("artist", "Unknown Artist"))
                        album = row.get("album", "")
                        st.markdown(f"**{title}**  \n*{artist}*")
                        if album:
                            st.caption(album)
                        preview = row.get("preview_url") or ""
                        if isinstance(preview, str) and preview.startswith(("http://", "https://")):
                            st.audio(preview)

                    with cside:
                        sc = row.get("score", None)
                        if sc is not None:
                            try:
                                score_fmt = f"{float(sc):.3f}"
                            except Exception:
                                score_fmt = str(sc)
                            st.metric("Match", score_fmt)
                        st.button("Add to queue ➕", key=f"queue_{i}")

                    st.markdown('</div>', unsafe_allow_html=True)
                    st.write("")

            with t2:
                st.dataframe(recs, use_container_width=True)

            st.balloons()

from __future__ import annotations

from html import escape
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
import streamlit as st

from book_recommender_app.config import get_project_paths
from book_recommender_app.data import load_books, load_ratings
from book_recommender_app.recommenders import HybridItemKNNRecommender

try:
    from st_keyup import st_keyup
except Exception:  # pragma: no cover
    st_keyup = None

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

BOOK_COL_CANDIDATES = {
    "id": ["book_id", "Book-ID", "ISBN"],
    "title": ["title", "Book-Title"],
    "author": ["author", "author_name", "Book-Author"],
    "image": ["image_url", "Image-URL-M", "Image-URL-L", "Image-URL-S"],
    "year": ["year", "Year-Of-Publication"],
    "popularity": ["explicit_ratings", "ratings_count"],
    "rating": ["avg_explicit_rating", "average_rating"],
}
RATING_OPTIONS = ["Default"] + [str(x) for x in range(1, 11)]
COMPONENT_NORM_OPTIONS = [None, "maxabs", "zscore"]


def first_present(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in columns:
            return col
    return None


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .app-subtle {
            color: #64748b;
            font-size: 0.95rem;
        }
        .seed-chip-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin: 0.25rem 0 1rem 0;
        }
        .seed-chip {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.45rem 0.75rem;
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 999px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            max-width: 100%;
        }
        .seed-chip__thumb {
            width: 40px;
            height: 56px;
            border-radius: 8px;
            overflow: hidden;
            background: #f2efe7;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
        }
        .seed-chip__thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .seed-chip__text {
            min-width: 0;
        }
        .seed-chip__title {
            font-weight: 600;
            color: #0f172a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 24rem;
        }
        .seed-chip__meta {
            font-size: 0.82rem;
            color: #64748b;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .search-shell {
            border: 1px solid rgba(15, 23, 42, 0.14);
            border-radius: 16px;
            padding: 0.9rem 1rem 0.4rem 1rem;
            background: #fafafa;
            margin-bottom: 0.9rem;
        }
        .search-row {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            background: #ffffff;
            padding: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.95rem;
            min-height: 104px;
        }
        .search-row--selected {
            background: #f7fafc;
            border-color: rgba(22, 163, 74, 0.30);
        }
        .search-row__thumb {
            width: 58px;
            height: 80px;
            border-radius: 10px;
            overflow: hidden;
            background: #f2efe7;
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
        }
        .search-row__thumb img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .search-row__body {
            min-width: 0;
            flex: 1;
        }
        .search-row__title {
            font-size: 1.04rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.35;
            margin-bottom: 0.25rem;
        }
        .search-row__author {
            color: #475569;
            margin-bottom: 0.25rem;
        }
        .search-row__meta {
            color: #64748b;
            font-size: 0.88rem;
        }
        .result-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            background: #ffffff;
            overflow: hidden;
            min-height: 420px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
        }
        .result-card__image {
            height: 190px;
            background: #f2efe7;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 1rem;
        }
        .result-card__image img {
            max-height: 158px;
            width: auto;
            max-width: 100%;
            object-fit: contain;
            box-shadow: 0 10px 20px rgba(15, 23, 42, 0.18);
            border-radius: 4px;
        }
        .result-card__body {
            padding: 1rem 1rem 1.1rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.7rem;
            flex: 1;
        }
        .result-card__title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #7f1d1d;
            line-height: 1.35;
            min-height: 3.9em;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .result-card__author {
            color: #334155;
            min-height: 2.4em;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .result-card__stats {
            color: #0f172a;
            font-size: 0.95rem;
            line-height: 1.45;
            margin-top: auto;
        }
        .seed-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 18px;
            background: #ffffff;
            overflow: hidden;
            min-height: 350px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.05);
        }
        .seed-card .result-card__image {
            height: 160px;
        }
        .seed-card .result-card__image img {
            max-height: 130px;
        }
        .seed-card .result-card__body {
            padding-bottom: 0.75rem;
        }
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stNumberInput"] label p,
        div[data-testid="stSlider"] label p,
        div[data-testid="stCheckbox"] label p {
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_clean_catalog(root: str) -> pd.DataFrame:
    paths = get_project_paths(Path(root))
    df = load_books(paths).copy()

    cols = list(df.columns)
    id_col = first_present(cols, BOOK_COL_CANDIDATES["id"])
    title_col = first_present(cols, BOOK_COL_CANDIDATES["title"])
    author_col = first_present(cols, BOOK_COL_CANDIDATES["author"])
    image_col = first_present(cols, BOOK_COL_CANDIDATES["image"])
    year_col = first_present(cols, BOOK_COL_CANDIDATES["year"])
    popularity_col = first_present(cols, BOOK_COL_CANDIDATES["popularity"])
    rating_col = first_present(cols, BOOK_COL_CANDIDATES["rating"])

    if id_col is None or title_col is None:
        raise ValueError("The cleaned catalog must contain at least book_id and title columns.")

    out = pd.DataFrame(
        {
            "book_id": df[id_col].astype(str),
            "title": df[title_col].fillna("").astype(str),
            "author": df[author_col].fillna("").astype(str) if author_col else "",
            "image_url": df[image_col].fillna("").astype(str) if image_col else "",
            "year": pd.to_numeric(df[year_col], errors="coerce") if year_col else pd.Series([pd.NA] * len(df)),
            "popularity": pd.to_numeric(df[popularity_col], errors="coerce").fillna(0) if popularity_col else pd.Series([0] * len(df)),
            "rating_meta": pd.to_numeric(df[rating_col], errors="coerce") if rating_col else pd.Series([pd.NA] * len(df)),
        }
    )

    out = out[out["title"].str.strip() != ""].copy()
    out = out.drop_duplicates(subset=["book_id"]).copy()
    out = out.sort_values(
        by=["popularity", "rating_meta", "title"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def search_books(root: str, query: str, limit: int = 18) -> pd.DataFrame:
    catalog = get_clean_catalog(root)
    q = (query or "").strip()
    if not q:
        return catalog.head(limit)

    safe_q = re.escape(q)
    starts = catalog["title"].str.contains(rf"\b{safe_q}", case=False, na=False, regex=True)
    contains = catalog["title"].str.contains(q, case=False, na=False, regex=False)

    matches = catalog.loc[contains].copy()
    if matches.empty:
        return matches

    matches["__starts"] = starts[contains].astype(int)
    matches = matches.sort_values(
        by=["__starts", "popularity", "rating_meta", "title"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    return matches.drop(columns="__starts").head(limit).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def get_selected_books(root: str, selected_ids: tuple[str, ...]) -> pd.DataFrame:
    if not selected_ids:
        return pd.DataFrame(columns=["book_id", "title", "author", "image_url", "year", "popularity", "rating_meta"])
    catalog = get_clean_catalog(root)
    selected = catalog[catalog["book_id"].isin(selected_ids)].copy()
    selected["__order"] = selected["book_id"].map({bid: i for i, bid in enumerate(selected_ids)})
    return selected.sort_values("__order").drop(columns="__order").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_model(
    root: str,
    implicit_weight: float,
    explicit_weight: float,
    neutral_rating: float,
    signal_deadzone: float,
    seed_deadzone: float,
    min_user_interactions: int,
    min_item_interactions: int,
    collaborative_weight: float,
    content_weight: float,
    component_norm: Optional[str],
    max_features: int,
    min_df: int,
    rerank_pool: int,
    top_k_by_abs: bool,
):
    paths = get_project_paths(Path(root))
    books = load_books(paths)
    ratings = load_ratings(paths)

    model = HybridItemKNNRecommender(
        implicit_weight=implicit_weight,
        explicit_weight=explicit_weight,
        neutral_rating=neutral_rating,
        signal_deadzone=signal_deadzone,
        seed_deadzone=seed_deadzone,
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions,
        collaborative_weight=collaborative_weight,
        content_weight=content_weight,
        component_norm=component_norm,
        max_features=max_features,
        min_df=min_df,
        rerank_pool=rerank_pool,
        top_k_by_abs=top_k_by_abs,
    ).fit(books, ratings)
    return model


def format_year(row: pd.Series) -> str:
    if pd.notna(row.get("year")):
        try:
            return str(int(float(row.get("year"))))
        except Exception:
            return str(row.get("year"))
    return ""


def format_popularity(value: object) -> str:
    try:
        num = int(float(value))
    except Exception:
        return ""
    if num <= 0:
        return ""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M ratings"
    if num >= 1_000:
        return f"{num / 1_000:.0f}k ratings"
    return f"{num} ratings"


def format_rating(value: object) -> str:
    try:
        return f"★ {float(value):.2f}"
    except Exception:
        return ""


def cover_html(image_url: str, css_class: str) -> str:
    image_url = (image_url or "").strip()
    if image_url:
        return f'<div class="{css_class}"><img src="{escape(image_url, quote=True)}" alt="Book cover"></div>'
    return f'<div class="{css_class}"><span style="color:#94a3b8;font-size:1.8rem;">📚</span></div>'


def build_search_row_html(row: pd.Series, selected: bool = False) -> str:
    title = escape(str(row.get("title", "")))
    author = escape(str(row.get("author", ""))) or "Unknown author"
    year = format_year(row)
    rating_text = format_rating(row.get("rating_meta"))
    pop_text = format_popularity(row.get("popularity"))
    meta = " • ".join([x for x in [year, rating_text, pop_text] if x])
    classes = "search-row search-row--selected" if selected else "search-row"
    return (
        f'<div class="{classes}">'
        f"{cover_html(str(row.get('image_url', '')), 'search-row__thumb')}"
        f'<div class="search-row__body">'
        f'<div class="search-row__title">{title}</div>'
        f'<div class="search-row__author">{author}</div>'
        f'<div class="search-row__meta">{escape(meta) if meta else "Book in catalog"}</div>'
        f"</div></div>"
    )


def build_result_card_html(row: pd.Series, *, seed_variant: bool = False) -> str:
    title = escape(str(row.get("title", "")))
    author = escape(str(row.get("author", ""))) or "Unknown author"
    year = format_year(row)

    rating_text = ""
    for col in ["avg_explicit_rating", "average_rating", "rating_meta"]:
        if col in row.index and pd.notna(row.get(col)):
            rating_text = format_rating(row.get(col))
            if rating_text:
                break

    popularity_text = ""
    for col in ["explicit_ratings", "ratings_count", "popularity"]:
        if col in row.index and pd.notna(row.get(col)):
            popularity_text = format_popularity(row.get(col))
            if popularity_text:
                break

    score_text = ""
    if "score" in row.index and pd.notna(row.get("score")):
        score_text = f"score: {float(row.get('score')):.3f}"

    collab_text = ""
    if "score_collaborative" in row.index and pd.notna(row.get("score_collaborative")):
        collab_text = f"collab: {float(row.get('score_collaborative')):.3f}"

    content_text = ""
    if "score_content" in row.index and pd.notna(row.get("score_content")):
        content_text = f"content: {float(row.get('score_content')):.3f}"

    stat_lines = [x for x in [year, rating_text, popularity_text, score_text, collab_text, content_text] if x]
    stats_html = "<br>".join(escape(x) for x in stat_lines) if stat_lines else "&nbsp;"
    wrapper_class = "seed-card" if seed_variant else "result-card"

    return (
        f'<div class="{wrapper_class}">'
        f"{cover_html(str(row.get('image_url', '')), 'result-card__image')}"
        f'<div class="result-card__body">'
        f'<div class="result-card__title">{title}</div>'
        f'<div class="result-card__author">{author}</div>'
        f'<div class="result-card__stats">{stats_html}</div>'
        f"</div></div>"
    )


def build_seed_chips_html(df: pd.DataFrame, overrides: Dict[str, float]) -> str:
    chips: List[str] = []
    for _, row in df.iterrows():
        book_id = str(row.get("book_id"))
        rating_text = f" • your rating {overrides[book_id]:.0f}/10" if book_id in overrides else ""
        title = escape(str(row.get("title", "")))
        author = escape(str(row.get("author", ""))) or "Unknown author"
        chips.append(
            '<div class="seed-chip">'
            f"{cover_html(str(row.get('image_url', '')), 'seed-chip__thumb')}"
            '<div class="seed-chip__text">'
            f'<div class="seed-chip__title">{title}</div>'
            f'<div class="seed-chip__meta">{author}{escape(rating_text)}</div>'
            "</div></div>"
        )
    return '<div class="seed-chip-wrap">' + "".join(chips) + "</div>"


def ensure_state() -> None:
    if "selected_seed_ids" not in st.session_state:
        st.session_state["selected_seed_ids"] = []
    if "seed_rating_overrides" not in st.session_state:
        st.session_state["seed_rating_overrides"] = {}
    if "recommend_results" not in st.session_state:
        st.session_state["recommend_results"] = None
    if "project_root" not in st.session_state:
        st.session_state["project_root"] = str(get_project_paths().root)
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = ""


def add_seed(book_id: str) -> None:
    current = list(st.session_state.get("selected_seed_ids", []))
    if book_id not in current:
        current.append(book_id)
        st.session_state["selected_seed_ids"] = current


def remove_seed(book_id: str) -> None:
    st.session_state["selected_seed_ids"] = [
        x for x in st.session_state.get("selected_seed_ids", []) if x != book_id
    ]
    overrides = dict(st.session_state.get("seed_rating_overrides", {}))
    overrides.pop(book_id, None)
    st.session_state["seed_rating_overrides"] = overrides


def clear_seeds() -> None:
    st.session_state["selected_seed_ids"] = []
    st.session_state["seed_rating_overrides"] = {}
    st.session_state["recommend_results"] = None


def set_seed_rating_override(book_id: str, value: str) -> None:
    overrides = dict(st.session_state.get("seed_rating_overrides", {}))
    if value == "Default":
        overrides.pop(book_id, None)
    else:
        overrides[book_id] = float(value)
    st.session_state["seed_rating_overrides"] = overrides


def get_seed_ratings(seed_ids: Sequence[str]) -> Optional[List[float]]:
    overrides = st.session_state.get("seed_rating_overrides", {})
    if not any(book_id in overrides for book_id in seed_ids):
        return None
    return [float(overrides.get(book_id, 10.0)) for book_id in seed_ids]


def run_recommendations(seed_ids: List[str], settings: Dict[str, object]) -> pd.DataFrame:
    model = load_model(
        root=str(settings["project_root"]),
        implicit_weight=float(settings["implicit_weight"]),
        explicit_weight=float(settings["explicit_weight"]),
        neutral_rating=float(settings["neutral_rating"]),
        signal_deadzone=float(settings["signal_deadzone"]),
        seed_deadzone=float(settings["seed_deadzone"]),
        min_user_interactions=int(settings["min_user_interactions"]),
        min_item_interactions=int(settings["min_item_interactions"]),
        collaborative_weight=float(settings["collaborative_weight"]),
        content_weight=float(settings["content_weight"]),
        component_norm=settings["component_norm"],
        max_features=int(settings["max_features"]),
        min_df=int(settings["min_df"]),
        rerank_pool=int(settings["rerank_pool"]),
        top_k_by_abs=bool(settings["top_k_by_abs"]),
    )

    return model.recommend_by_ids(
        seed_ids,
        n=int(settings["n_recs"]),
        seed_ratings=get_seed_ratings(seed_ids),
        exclude_input=bool(settings["exclude_input"]),
        return_components=bool(settings["show_component_scores"]),
        top_k_per_seed=int(settings["top_k_per_seed"]) if settings["top_k_per_seed"] is not None else None,
        rerank_diversity=bool(settings["rerank_diversity"]),
        candidate_pool=int(settings["candidate_pool"]) if settings["rerank_diversity"] else None,
        mmr_lambda=float(settings["mmr_lambda"]) if settings["rerank_diversity"] else None,
    )


def display_results(df: pd.DataFrame) -> None:
    st.markdown("### Recommendations")
    if df is None or df.empty:
        st.info("No recommendations returned.")
        return

    cols = st.columns(5)
    for idx, (_, row) in enumerate(df.head(20).iterrows()):
        with cols[idx % 5]:
            st.markdown(build_result_card_html(row), unsafe_allow_html=True)

    with st.expander("Show result table"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_selected_books(root: str) -> None:
    st.markdown("### Selected seed books")
    selected_ids = list(st.session_state.get("selected_seed_ids", []))
    if not selected_ids:
        st.info("Search for books below and add them here as recommendation seeds.")
        return

    selected_df = get_selected_books(root, tuple(selected_ids))
    if selected_df.empty:
        st.warning("Selected IDs are no longer present in the loaded catalog.")
        return

    overrides = st.session_state.get("seed_rating_overrides", {})
    st.markdown(build_seed_chips_html(selected_df, overrides), unsafe_allow_html=True)

    top_left, top_mid, top_right = st.columns([3, 4, 1])
    with top_left:
        st.caption(f"{len(selected_df)} book(s) selected")
    with top_mid:
        st.caption(
            "Optional seed ratings are used as signed query weights in the hybrid model. "
            "Leave a book on Default if you want all selected books to contribute equally."
        )
    with top_right:
        if st.button("Clear all", use_container_width=True):
            clear_seeds()
            st.rerun()

    grid = st.columns(4)
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        book_id = str(row["book_id"])
        with grid[idx % 4]:
            st.markdown(build_result_card_html(row, seed_variant=True), unsafe_allow_html=True)
            current_value = str(int(overrides[book_id])) if book_id in overrides else "Default"
            selected_value = st.selectbox(
                "Optional seed rating",
                options=RATING_OPTIONS,
                index=RATING_OPTIONS.index(current_value),
                key=f"seed_rating_select_{book_id}",
            )
            set_seed_rating_override(book_id, selected_value)
            if st.button("Remove", key=f"remove_{book_id}", use_container_width=True):
                remove_seed(book_id)
                st.rerun()


def render_search_results(root: str) -> None:
    st.markdown("### Find books by title")
    st.markdown('<div class="search-shell">', unsafe_allow_html=True)

    if st_keyup is not None:
        query = st_keyup(
            "Type part of a title",
            value=st.session_state.get("search_query", ""),
            key="search_query_live",
            debounce=250,
            placeholder="Dune, Hobbit, Harry Potter, ...",
        )
        st.session_state["search_query"] = query
    else:
        query = st.text_input(
            "Type part of a title",
            value=st.session_state.get("search_query", ""),
            placeholder="Dune, Hobbit, Harry Potter, ...",
            key="search_query",
        )
        st.caption(
            "Install streamlit-keyup for live updates on every keystroke. "
            "Without it, the search still works with the standard Streamlit input."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    if query.strip() and len(query.strip()) < 2:
        st.caption("Type at least 2 characters to search.")
        return

    matches = search_books(root, query, limit=18)
    if query.strip() and matches.empty:
        st.info("No matching titles found.")
        return

    if not query.strip():
        st.caption("Showing a few popular books from the catalog. Start typing to narrow the list.")

    selected_ids = set(st.session_state.get("selected_seed_ids", []))
    for _, row in matches.iterrows():
        book_id = str(row["book_id"])
        left, right = st.columns([8, 1.3])
        with left:
            st.markdown(build_search_row_html(row, selected=book_id in selected_ids), unsafe_allow_html=True)
        with right:
            if book_id in selected_ids:
                if st.button("Added", key=f"added_{book_id}", use_container_width=True):
                    remove_seed(book_id)
                    st.rerun()
            else:
                if st.button("Add", key=f"add_{book_id}", use_container_width=True):
                    add_seed(book_id)
                    st.rerun()


def sidebar_settings() -> Dict[str, object]:
    detected_root = st.session_state.get("project_root", str(get_project_paths().root))

    with st.sidebar:
        st.header("Hybrid settings")
        project_root = st.text_input("Project root", value=detected_root)
        st.session_state["project_root"] = project_root

        n_recs = st.slider("Number of recommendations", min_value=1, max_value=50, value=12)
        exclude_input = st.checkbox("Exclude selected books from results", value=True)
        show_component_scores = st.checkbox("Include component scores in output", value=False)

        st.markdown("### Hybrid mix")
        collaborative_weight = st.slider("Collaborative weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)
        default_content = round(max(0.0, 1.0 - collaborative_weight), 2)
        content_weight = st.slider("Content weight", min_value=0.0, max_value=1.0, value=float(default_content), step=0.05)
        component_norm = st.selectbox(
            "Component normalization",
            options=COMPONENT_NORM_OPTIONS,
            format_func=lambda x: "none" if x is None else str(x),
            index=1,
        )
        top_k_per_seed = st.slider("Top similar items kept per seed", min_value=50, max_value=2000, value=500, step=50)
        top_k_by_abs = st.checkbox("Keep strongest similarities by absolute value", value=True)

        st.markdown("### Diversity reranking")
        rerank_diversity = st.checkbox("Enable diversity reranking", value=False)
        candidate_pool = st.slider("Candidate pool", min_value=20, max_value=500, value=150, step=10)
        mmr_lambda = st.slider("MMR lambda", min_value=0.0, max_value=1.0, value=0.80, step=0.05)
        rerank_pool = st.slider("Default rerank pool", min_value=20, max_value=500, value=150, step=10)

        with st.expander("Advanced fit parameters"):
            implicit_weight = st.number_input("Implicit weight", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            explicit_weight = st.number_input("Explicit weight", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            neutral_rating = st.number_input("Neutral rating", min_value=1.0, max_value=10.0, value=6.0, step=0.5)
            signal_deadzone = st.number_input("Signal deadzone", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            seed_deadzone = st.number_input("Seed deadzone", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            min_user_interactions = st.number_input("Min user interactions", min_value=1, max_value=100, value=2, step=1)
            min_item_interactions = st.number_input("Min item interactions", min_value=1, max_value=100, value=3, step=1)
            min_df = st.number_input("TF-IDF min_df", min_value=1, max_value=50, value=2, step=1)
            max_features = st.number_input("TF-IDF max_features", min_value=1000, max_value=500000, value=100000, step=1000)

    return {
        "project_root": project_root,
        "n_recs": n_recs,
        "exclude_input": exclude_input,
        "show_component_scores": show_component_scores,
        "collaborative_weight": collaborative_weight,
        "content_weight": content_weight,
        "component_norm": component_norm,
        "top_k_per_seed": top_k_per_seed,
        "top_k_by_abs": top_k_by_abs,
        "rerank_diversity": rerank_diversity,
        "candidate_pool": candidate_pool,
        "mmr_lambda": mmr_lambda,
        "rerank_pool": rerank_pool,
        "implicit_weight": implicit_weight,
        "explicit_weight": explicit_weight,
        "neutral_rating": neutral_rating,
        "signal_deadzone": signal_deadzone,
        "seed_deadzone": seed_deadzone,
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "min_df": min_df,
        "max_features": max_features,
    }


def main() -> None:
    ensure_state()
    inject_css()
    settings = sidebar_settings()

    st.title("📚 Book Recommender")
    st.write(
        "Search books by title, add them as seed books, optionally set seed ratings, "
        "and generate recommendations with the current HybridItemKNNRecommender."
    )

    project_root = str(settings["project_root"])

    render_selected_books(project_root)
    render_search_results(project_root)

    st.markdown("### Generate recommendations")
    if st.button("Recommend", type="primary", use_container_width=False):
        seed_ids = list(st.session_state.get("selected_seed_ids", []))
        if not seed_ids:
            st.warning("Please add at least one book first.")
            st.stop()
        try:
            with st.spinner("Running HybridItemKNNRecommender..."):
                st.session_state["recommend_results"] = run_recommendations(seed_ids, settings)
        except Exception as exc:
            st.session_state["recommend_results"] = exc

    payload = st.session_state.get("recommend_results")
    if payload is None:
        return
    if isinstance(payload, Exception):
        st.error(f"Hybrid recommender failed: {payload}")
        return
    display_results(payload)


if __name__ == "__main__":
    main()
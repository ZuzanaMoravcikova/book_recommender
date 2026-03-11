
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

ArrayLikeId = Union[str, int, float, Sequence[Union[str, int, float]]]
ArrayLikeTitle = Union[str, Sequence[str]]


def normalize_book_id_value(value: Any) -> str:
    """
    Convert book identifiers to a stable string form.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except Exception:
        pass
    return text


def normalize_book_id_series(s: pd.Series) -> pd.Series:
    """
    Normalize all book identifiers in a pandas Series.
    """
    return s.map(normalize_book_id_value)


def normalize_title(text: Any) -> str:
    """
    Convert a title into a normalized lookup form.
    """
    return re.sub(r"\s+", " ", str(text).strip().lower())


def clean_text_series(s: pd.Series) -> pd.Series:
    """
    Lowercase and whitespace-normalize a text Series.
    """
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_book_catalog(
    books: pd.DataFrame,
    *,
    id_col: str = "book_id",
    title_col: str = "title",
    author_col: str = "author",
    publisher_col: str = "publisher",
    isbn_col: str = "isbn",
    year_col: str = "year",
    image_url_col: str = "image_url",
    popularity_col: str = "explicit_ratings",
    rating_meta_col: str = "avg_explicit_rating",
) -> pd.DataFrame:
    """
    Create one clean metadata row per book.

    The catalog produced here is the shared reference table used by all models.
    It keeps one row per ``book_id``, ensures required metadata columns exist,
    creates a normalized title key for title-based lookup, and sorts duplicate
    records so that the best-supported row is kept.
    """
    required = [id_col, title_col]
    missing = [c for c in required if c not in books.columns]
    if missing:
        raise ValueError(f"`books` is missing required columns: {missing}")

    df = books.copy()
    df = df[df[id_col].notna()].copy()
    df[id_col] = normalize_book_id_series(df[id_col])

    for col in [title_col, author_col, publisher_col, isbn_col, image_url_col]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if year_col not in df.columns:
        df[year_col] = np.nan
    df[year_col] = df[year_col].astype("Int32")

    if popularity_col not in df.columns:
        df[popularity_col] = 0
    df[popularity_col] = df[popularity_col].astype("Int32").fillna(0)

    if rating_meta_col not in df.columns:
        df[rating_meta_col] = np.nan
    df[rating_meta_col] = pd.to_numeric(df[rating_meta_col], errors="coerce")

    df["__title_key"] = df[title_col].map(normalize_title)

    df = (
        df.sort_values(
            by=[popularity_col, rating_meta_col, title_col],
            ascending=[False, False, True],
            na_position="last",
        )
        .drop_duplicates(subset=[id_col], keep="first")
        .reset_index(drop=True)
    )
    return df


def prepare_explicit_interactions(
    ratings: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "book_id",
    rating_col: str = "book_rating",
    explicit_min: float = 1.0,
    explicit_max: float = 10.0,
    min_user_ratings: int = 10,
    min_item_ratings: int = 10,
) -> pd.DataFrame:
    """
    Prepare the explicit-feedback interaction table used by explicit models.

    Processing steps:
    1. keep only the required columns
    2. drop rows with missing user or item IDs
    3. convert ratings to numeric values
    4. keep explicit ratings above ``explicit_min``
    5. merge repeated user-item rows by averaging their ratings
    6. remove users and items that do not satisfy the minimum
       interaction thresholds
    """
    required = [user_col, item_col, rating_col]
    missing = [c for c in required if c not in ratings.columns]
    if missing:
        raise ValueError(f"`ratings` is missing required columns: {missing}")

    df = ratings[[user_col, item_col, rating_col]].copy()
    df = df[df[user_col].notna() & df[item_col].notna()].copy()

    df[user_col] = df[user_col].astype(str)
    df[item_col] = normalize_book_id_series(df[item_col])
    df[rating_col] = df[rating_col].astype("Int32")

    df = df[df[rating_col] >= explicit_min].copy()
    df = df[df[rating_col] <= explicit_max].copy()
    df = df.groupby([user_col, item_col], as_index=False)[rating_col].mean()


    keep_users = df[user_col].value_counts()
    keep_users = keep_users[keep_users >= min_user_ratings].index

    keep_items = df[item_col].value_counts()
    keep_items = keep_items[keep_items >= min_item_ratings].index

    df = df[df[user_col].isin(keep_users)].copy()
    df = df[df[item_col].isin(keep_items)].copy()

    return df.reset_index(drop=True)


class SeedBookRecommenderBase:
    """
    Base class for recommenders that generate suggestions from one or more seed books.

    This class implements the shared workflow used by all seed-based recommenders.
    It is responsible for:

    - building and storing the shared book catalog
    - validating that the model has been fitted
    - normalizing seed book IDs
    - resolving seed titles to IDs
    - mapping catalog IDs to row positions
    - formatting final recommendation outputs
    """

    def __init__(
        self,
        *,
        id_col: str = "book_id",
        title_col: str = "title",
        author_col: str = "author",
        publisher_col: str = "publisher",
        isbn_col: str = "isbn",
        year_col: str = "year",
        image_url_col: str = "image_url",
        popularity_col: str = "explicit_ratings",
        rating_meta_col: str = "avg_explicit_rating",
    ):
        self.id_col = id_col
        self.title_col = title_col
        self.author_col = author_col
        self.publisher_col = publisher_col
        self.isbn_col = isbn_col
        self.year_col = year_col
        self.image_url_col = image_url_col
        self.popularity_col = popularity_col
        self.rating_meta_col = rating_meta_col

        self.catalog_: Optional[pd.DataFrame] = None
        self.catalog_id_to_pos_: Optional[Dict[str, int]] = None

    def _set_catalog(self, books: pd.DataFrame) -> None:
        """
        Build and store the shared catalog used for lookup and output formatting.
        """
        self.catalog_ = build_book_catalog(
            books,
            id_col=self.id_col,
            title_col=self.title_col,
            author_col=self.author_col,
            publisher_col=self.publisher_col,
            isbn_col=self.isbn_col,
            year_col=self.year_col,
            image_url_col=self.image_url_col,
            popularity_col=self.popularity_col,
            rating_meta_col=self.rating_meta_col,
        )
        self.catalog_id_to_pos_ = {
            str(book_id): idx for idx, book_id in enumerate(self.catalog_[self.id_col].tolist())
        }

    def _check_is_fitted(self) -> None:
        """
        Raise an error if the model has not been fitted yet.
        """
        if self.catalog_ is None:
            raise RuntimeError("The recommender is not fitted yet. Call fit(...) first.")

    def _normalize_seed_ids(self, seed_book_ids: ArrayLikeId) -> List[str]:
        """
        Normalize one or more user-provided seed book IDs.

        Parameters
        ----------
        seed_book_ids : ArrayLikeId
            A single seed ID or a sequence of seed IDs. Accepted forms include
            strings, integers, floats, or sequences of those values.

        Returns
        -------
        List[str]
            A list of normalized non-empty book ID strings.

        """
        if isinstance(seed_book_ids, (str, int, float, np.integer, np.floating)):
            seed_book_ids = [seed_book_ids]
        out = [normalize_book_id_value(x) for x in seed_book_ids]
        out = [x for x in out if x]
        if not out:
            raise ValueError("No valid seed book IDs were provided.")
        return out

    def _resolve_titles_to_ids(
        self,
        seed_titles: ArrayLikeTitle,
        *,
        top_k_per_title: int = 1,
    ) -> List[str]:
        """
        Resolve one or more seed titles to catalog book IDs.

        Titles are normalized before matching. If multiple catalog rows share
        the same normalized title, the method returns the best-supported match
        or matches according to catalog popularity and rating metadata.

        Parameters
        ----------
        seed_titles : ArrayLikeTitle
            A single title or a sequence of titles to resolve.

        top_k_per_title : int, default=1
            Maximum number of matching IDs to return for each provided title.

        Returns
        -------
        List[str]
            A list of resolved catalog book IDs.
        """
        self._check_is_fitted()
        titles = [seed_titles] if isinstance(seed_titles, str) else list(seed_titles)
        out: List[str] = []

        for title in titles:
            key = normalize_title(title)
            matches = self.catalog_[self.catalog_["__title_key"] == key]
            if matches.empty:
                continue

            matches = matches.sort_values(
                by=[self.popularity_col, self.rating_meta_col, self.title_col],
                ascending=[False, False, True],
                na_position="last",
            )
            out.extend(matches[self.id_col].astype(str).head(top_k_per_title).tolist())

        if not out:
            raise ValueError("None of the provided titles were found in the catalog.")
        return out

    def _catalog_positions(self, book_ids: Sequence[str]) -> List[int]:
        """
        Convert catalog book IDs into integer row positions.

        Parameters
        ----------
        book_ids : Sequence[str]
            Sequence of normalized book IDs.

        Returns
        -------
        List[int]
            Row positions of the IDs that exist in the current catalog.
        """
        return [
            self.catalog_id_to_pos_[book_id]
            for book_id in book_ids
            if book_id in self.catalog_id_to_pos_
        ]

    def _build_output(
        self,
        scores: np.ndarray,
        *,
        seed_ids: Sequence[str],
        n: int,
        exclude_input: bool = True,
        component_scores: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Convert a catalog-aligned score vector into a recommendation table.

        Parameters
        ----------
        scores : np.ndarray
            Score vector aligned with the rows of ``self.catalog_``.
            Each position corresponds to one catalog book.

        seed_ids : Sequence[str]
            Normalized seed book IDs used to generate the recommendations.

        n : int
            Number of recommendations to return.

        exclude_input : bool, default=True
            Whether to remove the seed books themselves from the output.

        component_scores : dict[str, np.ndarray] or None, default=None
            Optional dictionary of additional score vectors aligned with the
            catalog, for example separate collaborative and content components.

        Returns
        -------
        pd.DataFrame
            A formatted recommendation table containing metadata columns,
            the final score, and optionally the component scores.

        """
        self._check_is_fitted()

        scores = np.asarray(scores, dtype=np.float64).copy()
        if len(scores) != len(self.catalog_):
            raise ValueError("Scores must be aligned with the catalog.")

        if exclude_input:
            for pos in self._catalog_positions(seed_ids):
                scores[pos] = -np.inf

        order = np.argsort(-scores)
        valid = np.isfinite(scores[order])
        order = order[valid][:n]

        cols = [
            self.id_col,
            self.title_col,
            self.author_col,
            self.publisher_col,
            self.year_col,
            self.isbn_col,
            self.image_url_col,
            self.popularity_col,
            self.rating_meta_col,
        ]
        cols = [c for c in cols if c in self.catalog_.columns]

        out = self.catalog_.iloc[order][cols].copy()
        out["score"] = scores[order].round(3)

        if component_scores:
            for name, arr in component_scores.items():
                out[name] = np.asarray(arr)[order]

        return out.reset_index(drop=True)

    def recommend_by_ids(
        self,
        seed_book_ids: ArrayLikeId,
        *,
        n: int = 10,
        seed_ratings: Optional[Sequence[float]] = None,
        exclude_input: bool = True,
        return_components: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Generate recommendations from one or more seed book IDs.

        Parameters
        ----------
        seed_book_ids : ArrayLikeId
            A single seed ID or a sequence of seed IDs.

        n : int, default=10
            Number of recommendations to return.

        seed_ratings : Sequence[float] or None, default=None
            Optional ratings associated with the seed books. Subclasses may use
            them to weight the influence of individual seeds.

        exclude_input : bool, default=True
            Whether to exclude the seed books from the final recommendation list.

        return_components : bool, default=False
            Whether to include model-specific component score columns in the
            output when available.

        **kwargs : Any
            Additional keyword arguments forwarded to the subclass scoring method.

        Returns
        -------
        pd.DataFrame
            A formatted recommendation table.
        """
        self._check_is_fitted()
        seed_ids = self._normalize_seed_ids(seed_book_ids)
        scores, components = self._score_from_seed_ids(
            seed_ids,
            seed_ratings=seed_ratings,
            return_components=return_components,
            **kwargs,
        )
        return self._build_output(
            scores,
            seed_ids=seed_ids,
            n=n,
            exclude_input=exclude_input,
            component_scores=components if return_components else None,
        )

    def recommend_by_title(
        self,
        seed_titles: ArrayLikeTitle,
        *,
        n: int = 10,
        seed_ratings: Optional[Sequence[float]] = None,
        exclude_input: bool = True,
        top_k_per_title: int = 1,
        return_components: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Recommend books from one or more seed titles.
        """
        seed_ids = self._resolve_titles_to_ids(seed_titles, top_k_per_title=top_k_per_title)
        return self.recommend_by_ids(
            seed_ids,
            n=n,
            seed_ratings=seed_ratings,
            exclude_input=exclude_input,
            return_components=return_components,
            **kwargs,
        )

    def _score_from_seed_ids(
        self,
        seed_ids: Sequence[str],
        *,
        seed_ratings: Optional[Sequence[float]] = None,
        return_components: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Subclasses must return a score vector aligned with ``self.catalog_``.
        """
        raise NotImplementedError



class HybridItemKNNRecommender(SeedBookRecommenderBase):
    """
    Hybrid item-to-item recommender that combines collaborative and content signals.

    The model builds two item representations:

    1. a collaborative item-user representation derived from signed rating signals
    2. a content representation derived from TF-IDF features of book metadata

    For a recommendation query, each seed book contributes both collaborative and
    content similarity scores. These scores are aggregated across seeds, optionally
    normalized per query, and then combined into a final hybrid score.

    The class also supports:
    - signed query-time seed weights from seed ratings
    - keeping only the strongest similarities per seed
    - optional Maximal Marginal Relevance reranking for diversity

    Parameters
    ----------
    user_col : str, default="user_id"
    item_col : str, default="book_id"
    rating_col : str, default="book_rating"

    implicit_weight : float, default=1.0
        Strength assigned to implicit interactions encoded as rating 0.

    explicit_weight : float, default=2.5
        Scaling factor applied to explicit ratings after centering them around

    neutral_rating : float, default=6.0
        Rating value treated as neutral. Ratings above it become positive and
        ratings below it become negative.

    signal_deadzone : float, default=0.1
        Threshold used when converting training ratings into signed collaborative
        signals. Values whose absolute centered magnitude is below this threshold
        are set to zero.

    seed_deadzone : float, default=0.1
        Threshold used when converting query-time seed ratings into signed seed
        weights. Values near neutral are suppressed.

    min_user_interactions : int, default=2
        Minimum number of retained interactions required for a user to remain in
        the collaborative training table.

    min_item_interactions : int, default=3
        Minimum number of retained interactions required for an item to remain in
        the collaborative training table.

    collaborative_weight : float, default=0.70
        Weight of the collaborative component in the final hybrid score.

    content_weight : float, default=0.30
        Weight of the content component in the final hybrid score.

    component_norm : {"maxabs", "zscore"} or None, default="maxabs"
        Optional per-query normalization applied separately to collaborative and
        content score vectors before they are mixed.

    max_features : int, default=100_000
        Maximum number of TF-IDF features retained by the vectorizer.

    min_df : int, default=2
        Minimum document frequency used by the TF-IDF vectorizer.

    ngram_range : tuple[int, int], default=(1, 2)
        Lower and upper n-gram range used by the TF-IDF vectorizer.

    diversity_lambda : float or None, default=0.80
        Relevance-diversity tradeoff used by MMR reranking.
        Higher values favor relevance more strongly.
        If None, diversity reranking is disabled.

    rerank_pool : int, default=150
        Number of top base candidates considered during MMR reranking.

    top_k_by_abs : bool, default=True
        If True, keep the strongest similarities by absolute value when applying
        the per-seed top-k mask.
    """

    def __init__(
        self,
        *,
        user_col: str = "user_id",
        item_col: str = "book_id",
        rating_col: str = "book_rating",
        implicit_weight: float = 1.0,
        explicit_weight: float = 2.5,
        neutral_rating: float = 6.0,
        signal_deadzone: float = 0.5,
        seed_deadzone: float = 0.5,
        min_user_interactions: int = 2,
        min_item_interactions: int = 3,
        collaborative_weight: float = 0.70,
        content_weight: float = 0.30,
        component_norm: Optional[str] = "maxabs",
        max_features: int = 100_000,
        min_df: int = 2,
        ngram_range: Tuple[int, int] = (1, 2),
        diversity_lambda: Optional[float] = 0.80,
        rerank_pool: int = 150,
        top_k_by_abs: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

        self.implicit_weight = implicit_weight
        self.explicit_weight = explicit_weight
        self.neutral_rating = neutral_rating
        self.signal_deadzone = signal_deadzone
        self.seed_deadzone = seed_deadzone

        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions

        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.component_norm = component_norm

        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range

        self.diversity_lambda = diversity_lambda
        self.rerank_pool = rerank_pool
        self.top_k_by_abs = top_k_by_abs

        self.item_user_norm_: Optional[csr_matrix] = None
        self.content_norm_: Optional[csr_matrix] = None
        self.vectorizer_: Optional[TfidfVectorizer] = None

    def _centered_signed_value(
        self,
        rating: float,
        *,
        deadzone: float,
    ) -> float:
        """
        Convert a rating on the 1 to 10 scale into a signed value around the
        configured neutral rating.

        Ratings above `neutral_rating` become positive, ratings below it become
        negative, and the output is scaled to approximately the interval [-1, 1].
        A deadzone is then applied so values close to neutral become exactly zero.
        """
        rating = float(np.clip(rating, 1.0, 10.0))

        if rating >= self.neutral_rating:
            scale = max(10.0 - self.neutral_rating, 1e-9)
            value = (rating - self.neutral_rating) / scale
        else:
            scale = max(self.neutral_rating - 1.0, 1e-9)
            value = (rating - self.neutral_rating) / scale

        if abs(value) < deadzone:
            value = 0.0
        return float(value)

    def _rating_to_signal(self, rating: float) -> float:
        """
        Convert a raw training rating into a signed collaborative signal.

        Explicit ratings are centered around `neutral_rating` and scaled by
        `explicit_weight`. Implicit interactions encoded as rating 0 are mapped
        to a weak positive signal controlled by `implicit_weight`.
        """
        if rating == 0:
            return float(self.implicit_weight)

        signed = self._centered_signed_value(rating, deadzone=self.signal_deadzone)
        return float(self.explicit_weight * signed)

    def _build_content_corpus(
        self,
        catalog: pd.DataFrame,
        title_weight: int = 3,
        author_weight: int = 2,
        publisher_weight: int = 1,
    ) -> pd.Series:
        """
        Build the metadata text corpus used by the TF-IDF content model.

        The corpus is formed by concatenating selected metadata fields and
        repeating them according to their weights. Repetition makes a field more
        influential in the resulting TF-IDF representation.
        """
        title_text = catalog[self.title_col].fillna("").astype(str)
        author_text = catalog[self.author_col].fillna("").astype(str)
        publisher_text = catalog[self.publisher_col].fillna("").astype(str)

        corpus = (
            ("title " + title_text + " ") * title_weight
            + ("author " + author_text + " ") * author_weight
            + ("publisher " + publisher_text + " ") * publisher_weight
        )
        return corpus.str.replace(r"\s+", " ", regex=True).str.strip()

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "HybridItemKNNRecommender":
        """
        Fit the hybrid recommender on books and ratings.

        Collaborative part:
        - build a signed item-user matrix from ratings
        - row-normalize it so cosine similarity can be used

        Content part:
        - build TF-IDF vectors from metadata
        - row-normalize them

        Parameters
        ----------
        books : pd.DataFrame
            Book metadata table used to build the shared catalog and content model.

        ratings : pd.DataFrame
            User-item-rating interaction table used to build the collaborative
            item-user matrix.

        Returns
        -------
        HybridItemKNNRecommender
            The fitted recommender instance.
        """
        self._set_catalog(books)
        catalog = self.catalog_

        df = ratings[[self.user_col, self.item_col, self.rating_col]].copy()
        df = df[df[self.user_col].notna() & df[self.item_col].notna()].copy()

        df[self.user_col] = df[self.user_col].astype(str)
        df[self.item_col] = normalize_book_id_series(df[self.item_col])
        df[self.rating_col] = pd.to_numeric(df[self.rating_col], errors="coerce").fillna(0.0)

        valid_ids = set(catalog[self.id_col].astype(str))
        df = df[df[self.item_col].isin(valid_ids)].copy()

        df = (
            df.groupby([self.user_col, self.item_col], as_index=False)[self.rating_col]
            .mean()
        )

        df["signal"] = df[self.rating_col].map(self._rating_to_signal)

        df = df[np.abs(df["signal"]) > 0].copy()

        changed = True
        while changed and not df.empty:
            before = len(df)

            if self.min_user_interactions > 1:
                keep_users = df[self.user_col].value_counts()
                keep_users = keep_users[keep_users >= self.min_user_interactions].index
                df = df[df[self.user_col].isin(keep_users)].copy()

            if self.min_item_interactions > 1:
                keep_items = df[self.item_col].value_counts()
                keep_items = keep_items[keep_items >= self.min_item_interactions].index
                df = df[df[self.item_col].isin(keep_items)].copy()

            changed = len(df) != before

        if df.empty:
            raise ValueError("No interactions remain after cleaning the hybrid feedback table.")

        user_codes, user_uniques = pd.factorize(df[self.user_col], sort=True)
        item_pos = df[self.item_col].map(self.catalog_id_to_pos_).to_numpy()
        values = df["signal"].to_numpy(np.float32)

        item_user = csr_matrix(
            (values, (item_pos, user_codes)),
            shape=(len(catalog), len(user_uniques)),
            dtype=np.float32,
        )
        self.item_user_norm_ = normalize(item_user, norm="l2", axis=1)

        corpus = self._build_content_corpus(catalog)
        self.vectorizer_ = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        content = self.vectorizer_.fit_transform(corpus)
        self.content_norm_ = normalize(content, norm="l2", axis=1)

        return self

    def _prepare_valid_seeds(
        self,
        seed_ids: Sequence[str],
        seed_ratings: Optional[Sequence[float]],
    ) -> Tuple[List[str], np.ndarray, Optional[np.ndarray]]:
        """
        Keep only seed books that exist in the catalog and preserve alignment with
        the optional seed ratings.

        Parameters
        ----------
        seed_ids : Sequence[str]
            Normalized seed book IDs provided by the user.

        seed_ratings : Sequence[float] or None
            Optional ratings associated with the seed books.

        Returns
        -------
        tuple[list[str], np.ndarray, np.ndarray or None]
            A tuple containing:
            1. valid seed IDs
            2. catalog row positions of those seeds
            3. aligned seed ratings if provided, otherwise None
        """
        if seed_ratings is not None and len(seed_ratings) != len(seed_ids):
            raise ValueError("seed_ratings must have the same length as seed_book_ids.")

        valid_ids: List[str] = []
        valid_positions: List[int] = []
        valid_ratings: List[float] = []

        for i, book_id in enumerate(seed_ids):
            pos = self.catalog_id_to_pos_.get(book_id)
            if pos is None:
                continue
            valid_ids.append(book_id)
            valid_positions.append(pos)
            if seed_ratings is not None:
                valid_ratings.append(float(seed_ratings[i]))

        if not valid_positions:
            raise KeyError("None of the provided seed book IDs exist in the catalog.")

        rating_arr = None
        if seed_ratings is not None:
            rating_arr = np.asarray(valid_ratings, dtype=np.float32)

        return valid_ids, np.asarray(valid_positions, dtype=np.int64), rating_arr

    def _seed_weights(self, seed_ratings: Optional[Sequence[float]], n_seeds: int) -> np.ndarray:
        """
        Convert optional query-time seed ratings into signed seed weights.

        - if `seed_ratings` is None, all seeds receive weight 1
        - liked seeds get positive weights
        - disliked seeds get negative weights
        - neutral seeds get weights near 0
        - if all weights are near zero, the method falls back to equal positive weights

        Parameters
        ----------
        seed_ratings : Sequence[float] or None
            Optional ratings given to the seed books by the user.

        n_seeds : int
            Number of valid seed books.

        Returns
        -------
        np.ndarray
            One weight per seed book.

        """
        if seed_ratings is None:
            return np.ones(n_seeds, dtype=np.float32)

        r = np.asarray(seed_ratings, dtype=np.float32)
        w = np.array(
            [self._centered_signed_value(x, deadzone=self.seed_deadzone) for x in r],
            dtype=np.float32,
        )

        # If all weights are near zero, fall back to equal positive seeds.
        if np.sum(np.abs(w)) < 1e-9:
            w = np.ones(n_seeds, dtype=np.float32)

        return w

    def _normalize_component_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize a component score vector before hybrid mixing.

        Supported modes:
        - None: no normalization
        - "maxabs": divide by the maximum absolute value
        - "zscore": subtract mean and divide by standard deviation

        Parameters
        ----------
        x : np.ndarray
            Raw component score vector aligned with the catalog.

        Returns
        -------
        np.ndarray
            Normalized score vector.
        """
        if self.component_norm is None:
            return x

        x = np.asarray(x, dtype=np.float32).copy()
        mask = np.isfinite(x) & (x != 0)

        if not np.any(mask):
            return x

        if self.component_norm == "maxabs":
            scale = float(np.max(np.abs(x[mask])))
            return x / max(scale, 1e-9)

        if self.component_norm == "zscore":
            mu = float(np.mean(x[mask]))
            sigma = float(np.std(x[mask]))
            return (x - mu) / max(sigma, 1e-9)

        raise ValueError(f"Unknown component_norm: {self.component_norm}")

    def _apply_top_k_mask(self, sim: np.ndarray, top_k: Optional[int]) -> np.ndarray:
        """
        Keep only the strongest similarities for a single seed.

        Parameters
        ----------
        sim : np.ndarray
            Similarity vector from one seed book to all catalog books.

        top_k : int or None
            Number of strongest similarities to keep.
            If None, the full vector is preserved.

        Returns
        -------
        np.ndarray
            Similarity vector where only the selected entries are kept and all
            others are set to zero.
        """
        if top_k is None or top_k >= len(sim):
            return sim

        sim = sim.copy()
        k = int(top_k)

        if self.top_k_by_abs:
            idx = np.argpartition(-np.abs(sim), k - 1)[:k]
        else:
            idx = np.argpartition(-sim, k - 1)[:k]

        out = np.zeros_like(sim)
        out[idx] = sim[idx]
        return out

    def _score_from_seed_ids(
        self,
        seed_ids: Sequence[str],
        *,
        seed_ratings: Optional[Sequence[float]] = None,
        return_components: bool = False,
        top_k_per_seed: Optional[int] = 500,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Compute hybrid recommendation scores for all catalog items from one or
        more seed books.

        For each seed:
        - collaborative cosine similarity is computed against all items
        - content cosine similarity is computed against all items
        - both vectors may be top-k masked
        - both are weighted by the signed seed weight

        The aggregated components are divided by the sum of absolute seed weights,
        optionally normalized, and then linearly combined.

        Parameters
        ----------
        seed_ids : Sequence[str]
            Seed book IDs used to generate recommendations.

        seed_ratings : Sequence[float] or None, default=None
            Optional query-time ratings for the seed books. These are converted
            into signed seed weights.

        return_components : bool, default=False
            Whether to also return separate collaborative and content score vectors.

        top_k_per_seed : int or None, default=500
            Number of strongest similarities kept per seed and per component.
            If None, the full similarity vectors are used.

        Returns
        -------
        tuple[np.ndarray, dict[str, np.ndarray] or None]
            A pair containing:
            1. the final hybrid score vector aligned with the catalog
            2. an optional dictionary with component score vectors
        """
        self._check_is_fitted()

        valid_seed_ids, seed_positions, valid_seed_ratings = self._prepare_valid_seeds(
            seed_ids, seed_ratings
        )

        weights = self._seed_weights(valid_seed_ratings, len(valid_seed_ids))

        collab = np.zeros(len(self.catalog_), dtype=np.float32)
        content = np.zeros(len(self.catalog_), dtype=np.float32)

        for weight, pos in zip(weights, seed_positions):
            if weight == 0:
                continue

            sim_collab = linear_kernel(self.item_user_norm_[pos], self.item_user_norm_).ravel()
            sim_content = linear_kernel(self.content_norm_[pos], self.content_norm_).ravel()

            sim_collab = self._apply_top_k_mask(sim_collab, top_k_per_seed)
            sim_content = self._apply_top_k_mask(sim_content, top_k_per_seed)

            collab += weight * sim_collab.astype(np.float32)
            content += weight * sim_content.astype(np.float32)

        denom = float(np.sum(np.abs(weights))) if len(weights) else 1.0
        collab /= max(denom, 1e-9)
        content /= max(denom, 1e-9)

        collab = self._normalize_component_scores(collab)
        content = self._normalize_component_scores(content)

        final_scores = (
            self.collaborative_weight * collab
            + self.content_weight * content
        )

        components = None
        if return_components:
            components = {
                "score_collaborative": collab,
                "score_content": content,
                "score_hybrid_base": final_scores.copy(),
            }

        return final_scores, components

    def _mmr_rerank_scores(
        self,
        base_scores: np.ndarray,
        *,
        seed_ids: Sequence[str],
        n: int,
        exclude_input: bool = True,
        candidate_pool: Optional[int] = None,
        lambda_relevance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Greedy MMR reranking using content similarity as the redundancy term.

        MMR selects items greedily. At each step it chooses the candidate with the
        best tradeoff between:
        - relevance according to the base score
        - novelty with respect to already selected items

        Redundancy is measured using content cosine similarity

        High lambda -> more relevance
        Low lambda -> more diversity
        """
        scores = np.asarray(base_scores, dtype=np.float64).copy()

        if exclude_input:
            for pos in self._catalog_positions(seed_ids):
                scores[pos] = -np.inf

        order = np.argsort(-scores)
        order = order[np.isfinite(scores[order])]

        pool_size = candidate_pool or self.rerank_pool
        pool = order[:pool_size].tolist()
        if not pool:
            return scores

        lam = self.diversity_lambda if lambda_relevance is None else lambda_relevance
        if lam is None:
            return scores

        rel = scores[pool].copy()
        rel = rel / max(np.max(np.abs(rel)), 1e-9)
        rel_map = {pos: float(val) for pos, val in zip(pool, rel)}

        selected: List[int] = []

        while pool and len(selected) < n:
            best_pos = None
            best_val = -np.inf

            for cand in pool:
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = float(
                        linear_kernel(
                            self.content_norm_[cand],
                            self.content_norm_[selected],
                        ).max()
                    )

                mmr_val = lam * rel_map[cand] - (1.0 - lam) * redundancy

                if mmr_val > best_val:
                    best_val = mmr_val
                    best_pos = cand

            selected.append(best_pos)
            pool.remove(best_pos)

        reranked = np.full_like(scores, -np.inf, dtype=np.float64)

        for rank, pos in enumerate(selected):
            reranked[pos] = float(len(selected) - rank) + 1e-3 * rel_map[pos]

        return reranked

    def recommend_by_ids(
        self,
        seed_book_ids: ArrayLikeId,
        *,
        n: int = 10,
        seed_ratings: Optional[Sequence[float]] = None,
        exclude_input: bool = True,
        return_components: bool = False,
        rerank_diversity: bool = False,
        candidate_pool: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Recommend books from one or more seed book IDs.

        Parameters
        ----------
        seed_book_ids : ArrayLikeId
            One seed ID or a sequence of seed IDs.

        n : int, default=10
            Number of recommendations to return.

        seed_ratings : Sequence[float] or None, default=None
            Optional query-time ratings for the seed books.

        exclude_input : bool, default=True
            Whether to exclude the seed books themselves from the output.

        return_components : bool, default=False
            Whether to include separate score components in the result table.

        rerank_diversity : bool, default=False
            Whether to apply MMR reranking on top of the hybrid base score.

        candidate_pool : int or None, default=None
            Number of top hybrid candidates considered for diversity reranking.
            If None, `self.rerank_pool` is used.

        mmr_lambda : float or None, default=None
            Relevance-diversity tradeoff used by MMR reranking.
            If None, `self.diversity_lambda` is used.

        **kwargs : Any
            Additional keyword arguments forwarded to `_score_from_seed_ids`,
            for example `top_k_per_seed`.

        Returns
        -------
        pd.DataFrame
            Final recommendation table with metadata and score columns.

        """
        self._check_is_fitted()
        seed_ids = self._normalize_seed_ids(seed_book_ids)

        scores, components = self._score_from_seed_ids(
            seed_ids,
            seed_ratings=seed_ratings,
            return_components=return_components,
            **kwargs,
        )

        if rerank_diversity:
            base_scores = scores.copy()
            scores = self._mmr_rerank_scores(
                base_scores,
                seed_ids=seed_ids,
                n=n,
                exclude_input=exclude_input,
                candidate_pool=candidate_pool,
                lambda_relevance=mmr_lambda,
            )
            if return_components:
                components = {} if components is None else dict(components)
                components["score_reranked"] = scores
                components["score_hybrid_base"] = base_scores

        return self._build_output(
            scores,
            seed_ids=seed_ids,
            n=n,
            exclude_input=exclude_input,
            component_scores=components if return_components else None,
        )
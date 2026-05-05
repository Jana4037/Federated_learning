"""
data_preparation.py — Data loading, exploration, and federated partitioning.

Run this script once before training:
    python data_preparation.py

What this script does (step by step)
-------------------------------------
1.  Load the three Kaggle CSV files (Books, Users, Ratings).
2.  Basic cleaning: drop nulls, fix encoding issues, keep only explicit ratings.
3.  Exploratory analysis: print statistics and save plots to data/plots/.
4.  Extract the city name from the Users.Location field.
5.  Keep only the top-N most populated cities so each node has enough data.
6.  Build a global vocabulary: map every user-ID and ISBN to a compact
    integer index (required by the embedding layers).
7.  Simulate bookstore heterogeneity:
      a. Designate ~10% of books as "global" (available in every store).
      b. Assign a random subset of the remaining books to each city node.
      c. Drop ratings for books not in a node's assortment.
8.  Optionally skew the rating scale per node to simulate stores with
    harsher or more generous customers.
9.  Save one CSV per node to data/partitions/ plus a meta.csv that records
    global vocabulary sizes and node IDs.

Dataset download:
    https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data
Place Books.csv, Users.csv, Ratings.csv inside the data/ folder.
"""

import os
import random
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
TOP_N_CITIES         = 5
MIN_RATINGS_PER_CITY = 200
GLOBAL_BOOK_FRACTION = 0.10
LOCAL_BOOKS_PER_NODE = 8_000
SKEW_DISTRIBUTIONS   = True
RANDOM_SEED          = 42


# ── Logger ────────────────────────────────────────────────────────────────────

def _get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

logger = _get_logger("data_preparation")


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── Step 1: Load raw CSVs ─────────────────────────────────────────────────────

def _detect_separator(path: str) -> str:
    with open(path, "r", encoding="ISO-8859-1", errors="replace") as f:
        first_line = f.readline()
    n_commas     = first_line.count(",")
    n_semicolons = first_line.count(";")
    sep = ";" if n_semicolons > n_commas else ","
    logger.info("Detected separator for %s: %r  (commas=%d, semicolons=%d)",
                os.path.basename(path), sep, n_commas, n_semicolons)
    return sep


def load_raw_data(data_dir: str = "data"):
    logger.info("Loading raw CSV files from '%s' ...", data_dir)
    paths = {
        "books":   os.path.join(data_dir, "Books.csv"),
        "users":   os.path.join(data_dir, "Users.csv"),
        "ratings": os.path.join(data_dir, "Ratings.csv"),
    }
    for p in paths.values():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Could not find: {p}\n"
                "Download from: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data\n"
                "and place Books.csv, Users.csv, Ratings.csv in the data/ folder."
            )
    common = dict(encoding="ISO-8859-1", on_bad_lines="skip", low_memory=False)
    books   = pd.read_csv(paths["books"],   sep=_detect_separator(paths["books"]),   dtype=str, **common)
    users   = pd.read_csv(paths["users"],   sep=_detect_separator(paths["users"]),   **common)
    ratings = pd.read_csv(paths["ratings"], sep=_detect_separator(paths["ratings"]), **common)
    logger.info("Loaded %d books | %d users | %d ratings", len(books), len(users), len(ratings))
    return books, users, ratings


# ── Step 2: Basic cleaning ────────────────────────────────────────────────────

def clean_data(books, users, ratings):
    logger.info("Cleaning data ...")
    for df in (books, users, ratings):
        df.columns = [c.strip().replace("-", "_").lower() for c in df.columns]

    books = books.rename(columns={"book_title": "title", "book_author": "author",
                                  "year_of_publication": "year"})
    if "book_rating" in ratings.columns:
        ratings = ratings.rename(columns={"book_rating": "rating"})
    for df, col in [(ratings, "user_id"), (users, "user_id")]:
        if col not in df.columns:
            for alt in ["userid", "user id", "uid"]:
                if alt in df.columns:
                    df.rename(columns={alt: col}, inplace=True)
                    break

    ratings["rating"]  = pd.to_numeric(ratings["rating"],  errors="coerce")
    ratings["user_id"] = pd.to_numeric(ratings["user_id"], errors="coerce")
    users["user_id"]   = pd.to_numeric(users["user_id"],   errors="coerce")
    users["age"]       = pd.to_numeric(users["age"],       errors="coerce")

    ratings.dropna(subset=["user_id", "isbn", "rating"], inplace=True)
    users.dropna(subset=["user_id", "location"],          inplace=True)
    books.dropna(subset=["isbn", "title"],                 inplace=True)

    ratings = ratings[(ratings["rating"] >= 1) & (ratings["rating"] <= 10)].copy()
    ratings["rating"] = ratings["rating"].astype(float)
    ratings.drop_duplicates(subset=["user_id", "isbn"], keep="last", inplace=True)

    logger.info("After cleaning: %d explicit ratings remain", len(ratings))
    return books, users, ratings


# ── Step 3: Exploratory data analysis ────────────────────────────────────────

def explore_data(books, users, ratings, plots_dir: str = "data/plots"):
    os.makedirs(plots_dir, exist_ok=True)
    n_users, n_books, n_ratings = (ratings["user_id"].nunique(),
                                   ratings["isbn"].nunique(), len(ratings))
    sparsity = 1 - n_ratings / (n_users * n_books)

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Unique users       : {n_users:>10,}")
    print(f"  Unique books       : {n_books:>10,}")
    print(f"  Explicit ratings   : {n_ratings:>10,}")
    print(f"  Matrix sparsity    : {sparsity:.4%}")
    print(f"  Avg ratings/user   : {n_ratings/n_users:>10.2f}")
    print(f"  Avg ratings/book   : {n_ratings/n_books:>10.2f}")
    print("=" * 60 + "\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Rating Distribution (1-10)"); ax.set_xlabel("Rating"); ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "rating_distribution.png"), dpi=120)
    plt.close(fig)

    rpu = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rpu, bins=50, color="coral", log=True)
    ax.set_title("Ratings per User (log scale)"); ax.set_xlabel("# ratings"); ax.set_ylabel("# users (log)")
    fig.tight_layout(); fig.savefig(os.path.join(plots_dir, "ratings_per_user.png"), dpi=120)
    plt.close(fig)

    return {"n_users": n_users, "n_books": n_books, "n_ratings": n_ratings, "sparsity": sparsity}


# ── Step 4: Extract city ──────────────────────────────────────────────────────

def _extract_city(location: str) -> str:
    if not isinstance(location, str):
        return "unknown"
    parts = [p.strip().lower() for p in location.split(",")]
    city  = parts[0] if parts else "unknown"
    return "unknown" if (len(city) < 2 or city.isnumeric()) else city


# ── Step 5: Build city partitions ─────────────────────────────────────────────

def build_city_partitions(users, ratings, top_n=TOP_N_CITIES, min_ratings=MIN_RATINGS_PER_CITY):
    users = users.copy()
    users["city"] = users["location"].apply(_extract_city)
    users = users[users["city"] != "unknown"]

    merged = ratings.merge(users[["user_id", "city"]], on="user_id", how="inner")
    city_counts = merged.groupby("city").size().sort_values(ascending=False)
    city_counts = city_counts[city_counts >= min_ratings]
    top_cities  = city_counts.head(top_n).index.tolist()
    merged      = merged[merged["city"].isin(top_cities)].copy()

    logger.info("Selected top-%d cities — %d ratings in scope.", len(top_cities), len(merged))
    return merged, top_cities


# ── Step 6: Global integer vocabulary ────────────────────────────────────────

def build_vocabulary(merged: pd.DataFrame):
    all_users = sorted(merged["user_id"].unique())
    all_books = sorted(merged["isbn"].unique())
    user2idx  = {uid: i  for i, uid  in enumerate(all_users)}
    book2idx  = {isbn: i for i, isbn in enumerate(all_books)}
    merged    = merged.copy()
    merged["user_idx"] = merged["user_id"].map(user2idx)
    merged["book_idx"] = merged["isbn"].map(book2idx)
    logger.info("Vocabulary: %d users, %d books", len(user2idx), len(book2idx))
    return user2idx, book2idx, merged


# ── Step 7: Simulate assortment heterogeneity ─────────────────────────────────

def simulate_assortment(merged, book2idx, global_fraction=GLOBAL_BOOK_FRACTION,
                         local_books_per_node=LOCAL_BOOKS_PER_NODE, seed=RANDOM_SEED):
    rng = random.Random(seed)
    book_counts  = merged.groupby("isbn").size().sort_values(ascending=False)
    n_global     = max(1, int(len(book2idx) * global_fraction))
    global_books = set(book_counts.head(n_global).index.tolist())
    remaining    = [b for b in book2idx if b not in global_books]

    city_assortment = {
        city: global_books | set(rng.sample(remaining, min(local_books_per_node, len(remaining))))
        for city in merged["city"].unique()
    }
    mask     = merged.apply(lambda r: r["isbn"] in city_assortment.get(r["city"], set()), axis=1)
    filtered = merged[mask].copy()
    logger.info("After assortment filtering: %d ratings (%.1f%% of original)",
                len(filtered), 100 * len(filtered) / len(merged))
    return filtered


# ── Step 8: Skew rating distributions ────────────────────────────────────────

def skew_ratings(merged, seed=RANDOM_SEED):
    rng      = random.Random(seed)
    cities   = merged["city"].unique().tolist()
    personas = ["generous", "neutral", "critical"]
    city_p   = {city: rng.choice(personas) for city in cities}

    def _shift(row):
        p = city_p.get(row["city"], "neutral")
        r = row["rating"]
        if p == "generous": return min(10.0, r + 1.0)
        if p == "critical":  return max(1.0,  r - 1.0)
        return r

    merged = merged.copy()
    merged["rating"] = merged.apply(_shift, axis=1)
    return merged, city_p


# ── Step 9: Save partitions ───────────────────────────────────────────────────

def save_partitions(merged, user2idx, book2idx, top_cities, city_persona,
                    partitions_dir="data/partitions", plots_dir="data/plots"):
    os.makedirs(partitions_dir, exist_ok=True)
    os.makedirs(plots_dir,      exist_ok=True)

    node_ids, rating_counts = [], []
    for city in top_cities:
        city_df = merged[merged["city"] == city][["user_idx","book_idx","rating","city"]].copy()
        if len(city_df) == 0:
            continue
        safe    = city.replace(" ","_").replace("/","_").replace("\\","_")
        node_id = f"city_{safe}"
        node_ids.append(node_id); rating_counts.append(len(city_df))
        city_df.to_csv(os.path.join(partitions_dir, f"{node_id}.csv"), index=False)
        logger.info("  %-40s  %6d ratings  persona=%s",
                    node_id, len(city_df), city_persona.get(city, "neutral"))

    pd.DataFrame({"num_users": [len(user2idx)], "num_books": [len(book2idx)],
                  "node_ids": ["|".join(node_ids)]}).to_csv(
        os.path.join(partitions_dir, "meta.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(node_ids, rating_counts, color="mediumseagreen")
    ax.set_xticks(range(len(node_ids)))
    ax.set_xticklabels(node_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of ratings"); ax.set_title("Ratings per federated node")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ratings_per_node.png"), dpi=120)
    plt.close(fig)

    logger.info("Saved %d node partitions + meta.csv", len(node_ids))
    return node_ids


# ── Shared data-loading helpers ───────────────────────────────────────────────

def load_partition(node_id: str, data_dir: str = "data/partitions") -> pd.DataFrame:
    path = os.path.join(data_dir, f"{node_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Partition not found: {path}. Run data_preparation.py first.")
    return pd.read_csv(path)


def load_all_partitions(data_dir: str = "data/partitions") -> pd.DataFrame:
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "meta.csv"]
    if not files:
        raise FileNotFoundError(f"No partitions in {data_dir}. Run data_preparation.py first.")
    return pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in files], ignore_index=True)


def load_meta(data_dir: str = "data/partitions") -> dict:
    path = os.path.join(data_dir, "meta.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta.csv not found. Run data_preparation.py first.")
    df = pd.read_csv(path)
    return {
        "num_users": int(df["num_users"].iloc[0]),
        "num_books": int(df["num_books"].iloc[0]),
        "node_ids":  df["node_ids"].iloc[0].split("|"),
    }


def train_test_split_df(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42):
    df    = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split = int(len(df) * (1 - test_fraction))
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    set_seed(RANDOM_SEED)
    print("\n" + "=" * 60)
    print("  FEDERATED BOOK RECOMMENDER — Data Preparation")
    print("=" * 60 + "\n")

    books, users, ratings       = load_raw_data("data")
    books, users, ratings       = clean_data(books, users, ratings)
    _                           = explore_data(books, users, ratings)
    merged, top_cities          = build_city_partitions(users, ratings)
    user2idx, book2idx, merged  = build_vocabulary(merged)
    merged                      = simulate_assortment(merged, book2idx)
    if SKEW_DISTRIBUTIONS:
        merged, city_persona    = skew_ratings(merged)
    else:
        city_persona            = {city: "neutral" for city in top_cities}
    node_ids = save_partitions(merged, user2idx, book2idx, top_cities, city_persona)

    print("\n" + "=" * 60)
    print(f"  Done! Created {len(node_ids)} partitions in data/partitions/")
    print("  Next steps:")
    print("    python centralized.py    ← train centralized baseline")
    print("    python simulation.py     ← run federated simulation")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
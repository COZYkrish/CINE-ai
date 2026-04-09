"""
CineAI - Movie Recommendation System
Run locally: python app.py
Then open: http://127.0.0.1:5000
"""

import ast
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = BASE_DIR / "data" / "tmdb_5000_credits.csv"
CSV_PATH = Path(os.getenv("CSV_PATH", DEFAULT_CSV_PATH))

app = Flask(__name__, static_folder=".")

model_state = {
    "ready": False,
    "error": None,
    "df": None,
    "indices": None,
    "tfidf_matrix": None,
}


def extract_names(json_str, key="name", limit=5):
    try:
        items = ast.literal_eval(str(json_str))
        return [item[key] for item in items[:limit]]
    except Exception:
        return []


def extract_director(crew_json):
    try:
        crew = ast.literal_eval(str(crew_json))
        return [p["name"] for p in crew if p.get("job") == "Director"][:1]
    except Exception:
        return []


def load_model_state():
    if model_state["ready"] or model_state["error"]:
        return

    try:
        print("CineAI starting up...")
        print(f"Loading dataset from: {CSV_PATH}")

        df = pd.read_csv(CSV_PATH)
        df["cast_names"] = df["cast"].apply(lambda x: extract_names(x, "name", 5))
        df["director"] = df["crew"].apply(extract_director)
        df["soup"] = df.apply(
            lambda r: " ".join(r["cast_names"]) + " " + " ".join(r["director"]),
            axis=1,
        )

        print("Building TF-IDF matrix...")
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df["soup"])

        num_clusters = min(50, max(8, len(df) // 100))
        print(f"Training MiniBatchKMeans with {num_clusters} clusters...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=42,
            n_init=3,
            batch_size=1024,
        )
        df["cluster"] = kmeans.fit_predict(tfidf_matrix)
        indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

        model_state["df"] = df
        model_state["indices"] = indices
        model_state["tfidf_matrix"] = tfidf_matrix
        model_state["ready"] = True
        print(f"{len(df)} movies indexed. Server ready.\n")
    except Exception as exc:
        model_state["error"] = str(exc)
        print(f"Startup failed: {exc}")


def get_model_or_error():
    load_model_state()
    if not model_state["ready"]:
        message = (
            "Dataset not available. Set CSV_PATH or add the dataset to "
            f"{DEFAULT_CSV_PATH}."
        )
        if model_state["error"]:
            message = f"{message} Details: {model_state['error']}"
        return None, (jsonify({"error": message}), 503)
    return model_state, None


def find_title_match(query):
    indices = model_state["indices"]

    if query in indices:
        return query

    for title in indices.index:
        if title.lower() == query.lower():
            return title

    for title in indices.index:
        if query.lower() in title.lower():
            return title

    return None


def build_result_row(row, score):
    cast = row["cast_names"] if isinstance(row["cast_names"], list) else []
    director = row["director"][0] if row["director"] else "Unknown"
    return {
        "title": row["title"],
        "cast": cast[:4],
        "director": director,
        "score": round(float(score), 4),
        "id": int(row["movie_id"]),
    }


def recommend_by_kmeans(idx, limit=10):
    df = model_state["df"]
    tfidf_matrix = model_state["tfidf_matrix"]
    cluster_id = df.iloc[idx]["cluster"]
    cluster_indices = df.index[df["cluster"] == cluster_id].tolist()

    query_vector = tfidf_matrix[idx]
    same_cluster = [i for i in cluster_indices if i != idx]
    ranked_cluster = []

    if same_cluster:
        same_cluster_scores = cosine_similarity(
            query_vector, tfidf_matrix[same_cluster]
        )[0]
        ranked_cluster = sorted(
            zip(same_cluster, same_cluster_scores), key=lambda x: x[1], reverse=True
        )

    if len(ranked_cluster) < limit:
        other_indices = [
            i for i in range(len(df)) if i != idx and i not in set(cluster_indices)
        ]
        other_scores = cosine_similarity(query_vector, tfidf_matrix[other_indices])[0]
        fallback = sorted(
            zip(other_indices, other_scores), key=lambda x: x[1], reverse=True
        )
        ranked_cluster.extend(fallback[: limit - len(ranked_cluster)])

    return ranked_cluster[:limit]


@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/movies")
def get_movies():
    state, error_response = get_model_or_error()
    if error_response:
        return error_response

    df = state["df"]
    movies = []
    for _, row in df.iterrows():
        cast = row["cast_names"] if isinstance(row["cast_names"], list) else []
        director = row["director"][0] if row["director"] else "Unknown"
        movies.append(
            {
                "id": int(row["movie_id"]),
                "title": row["title"],
                "cast": cast[:4],
                "director": director,
            }
        )
    return jsonify(movies)


@app.route("/api/recommend")
def recommend():
    state, error_response = get_model_or_error()
    if error_response:
        return error_response

    df = state["df"]
    indices = state["indices"]
    tfidf_matrix = state["tfidf_matrix"]
    query = request.args.get("title", "").strip()
    method = request.args.get("method", "kmeans").strip().lower()
    if not query:
        return jsonify({"error": "No title provided"}), 400

    matched_title = find_title_match(query)
    if not matched_title:
        return jsonify({"error": f'Movie "{query}" not found'}), 404

    idx = indices[matched_title]
    if method == "cosine":
        scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
        ranked_items = sorted(
            ((i, score) for i, score in enumerate(scores) if i != idx),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
    else:
        ranked_items = recommend_by_kmeans(idx)

    queried_row = df.loc[idx]
    queried_cast = (
        queried_row["cast_names"] if isinstance(queried_row["cast_names"], list) else []
    )
    queried_dir = queried_row["director"][0] if queried_row["director"] else "Unknown"

    results = []
    for i, score in ranked_items:
        results.append(build_result_row(df.iloc[i], score))

    return jsonify(
        {
            "query": matched_title,
            "queried_cast": queried_cast[:4],
            "queried_dir": queried_dir,
            "method": "cosine" if method == "cosine" else "kmeans",
            "cluster": int(df.iloc[idx]["cluster"]),
            "results": results,
        }
    )


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    print(f"Expected dataset path: {CSV_PATH}")
    print("Open your browser at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, port=5000)

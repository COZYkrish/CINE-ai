"""
CineAI - Movie Recommendation System
Run: python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder=".")

CSV_PATH = r"D:\Desktop\tmdb_5000_credits.csv"

print("CineAI starting up...")
print(f"Loading: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)


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


df["cast_names"] = df["cast"].apply(lambda x: extract_names(x, "name", 5))
df["director"] = df["crew"].apply(extract_director)
df["soup"] = df.apply(
    lambda r: " ".join(r["cast_names"]) + " " + " ".join(r["director"]), axis=1
)

print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["soup"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
num_clusters = min(50, max(8, len(df) // 100))
print(f"Training K-Means with {num_clusters} clusters...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
print(f"{len(df)} movies indexed. Server ready.\n")


def find_title_match(query):
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
    cluster_id = df.iloc[idx]["cluster"]
    cluster_indices = df.index[df["cluster"] == cluster_id].tolist()

    ranked_cluster = sorted(
        ((i, cosine_sim[idx][i]) for i in cluster_indices if i != idx),
        key=lambda x: x[1],
        reverse=True,
    )

    if len(ranked_cluster) < limit:
        fallback = sorted(
            (
                (i, cosine_sim[idx][i])
                for i in range(len(df))
                if i != idx and i not in cluster_indices
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        ranked_cluster.extend(fallback[: limit - len(ranked_cluster)])

    return ranked_cluster[:limit]


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/api/movies")
def get_movies():
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
    query = request.args.get("title", "").strip()
    method = request.args.get("method", "kmeans").strip().lower()
    if not query:
        return jsonify({"error": "No title provided"}), 400

    matched_title = find_title_match(query)
    if not matched_title:
        return jsonify({"error": f'Movie "{query}" not found'}), 404

    idx = indices[matched_title]
    if method == "cosine":
        ranked_items = sorted(
            enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True
        )[1:11]
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
    print("Open your browser at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(debug=False, port=5000)

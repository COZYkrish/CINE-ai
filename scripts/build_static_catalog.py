import ast
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "data" / "tmdb_5000_credits.csv"
OUTPUT_JSON = ROOT / "data" / "catalog.json"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def extract_names(json_str, key="name", limit=5):
    try:
        items = ast.literal_eval(str(json_str))
    except (ValueError, SyntaxError):
        return []
    return [item.get(key) for item in items[:limit] if item.get(key)]


def extract_director(crew_json):
    try:
        crew = ast.literal_eval(str(crew_json))
    except (ValueError, SyntaxError):
        return ""

    for person in crew:
        if person.get("job") == "Director":
            return person.get("name", "")
    return ""


def build_result_row(row, score):
    return {
        "id": int(row["movie_id"]),
        "title": row["title"],
        "cast": row["cast_names"][:4],
        "director": row["director"] or "Unknown",
        "score": round(float(score), 4),
    }


def recommend_by_kmeans(df, tfidf_matrix, idx, limit=10):
    cluster_id = df.iloc[idx]["cluster"]
    cluster_indices = df.index[df["cluster"] == cluster_id].tolist()
    query_vector = tfidf_matrix[idx]

    same_cluster = [i for i in cluster_indices if i != idx]
    ranked = []

    if same_cluster:
        same_cluster_scores = cosine_similarity(query_vector, tfidf_matrix[same_cluster])[0]
        ranked = sorted(
            zip(same_cluster, same_cluster_scores),
            key=lambda item: item[1],
            reverse=True,
        )

    if len(ranked) < limit:
        cluster_index_set = set(cluster_indices)
        other_indices = [i for i in range(len(df)) if i != idx and i not in cluster_index_set]
        other_scores = cosine_similarity(query_vector, tfidf_matrix[other_indices])[0]
        fallback = sorted(
            zip(other_indices, other_scores),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked.extend(fallback[: limit - len(ranked)])

    return ranked[:limit]


def main():
    df = pd.read_csv(SOURCE_CSV)
    df["cast_names"] = df["cast"].apply(lambda value: extract_names(value, "name", 5))
    df["director"] = df["crew"].apply(extract_director)
    df["soup"] = df.apply(
        lambda row: " ".join(row["cast_names"]) + " " + row["director"],
        axis=1,
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    num_clusters = min(50, max(8, len(df) // 100))
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters,
        random_state=42,
        n_init=3,
        batch_size=1024,
    )
    df["cluster"] = kmeans.fit_predict(tfidf_matrix)

    movies = []
    for idx, row in df.iterrows():
        ranked_items = recommend_by_kmeans(df, tfidf_matrix, idx)
        movies.append(
            {
                "id": int(row["movie_id"]),
                "title": row["title"],
                "cast": row["cast_names"][:4],
                "director": row["director"] or "Unknown",
                "cluster": int(row["cluster"]),
                "results": [
                    build_result_row(df.iloc[result_idx], score)
                    for result_idx, score in ranked_items
                ],
            }
        )

    payload = {
        "movie_count": len(movies),
        "method": "kmeans",
        "pipeline": "TF-IDF vectorization + cosine scoring + K-Means clustering",
        "movies": movies,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON} with {len(movies)} movies")


if __name__ == "__main__":
    main()

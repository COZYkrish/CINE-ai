# 🎬 CineAI

**CineAI** is a movie recommendation system built with **Flask**, **HTML/CSS/JavaScript**, and **machine learning techniques** such as **TF-IDF vectorization**, **cosine similarity**, and **K-Means clustering**.

It provides a clean cinematic interface where users can search for a movie title and instantly receive similar movie recommendations based on cast and director relationships from the dataset.

## ✨ Highlights

- 🎥 Modern single-page movie recommendation interface
- 🧠 Recommendation engine powered by TF-IDF + cosine similarity
- 🗂️ K-Means clustering to group similar movies for faster refinement
- 🔎 Search-driven recommendation flow with title matching
- 📡 Flask API backend serving movie data and recommendation results
- 🎨 Visually polished frontend with responsive layout and interactive cards

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Data Handling:** Pandas, `ast`
- **Machine Learning:** scikit-learn
- **Frontend:** HTML, CSS, JavaScript
- **Dataset Source Format:** TMDB credits-style CSV data

## ⚙️ How It Works

The recommendation pipeline in CineAI is intentionally simple and understandable:

1. Movie metadata is loaded from a CSV file.
2. Cast names and director names are extracted from structured JSON-like fields.
3. A combined text feature called a "soup" is built from cast and director information.
4. TF-IDF converts that text into numerical vectors.
5. Cosine similarity measures how close one movie is to another.
6. K-Means clustering groups movies into related clusters.
7. When a user searches for a movie, CineAI finds the closest matches and returns the top recommendations.

This design makes the project easy to study, extend, and explain, which is useful for portfolio work, ML demos, and beginner-friendly recommender-system projects.

## 🚀 Features

### Frontend

- Elegant glassmorphism-inspired UI
- Search box with suggestion support
- Featured movie summary after search
- Top recommendation cards with similarity scores
- Online/offline API status indicator
- Sample movie picks for quick testing

### Backend

- `GET /` serves the main interface
- `GET /api/movies` returns the available movie list
- `GET /api/recommend?title=<movie>` returns recommendations for a title
- Case-insensitive and partial title matching support
- CORS enabled for flexible frontend access

## 📦 Installation

### 1. Clone the project

```powershell
git clone <your-repo-url>
cd "cine ai"
```

### 2. Install dependencies

```powershell
pip install flask pandas scikit-learn
```

### 3. Configure the dataset path

The current backend uses this absolute path:

```python
CSV_PATH = r"D:\Desktop\tmdb_5000_credits.csv"
```

Make sure the dataset exists there, or update `CSV_PATH` in [app.py](/d:/Desktop/cine%20ai/app.py) to match your local machine.

## ▶️ Run the Project

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 🔌 API Overview

### `GET /api/movies`

Returns a list of movies with:

- `id`
- `title`
- `cast`
- `director`

### `GET /api/recommend?title=Inception`

Returns:

- searched movie title
- cast of the queried movie
- director of the queried movie
- recommendation method
- cluster id
- top recommended movies with similarity scores

## 🧪 Example Use Case

Search for a movie like **Inception** and CineAI will:

- find the best matching title
- analyze its cast/director feature vector
- compare it with related movies
- return the top ranked recommendations

This makes the app useful for:

- ML mini-projects
- recommendation system demos
- academic presentations
- portfolio showcases

## 📘 Learning Value

CineAI is a strong beginner-to-intermediate project because it demonstrates:

- how recommendation systems can be built without deep learning
- how text vectorization can power similarity search
- how clustering can improve retrieval strategy
- how to connect an ML backend with a polished frontend
- how to structure a small full-stack AI-style application

## 🔮 Possible Improvements

- Add posters, genres, overview text, and release year
- Move the dataset into the project folder for easier setup
- Add fuzzy matching for better title search
- Support multiple recommendation modes from the UI
- Add genre-based filtering
- Deploy with Render, Railway, or Docker
- Cache model artifacts for faster startup

## 📝 Notes

- The dataset file is not currently included in this repository.
- The backend loads and processes the dataset on startup.
- Recommendation quality is based mainly on **cast** and **director** features in the current implementation.

## 📜 License

Add your preferred license here, such as **MIT**.

## 📁 Project Structure

```text
cine ai/
├── app.py
├── index.html
└── __pycache__/
    └── app.cpython-314.pyc
```

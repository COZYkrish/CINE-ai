# CineAI

CineAI is now packaged as a static movie recommendation site for Vercel.

The recommendation catalog is built ahead of time from the TMDB credits-style CSV, then served as plain JSON to the frontend. Vercel only needs to host `index.html` and the generated data files. It does not run Flask or Python in production.

## Live Demo

<https://cine-ai-xi.vercel.app/>

## Stack

- Frontend: HTML, CSS, JavaScript
- Build-time data pipeline: Python, pandas, scikit-learn
- Hosting: Vercel static deployment

## How It Works

1. `scripts/build_static_catalog.py` reads `data/tmdb_5000_credits.csv`.
2. Cast and director names are extracted from the dataset.
3. TF-IDF vectors are generated from cast and director metadata.
4. K-Means clustering and cosine similarity are used to rank related titles.
5. The script writes `data/catalog.json`.
6. `index.html` loads that JSON directly in the browser.

## Project Structure

```text
cine ai/
|-- data/
|   |-- catalog.json
|   `-- tmdb_5000_credits.csv
|-- scripts/
|   `-- build_static_catalog.py
|-- index.html
|-- vercel.json
`-- .vercelignore
```

## Rebuild The Catalog

Install the Python dependencies locally:

```powershell
pip install pandas scikit-learn
```

Then regenerate the static catalog:

```powershell
python scripts/build_static_catalog.py
```

The generated output is [`data/catalog.json`](/d:/Desktop/cine%20ai/data/catalog.json).

## Deploy To Vercel

The repo is configured for static deployment:

- [`index.html`](/d:/Desktop/cine%20ai/index.html) reads `./data/catalog.json`
- [`vercel.json`](/d:/Desktop/cine%20ai/vercel.json) only sets static headers
- [`.vercelignore`](/d:/Desktop/cine%20ai/.vercelignore) excludes the Flask app, Python runtime files, and the raw CSV from the deployment bundle

After rebuilding `data/catalog.json`, push to `main` and redeploy on Vercel.

## Notes

- The old Flask files are still in the repo, but Vercel no longer uses them.
- If you update the CSV, regenerate `data/catalog.json` before deploying.

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    # SQLite DB in instance folder
    SQLITE_PATH = BASE_DIR / "instance" / "probing.db"

    # Data dirs
    DATA_RAW_DIR = BASE_DIR / "data" / "raw"
    DATA_CACHE_DIR = BASE_DIR / "data" / "cache"

    # Probing defaults
    DEFAULT_MAX_SAMPLES = 1000
    MAX_SAMPLES_HARD_LIMIT = 5000

    # Models
    TRANSFORMER_MODELS = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "distilroberta-base",
        "roberta-base",
    ]

    # Kaggle datasets (slug, text_column, label_column)
    KAGGLE_DATASETS = {
        "sarcasm": {
            "slug": "rmisra/news-headlines-dataset-for-sarcasm-detection",
            "file": "Sarcasm_Headlines_Dataset.json",
            "text_col": "headline",
            "label_col": "is_sarcastic",
            "task_name": "Sarcasm Detection",
        },
        "fake_news": {
            "slug": "clmentbisaillon/fake-and-real-news-dataset",
            "file": "Fake.csv",  # we'll sample equally from Fake + True
            "text_col": "text",
            "label_col": "label",  # we'll map "FAKE"/"REAL" -> 0/1
            "task_name": "Fake vs Real News",
        },
           "amazon_reviews": {
        "slug": "bittlingmayer/amazonreviews",
        "file": "train.ft.txt.bz2",   # <- important
        "text_col": "text",
        "label_col": "label",
        "task_name": "Amazon Sentiment",
    },

        "hate_speech": {
            "slug": "mrmorj/hate-speech-and-offensive-language-dataset",
            "file": "labeled_data.csv",
            "text_col": "tweet",
            "label_col": "class",  # 0=hate,1=offensive,2=neither
            "task_name": "Hate/Offense Detection",
        },
    }

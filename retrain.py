import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from typing import List
import pickle
import json
import requests
import os
import argparse
from common import SentimentClassifier, load_sentiment140, clean_text

def retrain_with_pseudo_labels(new_texts: list[str], run_id: str, base_data_path=""):
    # 1. Load your original labeled dataset
    df = load_sentiment140(base_data_path)
    df["clean_text"] = df["text"].apply(clean_text)

    # 2. Load current model & vectorizer from MLflow
    vectorizer_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/artifacts/artifacts/model.pkl")
    with open(vectorizer_path, "rb") as f:
        print('loading model ...')
        model = pickle.load(f)
    vectorizer_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/artifacts/artifacts/vectorizer.pkl")
    with open(vectorizer_path, "rb") as f:
        print('loading vectorizer ...')
        vectorizer = pickle.load(f)

    # 3. Clean new texts
    pseudo_df = pd.DataFrame(new_texts, columns=["text"])
    pseudo_df["clean_text"] = pseudo_df["text"].apply(clean_text)

    # 4. Pseudo-label new data
    transformed = vectorizer.transform(pseudo_df["clean_text"])
    pseudo_df["label"] = model.predict(transformed)

    # 5. Combine original + pseudo-labeled
    combined_df = pd.concat([df[["clean_text", "label"]], pseudo_df[["clean_text", "label"]]])

    # 6. Retrain a new model
    X_train, X_test, y_train, y_test = train_test_split(
        combined_df["clean_text"], combined_df["label"], test_size=0.2
    )

    new_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_tfidf = new_vectorizer.fit_transform(X_train)
    X_test_tfidf = new_vectorizer.transform(X_test)

    new_model = LogisticRegression(max_iter=1000)
    new_model.fit(X_train_tfidf, y_train)

    y_pred = new_model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("Retrained Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save new artifacts
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(new_vectorizer, f)
    with open("model.pkl", "wb") as f:
        pickle.dump(new_model, f)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression (Retrained w/ Pseudo)")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_metric("accuracy", acc)

        artifacts = {"vectorizer": "vectorizer.pkl", "model": "model.pkl"}

        mlflow.pyfunc.log_model(
            artifact_path="artifacts",
            python_model=SentimentClassifier(),
            artifacts=artifacts,
            registered_model_name="Sentiment140",
            code_path=["common.py"],
        )

def bluesky_login(handle, password):
    url = "https://bsky.social/xrpc/com.atproto.server.createSession"
    resp = requests.post(url, json={"identifier": handle, "password": password})
    if resp.status_code == 200:
        return resp.json()["accessJwt"]
    else:
        raise Exception(f"Login failed: {resp.status_code}\n{resp.text}")
    
def get_latest_feed_posts(jwt, limit=10):
    url = f"https://bsky.social/xrpc/app.bsky.feed.getFeed"
    headers = {"Authorization": f"Bearer {jwt}"}
    params = {
        'feed': 'at://did:plc:wlnfxpukwvrma5lsszlz4hvm/app.bsky.feed.generator/aaaotgiddxmyq',
        'limit': limit,
    }
    resp = requests.get(url, params=params, headers=headers)

    if resp.status_code == 200:
        feed = resp.json()["feed"]
        return [item["post"]["record"]["text"] for item in feed if "post" in item]
    else:
        raise Exception(f"Failed to fetch feed: {resp.status_code}\n{resp.text}")
    
def main(handle, password, run_id, base_data_path=""):
    # 1. Login to Bluesky
    jwt = bluesky_login(handle, password)

    # 2. Fetch latest posts
    new_texts = get_latest_feed_posts(jwt, limit=100)

    # 3. Retrain model with pseudo-labeling
    retrain_with_pseudo_labels(new_texts, run_id, base_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain sentiment classifier with new data from Bluesky.")
    parser.add_argument("--handle", type=str, required=True, help="Bluesky handle (e.g., @your_handle)")
    parser.add_argument("--password", type=str, required=True, help="Bluesky password")
    parser.add_argument("--run_id", type=str, required=True, help="MLflow run ID of the current model")
    parser.add_argument("--base_data_path", type=str, default="", help="Path to base dataset for initial training")

    args = parser.parse_args()
    main(args.handle, args.password, args.run_id, args.base_data_path)
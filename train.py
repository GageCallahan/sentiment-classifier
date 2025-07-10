from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import pickle
from common import SentimentClassifier, load_sentiment140, clean_text
import argparse

def train_model(df):
    df["clean_text"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the vectorizer and model
    vectorizer_path = "vectorizer.pkl"
    model_path = "model.pkl"

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("tfidf_max_features", 5000)
        mlflow.log_metric("accuracy", acc)

        artifacts = {
            "vectorizer": vectorizer_path,
            "model": model_path
        }

        mlflow.pyfunc.log_model(
            # input_example=pd.DataFrame(X_train.iloc[[0]]),
            artifact_path="artifacts",
            python_model=SentimentClassifier(),
            artifacts=artifacts,
            registered_model_name="Sentiment140",
            code_path=["common.py"]
        )

    print("Model logged to MLflow.")

def main(data_path):
    df = load_sentiment140(data_path)
    train_model(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("data_path", type=str, help="Path to the training data CSV file")
    args = parser.parse_args()
    main(args.data_path)
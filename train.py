from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import pickle
from common import SentimentClassifier, load_sentiment140, clean_text
import argparse
import re

class SentimentClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["vectorizer"], "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: List[str]) -> List[int]:
        if isinstance(model_input, pd.Series):
            texts = model_input.tolist()
        elif isinstance(model_input, list):
            texts = model_input
        else:
            raise ValueError("Input should be a list or pandas Series of strings.")
        
        transformed_texts = self.vectorizer.transform(texts)
        return self.model.predict(transformed_texts)

def load_sentiment140(path):
    df = pd.read_csv(path, encoding='latin-1', header=None)
    df.columns = ["polarity", "id", "date", "query", "user", "text"]
    df = df[["polarity", "text"]]
    df["label"] = df["polarity"].replace({0: 0, 4: 1})  # 0 = negative, 4 = positive
    return df[["text", "label"]]

def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove @user
    text = re.sub(r"#\w+", "", text)     # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation
    return text.lower().strip()

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
import pandas as pd
import re
import mlflow
from typing import List
import pickle

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

# ---------- Step 2: Clean Text ----------
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove @user
    text = re.sub(r"#\w+", "", text)     # remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation
    return text.lower().strip()

import os
import pandas as pd
import numpy as np
import gensim
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openpyxl import Workbook

def load_csv(version):
    return pd.read_csv(f"data/{version}.csv")

def get_column_name(version):
    return "lemmatized_text" if version == "lemmatized" else "stemmed_text"

def train_tfidf(version):
    df = load_csv(version)
    col = get_column_name(version)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[col])
    joblib.dump((vectorizer, tfidf_matrix), f"models/tfidf_{version}.pkl")

def tfidf_similarity(input_text, version):
    col = get_column_name(version)
    df = load_csv(version)
    vectorizer, tfidf_matrix = joblib.load(f"models/tfidf_{version}.pkl")
    input_vec = vectorizer.transform([input_text])
    scores = cosine_similarity(input_vec, tfidf_matrix)[0]
    top5 = np.argsort(scores)[-5:][::-1]
    return df.iloc[top5][col].tolist(), scores[top5].tolist()

def avg_vector(model, text):
    words = text.split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else None

def word2vec_similarity(input_text, model_path, version):
    col = get_column_name(version)
    df = load_csv(version)
    model = gensim.models.Word2Vec.load(model_path)
    input_vec = avg_vector(model, input_text)
    df["vec"] = df[col].apply(lambda x: avg_vector(model, x))
    df = df[df["vec"].notnull()]
    mat = np.vstack(df["vec"].values)
    scores = cosine_similarity([input_vec], mat)[0]
    top5 = np.argsort(scores)[-5:][::-1]
    return df.iloc[top5][col].tolist(), scores[top5].tolist()

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    input_text = load_csv("lemmatized")["lemmatized_text"].iloc[0]

    results = {}

    for v in ["lemmatized", "stemmed"]:
        train_tfidf(v)
        texts, scores = tfidf_similarity(input_text, v)
        results[f"tfidf_{v}"] = (texts, scores)

    for m in os.listdir("models"):
        if m.endswith(".model"):
            v = "lemmatized" if "lemmatized" in m else "stemmed"
            path = os.path.join("models", m)
            texts, scores = word2vec_similarity(input_text, path, v)
            results[m] = (texts, scores)

    wb = Workbook()
    ws = wb.active
    ws.title = "Sonuclar"
    ws.append(["Model", "Benzerlik Skorları"])

    for model, (texts, scores) in results.items():
        ws.append([model, ", ".join([f"{s:.4f}" for s in scores])])

    wb.save("outputs/benzerlik_raporu.xlsx")
    print("✅ Rapor oluşturuldu: outputs/benzerlik_raporu.xlsx")

if __name__ == "__main__":
    main()

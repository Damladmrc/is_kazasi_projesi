import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))

from veri_kaynagi import zipten_cikar
import pandas as pd
from on_isleme import on_isle
from vektor_tfidf import tfidf_uygula
from vektor_word2vec import word2vec_model_egit
from zipf_analizi import zipf_grafigi_olustur

# Gerekli klasörleri oluştur
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

# 1. Yerel veri dosyasını aç
zipten_cikar("data/osha_accidents.zip", hedef_klasor="data")

# 2. Veri okuma
df = pd.read_csv("data/osha_accidents.csv")

# 3. Ön işleme
df = on_isle(df)
df[["stemmed_text"]].to_csv("outputs/temiz_veri_stemmed.csv", index=False)
df[["lemmatized_text"]].to_csv("outputs/temiz_veri_lemmatized.csv", index=False)

# 4. Zipf grafikleri
zipf_grafigi_olustur(df["stemmed_text"].tolist(), "Stemmed Metin")
zipf_grafigi_olustur(df["lemmatized_text"].tolist(), "Lemmatized Metin")

# 5. TF-IDF
tfidf_uygula(df["stemmed_text"], "outputs/tfidf_stemmed.csv")
tfidf_uygula(df["lemmatized_text"], "outputs/tfidf_lemmatized.csv")

# 6. Word2Vec
stem_tokens = [text.split() for text in df["stemmed_text"].dropna()]
lem_tokens = [text.split() for text in df["lemmatized_text"].dropna()]

parametreler = [
    {"model_type": "cbow", "window": 2, "vector_size": 100},
    {"model_type": "skipgram", "window": 2, "vector_size": 100},
    {"model_type": "cbow", "window": 4, "vector_size": 100},
    {"model_type": "skipgram", "window": 4, "vector_size": 100},
    {"model_type": "cbow", "window": 2, "vector_size": 300},
    {"model_type": "skipgram", "window": 2, "vector_size": 300},
    {"model_type": "cbow", "window": 4, "vector_size": 300},
    {"model_type": "skipgram", "window": 4, "vector_size": 300}
]

for p in parametreler:
    word2vec_model_egit(lem_tokens, p["model_type"], p["window"], p["vector_size"], "lemmatized")
    word2vec_model_egit(stem_tokens, p["model_type"], p["window"], p["vector_size"], "stemmed")

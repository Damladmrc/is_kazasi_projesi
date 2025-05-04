import pandas as pd
from gensim.models import Word2Vec
import os

def word2vec_model_egit(kelimeler_listesi, model_type, window, vector_size, veri_tipi):
    sg = 1 if model_type == "skipgram" else 0
    model = Word2Vec(
        sentences=kelimeler_listesi,
        vector_size=vector_size,
        window=window,
        sg=sg,
        min_count=1,  # DÜZENLENDİ: 2 yerine 1 yapıldı
        workers=4,
        epochs=10
    )
    model_adi = f"word2vec_{veri_tipi}_{model_type}_win{window}_dim{vector_size}.model"
    model.save(os.path.join("models", model_adi))
    print(f"{model_adi} modeli kaydedildi.")
    return model

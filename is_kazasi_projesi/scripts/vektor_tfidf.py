import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_uygula(textler, dosya_adi):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textler)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.to_csv(dosya_adi, index=False)
    print(f"TF-IDF çıktısı kaydedildi: {dosya_adi}")

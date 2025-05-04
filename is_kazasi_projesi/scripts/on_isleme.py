import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def temizle(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def token_ayikla(text):
    return word_tokenize(text)

def stopword_sil(tokens):
    return [w for w in tokens if w not in stop_words]

def stem(tokens):
    return [stemmer.stem(w) for w in tokens]

def lemmatize(tokens):
    return [lemmatizer.lemmatize(w) for w in tokens]

def on_isle(df, text_column="description"):
    stemmed_list, lemmatized_list = [], []
    for text in df[text_column]:
        temiz = temizle(str(text))
        tokens = token_ayikla(temiz)
        filtered = stopword_sil(tokens)
        stemmed_list.append(" ".join(stem(filtered)))
        lemmatized_list.append(" ".join(lemmatize(filtered)))
    df["stemmed_text"] = stemmed_list
    df["lemmatized_text"] = lemmatized_list
    return df

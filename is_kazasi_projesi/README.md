# İş Sağlığı Projesi: İş Kazası Açıklaması - Sebep Eşleşmesi

Bu proje, metin tabanlı iş kazası açıklamalarını analiz ederek olası nedenleri anlamaya yönelik bir yapay zekâ çalışmasıdır. TF-IDF ve Word2Vec vektörleştirme yöntemleri kullanılmıştır.

## 🔍 Proje Özeti

- **Veri:** 20 MB'lık 300.000 satırlık örnek iş kazası açıklamaları (`osha_accidents.csv`)
- **Amaç:** Açıklamalardan anlamlı özellikler çıkarıp, neden gruplarını analiz etmek
- **Yöntemler:** 
  - Metin temizleme (noktalama, durak kelime, kök/alma, lemmatizasyon)
  - Zipf grafiği analizi
  - TF-IDF vektörleştirme
  - Word2Vec modeli (CBOW ve Skip-Gram)

---

## 📁 Proje Yapısı

```
is_kazasi_projesi/
├── data/
│   └── osha_accidents.zip
├── outputs/
│   ├── temiz_veri_stemmed.csv
│   ├── temiz_veri_lemmatized.csv
│   ├── tfidf_stemmed.csv
│   └── tfidf_lemmatized.csv
├── models/
│   └── word2vec_*.model
├── visuals/
│   └── zipf_*.png
├── scripts/
│   ├── on_isleme.py
│   ├── vektor_tfidf.py
│   ├── vektor_word2vec.py
│   ├── zipf_analizi.py
│   └── veri_kaynagi.py
├── main.py
└── requirements.txt
```

---

## ⚙️ Kullanım

### 1. Ortam Kurulumu
```bash
pip install -r requirements.txt
```

### 2. Projeyi Çalıştırma
```bash
python main.py
```

Tüm adımlar (veri açma, işleme, modelleme) otomatik çalışır.

---

## 📊 Üretilen Sonuçlar

- `outputs/` klasöründe: Temizlenmiş metinler ve TF-IDF vektörleri
- `models/` klasöründe: Word2Vec modelleri
- `visuals/` klasöründe: Zipf frekans dağılım grafikleri

---

## 🧠 Kullanılan Teknolojiler

- Python 3.10+
- NLTK
- Gensim
- Pandas
- Matplotlib
- Scikit-learn

---

## 👨‍💻 Hazırlayan

Bu proje, [Metin Tabanlı Veri Setleri ile Yapay Zekâ Modelleri Geliştirme] ödevi kapsamında geliştirilmiştir.

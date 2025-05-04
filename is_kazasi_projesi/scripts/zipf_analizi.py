import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def zipf_grafigi_olustur(text_list, baslik):
    kelimeler = " ".join(text_list).split()
    frekanslar = Counter(kelimeler)
    frekans_degerleri = sorted(frekanslar.values(), reverse=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(np.log(range(1, len(frekans_degerleri)+1)), np.log(frekans_degerleri))
    plt.title(f"Zipf Grafiği - {baslik}")
    plt.xlabel("log(Sıra)")
    plt.ylabel("log(Frekans)")
    plt.grid(True)
    plt.show()

import os
import requests
import zipfile

def veriyi_indir_kaydet(url, hedef_klasor="veri"):
    os.makedirs(hedef_klasor, exist_ok=True)
    dosya_adi = os.path.join(hedef_klasor, url.split("/")[-1])
    if not os.path.exists(dosya_adi):
        print(f"{dosya_adi} indiriliyor...")
        response = requests.get(url)
        if response.status_code != 200 or b'html' in response.content[:100].lower():
            raise Exception("Dosya indirilemedi veya geçerli bir ZIP dosyası değil.")
        with open(dosya_adi, "wb") as f:
            f.write(response.content)
        print("İndirme tamamlandı.")
    else:
        print("Veri daha önce indirilmiş.")
    return dosya_adi

def zipten_cikar(zip_yolu, hedef_klasor="veri"):
    with zipfile.ZipFile(zip_yolu, 'r') as zip_ref:
        zip_ref.extractall(hedef_klasor)
    print("Zip dosyası çıkarıldı.")

if __name__ == "__main__":
    kaggle_verisi_url = "https://github.com/bekirbostanci/public-datasets/raw/main/osha_accidents.zip"
    zip_yolu = veriyi_indir_kaydet(kaggle_verisi_url)
    zipten_cikar(zip_yolu)

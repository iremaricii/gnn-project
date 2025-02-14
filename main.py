import os  # Komut satırı işlemleri için

# ✅ Modeli eğitme işlemi başlatılıyor.
print("✅ GCN Modeli Eğitiliyor...")
os.system("python scripts/train.py")  # `train.py` dosyasını çalıştır.

# ✅ Model eğitildikten sonra test aşaması başlatılıyor.
print("\n✅ GCN Modeli Test Ediliyor...")
os.system("python scripts/test.py")  # `test.py` dosyasını çalıştır.



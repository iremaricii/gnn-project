# 📌 Gerekli kütüphaneleri içe aktarıyoruz.
import os  # Dosya işlemleri (modeli kaydetmek için)
import torch  # PyTorch kütüphanesi, derin öğrenme için
import torch.nn.functional as F  # Aktivasyon fonksiyonları ve loss hesaplamaları için
from torch.optim import Adam  # Optimizasyon algoritması
from torch.nn import CrossEntropyLoss  # Kayıp fonksiyonu (loss function)
from torch_geometric.datasets import Planetoid  # Cora veri setini yüklemek için
from models.gcn import GCN  # Daha önce tanımladığımız GCN modelini içe aktarıyoruz.

# 📌 Veri setini yükle (Cora veri seti: akademik makaleler arası bağlantıları içerir)
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]  # Veri setindeki tek grafı al (Cora zaten tek bir graf içerir)

# 📌 GCN Modelini Başlat
model = GCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)

# 📌 Optimizasyon ve Kayıp Fonksiyonunu Tanımla
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Adam optimizer (öğrenme oranı: 0.01)
criterion = CrossEntropyLoss()  # CrossEntropyLoss: Sınıflandırma için yaygın olarak kullanılır


# 📌 Modelin Eğitim Fonksiyonu
def train():
    """
    GCN modelini tek bir epoch boyunca eğitir.
    """
    model.train()  # Modeli eğitim moduna al.
    optimizer.zero_grad()  # Gradyanları sıfırla.

    out = model(data.x, data.edge_index)  # Modeli çalıştır, tahminleri al.

    # 🔹 Sadece eğitim verisi için loss hesapla (train_mask True olan düğümler)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()  # Geriye yayılım (backpropagation) işlemi yap.
    optimizer.step()  # Model ağırlıklarını güncelle.

    return loss.item()  # Loss değerini döndür.


# 📌 Eğitim Döngüsü
losses = []  # Loss değerlerini saklamak için bir liste oluştur
for epoch in range(200):  # 200 epoch boyunca eğit
    loss = train()  # Eğitim fonksiyonunu çağır
    losses.append(loss)  # Kayıp değerini kaydet

    # 🔹 Her 20 epoch'ta bir ekrana yazdır
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ✅ Modelin Kaydedileceği Klasörü Kontrol Et, Yoksa Oluştur
os.makedirs("models", exist_ok=True)  # "models" klasörü yoksa oluştur

# ✅ Eğitilmiş Modeli Kaydet
model_path = "models/gcn_model.pth"  # Modelin kaydedileceği dosya yolu
torch.save(model.state_dict(), model_path)  # Modelin ağırlıklarını kaydet

# ✅ Modelin Başarıyla Kaydedildiğini Kontrol Et ve Kullanıcıya Bilgi Ver
if os.path.exists(model_path):
    print(f"\n✅ Model başarıyla kaydedildi: {model_path}")
else:
    print("\n❌ Model kaydedilemedi! Dosya oluşturulamadı.")

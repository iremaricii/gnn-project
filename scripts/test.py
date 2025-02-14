# 📌 Gerekli kütüphaneleri içe aktarıyoruz.
import torch  # PyTorch, derin öğrenme işlemleri için
from torch_geometric.datasets import Planetoid  # Cora veri setini yüklemek için
from sklearn.metrics import accuracy_score, f1_score  # Modelin performansını ölçmek için
from models.gcn import GCN  # Daha önce tanımladığımız GCN modelini içe aktarıyoruz.

# 📌 Veri setini yükle (Cora veri seti: akademik makaleler arası bağlantıları içerir)
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]  # Veri setindeki tek grafı al (Cora tek bir graf içerir)

# 📌 Modeli Yükle
model = GCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)

# 🔹 Modelin eğitim sırasında kaydedilen ağırlıklarını yükle
model.load_state_dict(torch.load("models/gcn_model.pth", map_location=torch.device("cpu")))

# 🔹 Modeli test moduna geçir (Eğitim sırasında kullanılan dropout gibi işlemler devre dışı kalır)
model.eval()

# 📌 Modeli Test Et
with torch.no_grad():  # Gradyan hesaplamalarını devre dışı bırak (test sırasında gereksiz)
    out = model(data.x, data.edge_index)  # Modeli çalıştır, tahminleri al
    predictions = out.argmax(dim=1)  # En yüksek olasılığa sahip sınıfı seç

# 📌 Gerçek Etiketleri Al
y_true = data.y.cpu().numpy()  # Gerçek etiketler
y_pred = predictions.cpu().numpy()  # Modelin tahminleri

# 📌 Doğruluk (Accuracy) ve F1 Skorlarını Hesapla
accuracy = accuracy_score(y_true, y_pred)  # Genel doğruluk oranı
f1 = f1_score(y_true, y_pred, average="macro")  # Makro ortalama F1 skoru hesapla

# 📌 Sonuçları Konsola Yazdır
print(f"\n✅ Model Test Sonuçları:")
print(f"Test Doğruluğu: {accuracy:.4f}")  # Doğruluk oranını ekrana yazdır
print(f"F1 Skoru (Makro Ortalama): {f1:.4f}")  # Makro F1 skorunu ekrana yazdır

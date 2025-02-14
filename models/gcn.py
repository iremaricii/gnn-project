# 📌 Gerekli kütüphaneleri içe aktarıyoruz.
import torch  # PyTorch, derin öğrenme işlemleri için kullanılır.
import torch.nn.functional as F  # Aktivasyon fonksiyonları ve kayıp hesaplamaları için.
from torch_geometric.nn import GCNConv  # PyTorch Geometric içindeki Graph Convolutional Network katmanları.

# 📌 Graph Convolutional Network (GCN) modelini tanımlıyoruz.
class GCN(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) modeli.
    Bu model, düğümler arası ilişkileri kullanarak düğüm özelliklerini işler ve sınıflandırma yapar.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        GCN modelinin katmanlarını ve yapısını belirler.

        Parametreler:
        - in_channels (int): Giriş katmanındaki özellik sayısı (örneğin, düğüm başına kelime vektörü boyutu).
        - hidden_channels (int): Ara katmandaki nöron sayısı (gizli katman boyutu).
        - out_channels (int): Çıkış katmanındaki sınıf sayısı (Cora veri setinde 7 sınıf var).
        """
        super(GCN, self).__init__()  # Torch modelinin temel sınıfını başlat.

        # 📌 İlk Graph Convolutional Layer (GCN Katmanı)
        self.conv1 = GCNConv(in_channels, hidden_channels)

        # 📌 İkinci Graph Convolutional Layer (GCN Katmanı)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Modelin ileri besleme (forward pass) işlemini tanımlar.

        Parametreler:
        - x (Tensor): Düğümler için giriş özellikleri (node feature matrix).
        - edge_index (Tensor): Grafın düğümler arasındaki bağlantıları gösteren kenar bilgileri (edge list).

        Çıktı:
        - F.log_softmax(x, dim=1): Modelin tahmin ettiği olasılık dağılımı (her düğüm için bir sınıf skoru).
        """
        # 📌 İlk GCN katmanı: Özellikleri işleyip aktivasyon fonksiyonunu uygula.
        x = self.conv1(x, edge_index)  # İlk GCN katmanından geçir.
        x = F.relu(x)  # ReLU aktivasyon fonksiyonunu uygula.

        # 📌 İkinci GCN katmanı: Sonuçları hesapla.
        x = self.conv2(x, edge_index)

        # 📌 Softmax yerine Log-Softmax kullanarak olasılıkları normalize et.
        # CrossEntropyLoss doğrudan log-softmax uygulanmış girişleri beklediği için bunu kullanıyoruz.
        return F.log_softmax(x, dim=1)

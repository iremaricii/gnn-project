import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch_geometric.datasets import Planetoid
from models.gcn import GCN


def plot_loss(losses):
    """Eğitim sırasında kaydedilen loss (kayıp) değerlerini görselleştirir."""
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Eğitim Kaybı (Loss)", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GCN Modelinin Eğitim Kaybı Grafiği")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_f1_score(f1_scores):
    """Eğitim sürecinde epoch bazlı F1 skorunun değişimini gösterir."""
    plt.figure(figsize=(8, 5))
    plt.plot(f1_scores, label="F1 Skoru", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Skoru")
    plt.title("GCN Modelinin F1 Skoru Grafiği")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_graph(data):
    """Cora veri setindeki düğümleri ve kenarları görselleştirir."""
    G = nx.Graph()
    edge_index = data.edge_index.numpy()

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=30, alpha=0.6, edge_color="gray")
    plt.title("Cora Veri Seti - Düğümler ve Kenarlar")
    plt.show()


def visualize_tsne(model, data):
    """Modelin düğüm embedding'lerini t-SNE kullanarak 2D görselleştirir."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        embeddings = out.cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    labels = data.y.cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="jet", alpha=0.7)
    plt.colorbar(scatter, label="Düğüm Etiketleri")
    plt.title("GCN Modelinin t-SNE Embedding Görselleştirmesi")
    plt.xlabel("t-SNE Boyut 1")
    plt.ylabel("t-SNE Boyut 2")
    plt.show()


def plot_confusion_matrix(model, data):
    """Modelin tahmin sonuçlarına göre Confusion Matrix çizer."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

    y_true = data.y.cpu().numpy()
    y_pred = pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.title("Confusion Matrix - GCN Modeli")
    plt.show()


def visualize_graph_with_labels(data):
    """Düğümleri etiketlerine göre renklendirerek Cora veri setinin grafiğini çizer."""
    print("\n🔍 Cora Veri Seti - Etiketlere Göre Renklendirme Yapılıyor...")
    G = nx.Graph()
    edge_index = data.edge_index.numpy()

    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    labels = data.y.cpu().numpy()
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    node_colors = [colors[label % len(colors)] for label in labels]

    plt.figure(figsize=(8, 6))
    nx.draw(G, node_size=50, node_color=node_colors, alpha=0.8, edge_color="gray")
    plt.title("Cora Veri Seti - Etiketlere Göre Renklendirilmiş Düğümler")
    plt.show()


def visualize_pca(model, data):
    """Modelin düğüm embedding'lerini PCA kullanarak 2D görselleştirir."""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        embeddings = out.cpu().numpy()

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    labels = data.y.cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="jet", alpha=0.7)
    plt.colorbar(scatter, label="Düğüm Etiketleri")
    plt.title("GCN Modelinin PCA Embedding Görselleştirmesi")
    plt.xlabel("PCA Boyut 1")
    plt.ylabel("PCA Boyut 2")
    plt.show()


# Eğer bu dosya doğrudan çalıştırılırsa:
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "gcn_model.pth"))

    if os.path.exists(model_path):
        print(f"✅ Model dosyası bulundu: {model_path}")
    else:
        raise FileNotFoundError(f"❌ Model dosyası bulunamadı: {model_path}")

    dataset = Planetoid(root=os.path.join(current_dir, "..", "data", "Cora"), name="Cora")
    data = dataset[0]
    model = GCN(in_channels=dataset.num_features, hidden_channels=16, out_channels=dataset.num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("✅ Model başarıyla yüklendi ve değerlendirme moduna geçti.")

    print("\n📊 Eğitim Kaybı Grafiği Çiziliyor...")
    losses = [1.9438, 0.0948, 0.0131, 0.0143, 0.0169, 0.0156, 0.0139, 0.0127, 0.0117, 0.0109]
    plot_loss(losses)

    print("\n📊 F1 Skoru Grafiği Çiziliyor...")
    f1_scores = [0.65, 0.72, 0.76, 0.78, 0.80, 0.8080]  # F1 skorlarının epoch başına değişimi
    plot_f1_score(f1_scores)

    print("\n🔍 Cora Veri Seti Grafiği Çiziliyor...")
    visualize_graph(data)

    print("\n🎨 t-SNE Embedding Grafiği Çiziliyor...")
    visualize_tsne(model, data)

    print("\n📊 Confusion Matrix Çiziliyor...")
    plot_confusion_matrix(model, data)

    print("\n🔍 Etiketlere Göre Renklendirilmiş Graf Çiziliyor...")
    visualize_graph_with_labels(data)

    print("\n🎨 PCA Embedding Grafiği Çiziliyor...")
    visualize_pca(model, data)

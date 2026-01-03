import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

# =========================
# 1. VERİ OKUMA
# =========================

df = pd.read_csv(
    "hashtag_donaldtrump.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

df = df.dropna(subset=["tweet"])
df["tweet"] = df["tweet"].astype(str).str.lower()

# =========================
# 2. HASHTAG–HASHTAG NETWORK
# =========================

H = nx.Graph()

for tweet in df["tweet"]:
    hashtags = list(set(re.findall(r"#(\w+)", tweet)))
    for i in range(len(hashtags)):
        for j in range(i + 1, len(hashtags)):
            if H.has_edge(hashtags[i], hashtags[j]):
                H[hashtags[i]][hashtags[j]]["weight"] += 1
            else:
                H.add_edge(hashtags[i], hashtags[j], weight=1)

# =========================
# 3. FİLTRELEME
# =========================

MIN_WEIGHT = 30
H_filtered = nx.Graph(
    (u, v, d) for u, v, d in H.edges(data=True) if d["weight"] >= MIN_WEIGHT
)

print("\nFiltered Trump Hashtag Network")
print(f"Nodes: {H_filtered.number_of_nodes()}")
print(f"Edges: {H_filtered.number_of_edges()}")

# =========================
# 4. DEGREE CENTRALIZATION
# =========================

degrees = dict(H_filtered.degree())
max_deg = max(degrees.values())
n = H_filtered.number_of_nodes()

degree_centralization = sum(max_deg - d for d in degrees.values()) / ((n - 1) * (n - 2))
print(f"\nDegree Centralization: {degree_centralization:.4f}")

# =========================
# 5. DEGREE DISTRIBUTION (LOG SCALE)
# =========================

degree_values = list(degrees.values())

plt.figure(figsize=(8, 5))
plt.hist(degree_values, bins=30, log=True, edgecolor="black")
plt.xlabel("Degree")
plt.ylabel("Frequency (Log Scale)")
plt.title("Degree Distribution (Log Scale)")
plt.grid(alpha=0.3)
plt.show()

# =========================
# 6. GINI COEFFICIENT (DEGREE INEQUALITY)
# =========================

def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * np.sum(x)) - (n + 1) / n

gini_degree = gini(degree_values)
print(f"Gini Coefficient (Degree): {gini_degree:.4f}")

# =========================
# 7. MOST CENTRAL HASHTAGS (DEGREE CENTRALITY)
# =========================

degree_centrality = nx.degree_centrality(H_filtered)

top_10_degree = sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nMost Central Hashtags (Degree Centrality)")
for tag, score in top_10_degree:
    print(f"{tag} → {score:.4f}")

# ---- DEGREE SUBGRAPH ----
deg_nodes = [tag for tag, _ in top_10_degree]
deg_sub = H_filtered.subgraph(deg_nodes).copy()

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(deg_sub, seed=42, k=1.5)

node_sizes = [degree_centrality[n] * 4000 for n in deg_sub.nodes()]

nx.draw_networkx_edges(deg_sub, pos, alpha=0.4)
nx.draw_networkx_nodes(
    deg_sub, pos,
    node_size=node_sizes,
    node_color="steelblue",
    alpha=0.9
)
nx.draw_networkx_labels(deg_sub, pos, font_size=11, font_weight="bold")

plt.title("Most Central Hashtags (Degree Centrality)")
plt.axis("off")
plt.show()

# =========================
# 8. CORE HASHTAG NETWORK (TOP 50 DEGREE)
# =========================

top_50_degree = sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:50]

top_50_tags = [tag for tag, _ in top_50_degree]
H_core = H_filtered.subgraph(top_50_tags).copy()

print("\nCore Hashtag Network (Top 50 Degree)")
print(f"Nodes: {H_core.number_of_nodes()}")
print(f"Edges: {H_core.number_of_edges()}")

# =========================
# 9. BRIDGING HASHTAGS (BETWEENNESS CENTRALITY)
# =========================

betweenness = nx.betweenness_centrality(
    H_core,
    weight="weight",
    normalized=True
)

top_10_betweenness = sorted(
    betweenness.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nBridging Hashtags (Betweenness Centrality)")
for tag, score in top_10_betweenness:
    print(f"{tag} → {score:.6f}")

# ---- BETWEENNESS SUBGRAPH ----
bw_nodes = [tag for tag, _ in top_10_betweenness]
bw_sub = H_filtered.subgraph(bw_nodes).copy()

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(bw_sub, seed=24, k=1.8)

node_sizes = [betweenness.get(n, 0) * 80000 for n in bw_sub.nodes()]
node_colors = [betweenness.get(n, 0) for n in bw_sub.nodes()]

nx.draw_networkx_edges(bw_sub, pos, alpha=0.4)
nodes = nx.draw_networkx_nodes(
    bw_sub, pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="plasma",
    alpha=0.9
)
nx.draw_networkx_labels(bw_sub, pos, font_size=11, font_weight="bold")

plt.colorbar(nodes, label="Betweenness Centrality")
plt.title("Bridging Hashtags (Betweenness Centrality)")
plt.axis("off")
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# 1. Veri Hazırlığı (Sıralama ve Etiketler)
# top_15_degree verisinin en yüksekten en düşüğe sıralandığından emin olun
top_15_degree = sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:15]

# Görseldeki gibi en yüksek değerin en üstte görünmesi için listeleri ters çeviriyoruz
tags = [f"{tag}" for tag, _ in top_15_degree][::-1]
scores = [score for _, score in top_15_degree][::-1]

# 2. Grafik Tasarımı ve Figür Boyutu
# Görseldeki genişliği yakalamak için figsize parametresi artırıldı
fig, ax = plt.subplots(figsize=(18, 10))

# Renk Gradyanı: Blues colormap kullanılarak en merkezi tag en koyu mavi yapılır
colors = plt.cm.Blues(np.linspace(0.1, 0.9, len(tags)))

# Yatay Çubuk Grafiği (barh)
bars = ax.barh(tags, scores, color=colors, height=0.75)

# 3. Görseldeki Detayların Eklenmesi
# Çubukların sonuna veri etiketlerini (5 basamaklı) ekleme
ax.bar_label(bars, fmt='%.5f', padding=5, fontweight='bold', fontsize=11)

# Eksen ve Başlık Ayarları
ax.set_title("Top 15 Most Central Hashtags in Trump Dataset", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Degree Centrality Score (Normalized)", fontsize=13)
ax.set_ylabel("Hashtag", fontsize=13)

# Görseldeki dikey kesikli çizgiler (Grid)
ax.xaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True) # Izgarayı çubukların arkasına al

# Üst ve sağ çerçeve çizgilerini kaldırarak sadeleştirme
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Grafik sınırlarını verilere göre biraz genişletme (etiketlerin sığması için)
ax.set_xlim(0, max(scores) * 1.05)

plt.tight_layout()
plt.show()
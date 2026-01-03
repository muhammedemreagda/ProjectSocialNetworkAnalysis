import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity

# =========================
# VERİ OKUMA
# =========================

df = pd.read_csv(
    "hashtag_joebiden.csv",
    engine="python",
    encoding="utf-8",
    on_bad_lines="skip"
)

df = df.dropna(subset=["tweet"])
df["tweet"] = df["tweet"].astype(str).str.lower()

# =========================
# ADIM 1: Bipartite Network
# =========================

B = nx.Graph()

for _, row in df.iterrows():
    user = row["user_screen_name"]
    hashtags = re.findall(r"#(\w+)", row["tweet"])
    for tag in hashtags:
        B.add_edge(user, tag)

print("Bipartite Network")
print(f"Nodes: {B.number_of_nodes()}")
print(f"Edges: {B.number_of_edges()}")

# =========================
# ADIM 2: Hashtag–Hashtag Network
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

print("\nRaw Hashtag Network")
print(f"Nodes: {H.number_of_nodes()}")
print(f"Edges: {H.number_of_edges()}")

# =========================
# ADIM 3: Filtreleme
# =========================

H_filtered = nx.Graph(
    (u, v, d) for u, v, d in H.edges(data=True) if d["weight"] >= 30
)

print("\nFiltered Hashtag Network")
print(f"Nodes: {H_filtered.number_of_nodes()}")
print(f"Edges: {H_filtered.number_of_edges()}")

# =========================
# ADIM 4: Community Detection
# =========================

communities = greedy_modularity_communities(H_filtered)

print(f"\nTotal communities detected: {len(communities)}")

for i, c in enumerate(sorted(communities, key=len, reverse=True)[:5]):
    print(f"Community {i+1} size: {len(c)}")

mod_score = modularity(H_filtered, communities)
print(f"\nModularity score: {mod_score:.4f}")

# =========================
# ADIM 5: Degree Distribution + Gini
# =========================

degrees = [d for _, d in H_filtered.degree()]

plt.hist(degrees, bins=50)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Hashtag Degree Distribution")
plt.show()

def gini(x):
    x = np.array(x)
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1) * x))) / (n * np.sum(x)) - (n + 1) / n

print(f"Gini coefficient (degree): {gini(degrees):.4f}")

# =========================
# ADIM 6: Degree Centrality (Top 10)
# =========================

degree_centrality = nx.degree_centrality(H_filtered)

top_10_degree = sorted(
    degree_centrality.items(),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nEn merkezi 10 hashtag (degree):")
for tag, score in top_10_degree:
    print(f"{tag} → {score:.4f}")

# =========================
# ADIM 7: Degree Subgraph
# =========================

top_degree_tags = [tag for tag, _ in top_10_degree]
degree_subgraph = H_filtered.subgraph(top_degree_tags).copy()

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(degree_subgraph, seed=42, k=1.5)

node_sizes = [
    degree_centrality[n] * 4000 for n in degree_subgraph.nodes()
]

nx.draw_networkx_edges(degree_subgraph, pos, alpha=0.4, width=1.5)
nx.draw_networkx_nodes(
    degree_subgraph, pos,
    node_size=node_sizes,
    node_color="steelblue",
    alpha=0.9
)
nx.draw_networkx_labels(degree_subgraph, pos, font_size=11, font_weight="bold")

plt.title("Top 10 Degree-Central Hashtags")
plt.axis("off")
plt.show()

# =========================
# ADIM 8: Betweenness Centrality (OPTİMİZE)
# =========================

# Önce çekirdek ağı tanımla (Top 50 degree-central hashtag)
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

# Betweenness centrality artık çekirdek ağda hesaplanıyor
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

print("\nSöylemi birbirine bağlayan 10 hashtag (betweenness):")
for tag, score in top_10_betweenness:
    print(f"{tag} → {score:.6f}")


# =========================
# ADIM 9: Betweenness Subgraph (OPTİMİZE)
# =========================

top_bw_tags = [tag for tag, _ in top_10_betweenness]
bw_subgraph = H_filtered.subgraph(top_bw_tags).copy()

plt.figure(figsize=(10, 10))

pos = nx.spring_layout(bw_subgraph, seed=24, k=1.8)

node_sizes = [
    betweenness[n] * 80000
    for n in bw_subgraph.nodes()
]

node_colors = [
    betweenness[n]
    for n in bw_subgraph.nodes()
]

nx.draw_networkx_edges(
    bw_subgraph,
    pos,
    alpha=0.4,
    width=1.5
)

nodes = nx.draw_networkx_nodes(
    bw_subgraph,
    pos,
    node_size=node_sizes,
    node_color=node_colors,
    cmap="plasma",
    alpha=0.9
)

nx.draw_networkx_labels(
    bw_subgraph,
    pos,
    font_size=11,
    font_weight="bold"
)

plt.colorbar(nodes, label="Betweenness Centrality")
plt.title("Top 10 Betweenness-Central Hashtags (Narrative Bridges)")
plt.axis("off")
plt.show()

import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import combinations


# =========================
# VERİ OKUMA VE ÖN İŞLEME
# =========================
def get_processed_data(file_path):
    print(f"{file_path} işleniyor...")
    df = pd.read_csv(file_path, engine="python", encoding="utf-8", on_bad_lines="skip")
    df = df.dropna(subset=["tweet"])
    df["tweet"] = df["tweet"].astype(str).str.lower()

    H = nx.Graph()
    node_counts = Counter()
    total_tag_usage = 0

    for tweet in df["tweet"]:
        hashtags = list(set(re.findall(r"#(\w+)", tweet)))
        if hashtags:
            node_counts.update(hashtags)
            total_tag_usage += len(hashtags)
            for u, v in combinations(sorted(hashtags), 2):
                if H.has_edge(u, v):
                    H[u][v]["weight"] += 1
                else:
                    H.add_edge(u, v, weight=1)
    return H, node_counts, total_tag_usage


# =========================
# ANALİZ FONKSİYONLARI
# =========================

# 1. Degree Centralization (Biden Network)
def plot_centralization(G):
    # Formül: sum(max_deg - deg_i) / (n-2)
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    n = len(G)
    centralization = sum(max_deg - d for d in degrees.values()) / (n - 2) if n > 2 else 0
    print(f"Degree Centralization Score: {centralization:.4f}")
    return centralization


# 2. Degree Distribution (Log Scale)
def plot_degree_distribution(node_counts):
    degrees = list(node_counts.values())
    plt.figure(figsize=(12, 7))
    plt.hist(degrees, bins=50, color='#87CEEB', edgecolor='black', log=True)
    plt.title("Degree Distribution (Log Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Degree (Bağlantı Sayısı)")
    plt.ylabel("Frequency (Hashtag Sayısı - Log)")
    plt.grid(axis='y', linestyle='-', alpha=0.2)
    plt.show()


# 3. Gini Coefficient & Lorenz Curve
def plot_lorenz_gini(node_counts, dataset_name="Biden"):
    degrees = np.sort(list(node_counts.values()))
    n = len(degrees)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * degrees)) / (n * np.sum(degrees))

    lorenz = np.cumsum(degrees) / np.sum(degrees)
    lorenz = np.insert(lorenz, 0, 0)
    pos = np.linspace(0, 1, len(lorenz))

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Mükemmel Eşitlik Çizgisi')
    plt.plot(pos, lorenz, color='navy', lw=2, label=f'{dataset_name} Hashtag Dağılımı')
    plt.fill_between(pos, pos, lorenz, color='red', alpha=0.1, label='Eşitliksiz (Gini Alanı)')

    # Gini Kutucuğu
    plt.text(0.2, 0.8, f"Gini Skoru: {gini:.4f}", fontsize=14,
             bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=1'))

    plt.title(f"Lorenz Curve: {dataset_name} Network Inequality Analysis\nGini Coefficient: {gini:.4f}")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()


# 4. Core Hashtag Network (Top 50)
def plot_core_network(G, node_counts, title="Top 50 Global Core"):
    top_50 = [n for n, c in Counter(dict(G.degree())).most_common(50)]
    sub = G.subgraph(top_50)
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(sub, k=1.2, seed=42)

    # Düğüm boyutu dereceye göre
    sizes = [dict(sub.degree())[n] * 300 for n in sub.nodes()]

    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color="#4169E1", alpha=0.7)
    nx.draw_networkx_edges(sub, pos, alpha=0.1, edge_color="teal")
    nx.draw_networkx_labels(sub, pos, font_size=10, font_weight="bold")
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()


# 5. Most Central Hashtags (Top 15 Bar Chart)
def plot_centrality_bar(G, dataset_name="Biden"):
    centrality = nx.degree_centrality(G)
    top_15 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]
    tags = [x[0] for x in top_15][::-1]
    scores = [x[1] for x in top_15][::-1]

    plt.figure(figsize=(16, 9))
    colors = plt.cm.Blues(np.linspace(0.2, 0.9, 15))
    bars = plt.barh(tags, scores, color=colors)

    # Skorları bar sonuna ekle
    plt.bar_label(bars, fmt='%.5f', padding=10, fontweight='bold')

    plt.title(f"Top 15 Most Central Hashtags in {dataset_name} Dataset", fontsize=16, fontweight='bold')
    plt.xlabel("Degree Centrality Score (Normalized)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.show()


# =========================
# ÇALIŞTIR
# =========================
if __name__ == "__main__":
    file = "hashtag_joebiden.csv"
    H, node_counts, total_usage = get_processed_data(file)

    plot_degree_distribution(node_counts)
    plot_lorenz_gini(node_counts, "Biden")
    plot_centrality_bar(H, "Biden")
    plot_core_network(H, node_counts)
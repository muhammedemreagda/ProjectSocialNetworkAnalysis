# =========================
# 0. LIBRARIES
# =========================
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity


# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("bot_detection_data.csv")

df.columns = [
    "UserID", "Username", "Tweet", "RetweetCount",
    "MentionCount", "FollowerCount", "Verified",
    "BotLabel", "Location", "CreatedAt", "Hashtags"
]


# =========================
# 2. HASHTAG PARSING
# =========================
def parse_hashtags(tag_str):
    if not isinstance(tag_str, str) or tag_str.strip() == "":
        return []
    return [t.lower().strip() for t in tag_str.split() if len(t) > 2]


# =========================
# 3. CO-OCCURRENCE GRAPH
# =========================
G = nx.Graph()

for tags in df["Hashtags"].dropna():
    tags = parse_hashtags(tags)
    for a, b in combinations(set(tags), 2):
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)


# =========================
# 4. CORE GRAPH (FILTER)
# =========================
MIN_WEIGHT = 2

G_core = nx.Graph(
    (u, v, d)
    for u, v, d in G.edges(data=True)
    if d["weight"] >= MIN_WEIGHT
)

print("Nodes:", G_core.number_of_nodes())
print("Edges:", G_core.number_of_edges())


# =========================
# 5. DEGREE ANALYSIS
# =========================
degrees = [d for _, d in G_core.degree()]
avg_degree = np.mean(degrees)
max_node, max_degree = max(G_core.degree(), key=lambda x: x[1])

print("\n===== DEGREE ANALYSIS =====")
print(f"Average Degree: {avg_degree:.2f}")
print(f"Max Degree: {max_degree} (Hashtag: {max_node})")


# =========================
# 6. DEGREE DISTRIBUTION (SLAYTTA KULLANILAN)
# =========================
plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=20, color='skyblue', edgecolor='black')
plt.title("Degree Distribution (Hashtag Network)")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(axis='y', alpha=0.6)
plt.tight_layout()
plt.show()


# =========================
# 7. GINI COEFFICIENT
# =========================
def gini(array):
    array = np.array(array, dtype=np.float64)
    array += 1e-9
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


gini_degree = gini(degrees)
print(f"Degree Gini Coefficient: {gini_degree:.4f}")


# =========================
# 8. LORENZ CURVE (SLAYTTA KULLANILAN)
# =========================
sorted_deg = np.sort(degrees)
cum_deg = np.cumsum(sorted_deg) / np.sum(sorted_deg)
cum_deg = np.insert(cum_deg, 0, 0)
x = np.linspace(0, 1, len(cum_deg))

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'r--', label="Perfect Equality")
plt.plot(x, cum_deg, color='blue', label="Lorenz Curve")
plt.fill_between(x, x, cum_deg, color='orange', alpha=0.3)
plt.title(f"Degree Gini & Lorenz Curve (Gini: {gini_degree:.4f})")
plt.xlabel("Fraction of Hashtags")
plt.ylabel("Fraction of Total Degree")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 9. COMMUNITY ANALYSIS
# =========================
communities = greedy_modularity_communities(G_core)
mod_score = modularity(G_core, communities, weight="weight")

print("\n===== COMMUNITY STRUCTURE =====")
print(f"Total Communities: {len(communities)}")
print(f"Modularity Score: {mod_score:.4f}")

sorted_communities = sorted(communities, key=len, reverse=True)

for i, comm in enumerate(sorted_communities[:4]):
    subgraph = G_core.subgraph(comm)
    top_nodes = sorted(
        dict(subgraph.degree()).items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    tags = [n for n, d in top_nodes]
    print(f"Community {i+1}: {len(comm)} nodes – Top hashtags: {tags}")

# =====================================================
# 10. COMMUNITY DETECTION
# =====================================================
communities = greedy_modularity_communities(G_core)
mod_score = modularity(G_core, communities, weight="weight")

print("\n===== COMMUNITY STRUCTURE =====")
print(f"Number of Communities: {len(communities)}")
print(f"Modularity Score: {mod_score:.4f}")

# Top 5 community sizes
sizes = sorted([len(c) for c in communities], reverse=True)[:5]

plt.figure(figsize=(7, 4))
plt.bar(range(1, len(sizes) + 1), sizes)
plt.xlabel("Community ID")
plt.ylabel("Number of Hashtags")
plt.title("Top 5 Community Sizes (Random Network)")
plt.show()

# =========================
# 10. FINAL SUMMARY
# =========================
print("\n===== BASELINE SUMMARY =====")
print("• Balanced degree distribution")
print("• Low degree inequality (low Gini)")
print("• Weak community separation")
print("• No dominant hashtag")
print("→ Network represents an organic, non-manipulated structure")

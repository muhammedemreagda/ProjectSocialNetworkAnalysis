🔍 Project Overview
This repository implements a network science–based framework to analyze and compare political hashtag campaigns on X (formerly Twitter).
Using the #JoeBiden and #DonaldTrump hashtag datasets as case studies, the project models political discourse as hashtag co-occurrence networks and examines their structural properties to identify signs of coordination, dominance, and inequality.
Rather than analyzing tweet content, the focus is on interaction topology, making the approach robust against semantic noise and language variation.

🧠 Datasets
hashtag_joebiden.csv
hashtag_donaldtrump.csv
Each dataset contains tweet text and user metadata collected around politically relevant hashtags.

🧪 Analysis Pipeline
The same analytical pipeline is applied to both datasets to enable fair structural comparison.

1️⃣ Data Preprocessing
Removal of missing tweets
Text normalization (lowercasing)
Hashtag extraction using regular expressions

2️⃣ Network Construction
🔹 Bipartite Network (Joe Biden Dataset)
User–Hashtag graph
Captures participation structure and hashtag adoption behavior

🔹 Hashtag Co-occurrence Network
Nodes: hashtags
Edges: co-occurrence within the same tweet
Edge weights: frequency of joint usage

3️⃣ Core Network Filtering
Weak edges removed using a minimum co-occurrence threshold (MIN_WEIGHT = 30)
Focuses analysis on structurally meaningful hashtag interactions

4️⃣ Degree Centralization
Measures the extent to which the network is dominated by a small number of hashtags
High centralization may indicate coordinated narrative control

5️⃣ Degree Distribution Analysis
Degree histograms (linear & log scale)
Used to detect heavy-tailed or star-like structures

6️⃣ Inequality Measurement
Gini Coefficient
Lorenz Curve
Quantifies concentration of influence among hashtags

7️⃣ Centrality Analysis
Degree Centrality
Identifies dominant hashtags
Betweenness Centrality
Identifies bridging hashtags that connect different narrative clusters
Computed on a core subgraph for efficiency and interpretability

8️⃣ Community Structure (Joe Biden Network)
Detected using Greedy Modularity Optimization
Modularity score used to assess narrative separation
Large, dense communities may indicate coordinated campaigns

9️⃣ Network Visualization
The repository includes multiple visual representations:
Top 10 / Top 15 degree-central hashtags
Top betweenness-central hashtags (narrative bridges)
Core hashtag networks (Top 50)
Log-scale degree distributions
Node size and color encode structural importance.

📊 Comparative Insights Enabled
This framework allows:
Structural comparison of political discourse patterns
Identification of dominant narratives
Detection of coordination signals
Measurement of inequality and influence concentration
Discovery of hashtags that bridge ideological or topical clusters

🛠️ Technologies Used
Python
pandas
NetworkX
NumPy
Matplotlib

🎯 Use Cases
Political communication research
Social network analysis coursework
Bot & coordination detection
Computational social science studies

⚠️ Notes & Limitations
Content semantics are not directly analyzed
Structural signals indicate potential coordination, not definitive proof
Results should be interpreted comparatively

🚀 Future Work
Temporal network evolution
Cross-platform comparison
Bot-labeled subnetworks
Directed retweet and mention networks

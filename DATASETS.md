# Datasets

This project uses **publicly available Twitter/X datasets** obtained from **Kaggle**.

Due to file size constraints, the raw CSV files are **not stored directly** in this repository.
Please download them manually using the source links provided below.

---

## Kaggle Dataset Sources

### 1️⃣ Donald Trump vs Joe Biden – Twitter Hashtag Dataset
- **Platform:** Kaggle  
- **Description:**  
  A collection of Twitter/X posts containing hashtag-based political discourse
  related to **#JoeBiden** and **#DonaldTrump**.
- **Link:**  
  https://www.kaggle.com/code/gatandubuc/donald-trump-vs-joe-biden/input

This Kaggle resource contains multiple CSV files used in this project.

---

### 2️⃣ General Twitter Dataset (Baseline / Bot Detection)
- **Platform:** Kaggle  
- **Description:**  
  A general-purpose Twitter dataset containing tweet text and user metadata,
  used as a **baseline and comparison network** for bot and coordination analysis.
- **Link:**  
  https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk/input

---

## Included Datasets

### 1️⃣ General Twitter Dataset
- **File:** `bot_detection_data.csv`  
- **Purpose:**  
  Represents general Twitter activity and serves as a **baseline network**
  for comparison against politically targeted hashtag campaigns.

---

### 2️⃣ Joe Biden Hashtag Dataset
- **File:** `hashtag_joebiden.csv`  
- **Purpose:**  
  Tweets and user metadata collected around the **#JoeBiden** hashtag,
  used to construct user–hashtag and hashtag co-occurrence networks
  related to Joe Biden–centered political discourse.

---

### 3️⃣ Donald Trump Hashtag Dataset
- **File:** `hashtag_donaldtrump.csv`  
- **Purpose:**  
  Tweets and user metadata collected around the **#DonaldTrump** hashtag,
  used to analyze structural dominance, coordination, centralization,
  and inequality in Trump-related political hashtag campaigns.

---

## Usage Instructions

After downloading the datasets from Kaggle:

1. Create a local `data/` directory in the project root  
2. Place the CSV files inside the directory as shown below:

project-root/
│── data/

│ ├── bot_detection_data.csv

│ ├── hashtag_joebiden.csv

│ └── hashtag_donaldtrump.csv


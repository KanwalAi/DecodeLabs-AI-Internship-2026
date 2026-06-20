# 🔍 DecoRecommender — AI Tech Stack Recommender
### Project 3 | Decode Labs AI Internship | Batch 2026

> *"We are now building systems to cure 'Choice Overload.' Recommendation engines serve as digital matchmakers, connecting users to their specific needs before those needs are explicitly articulated."*
> — Decode Labs Industrial Training Kit

---

## 📌 Project Overview

**DecoRecommender** is a content-based filtering recommendation engine built as **Project 3** of the Decode Labs AI Internship (Batch 2026).

This project demonstrates the shift from supervised classification (Project 2) to **Active Prediction** — matching a user's raw skills and career goals to the most relevant tech roles and career paths using pure mathematical similarity logic.

| Field         | Details                                        |
|---------------|------------------------------------------------|
| **Intern**    | Kanwal Fatima                                  |
| **Track**     | Artificial Intelligence (AI)                   |
| **Company**   | Decode Labs (`decodelabs.tech`)               |
| **Mode**      | Remote / Virtual                               |
| **Language**  | Python 3.12.3                                  |
| **Dataset**   | `raw_skills.csv` — 15 job roles, 116 skill tags |
| **Algorithm** | TF-IDF Vectorization + Cosine Similarity       |
| **Method**    | Content-Based Filtering                        |

---

## 🏗️ Architecture — The IPO Model

```
INPUT              →        PROCESS           →       OUTPUT
   │                           │                         │
Ingest User State         TF-IDF Vectorize          Top-N List
(min 3 skills)            Cosine Similarity         (Top 3 Roles)
                          Score all 15 roles        Match % score
                          Sort descending           Matched skills
```

### The 4-Step Ranking Pipeline

```
1. INGESTION   →   2. SCORING   →   3. SORTING   →   4. FILTERING
   Capture           Compute          Rank             Truncate to
   user skills       cos similarity   descending       Top-N (3)
```

---

## 📐 Why TF-IDF + Cosine Similarity?

### The Problem with Binary Vectors (1s and 0s)
Simple tag overlap (Jaccard Similarity) treats every skill equally. The word **"Python"** (common across 11 roles) gets the same weight as **"Kubernetes"** (specific to 3 roles). This creates noise.

### TF-IDF: The Solution
**Term Frequency-Inverse Document Frequency** penalizes generic skills and rewards specific ones.

```
TF  = (Count of skill in role) / (Total skills in role)
IDF = log(Total roles / Roles with this skill)
Weight = TF × IDF
```

A skill like "Python" appears in 11/15 roles → low IDF weight.
A skill like "Kubernetes" appears in 3/15 roles → high IDF weight.

### Why Cosine Similarity over Euclidean Distance?
Euclidean distance is sensitive to vector **magnitude** (size of profile). A user with 3 skills would score poorly against a user with 10, even if their interests perfectly align.

Cosine similarity measures the **angle** between vectors, making it invariant to profile size:

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)

Score = 1.0  →  Perfect alignment
Score = 0.0  →  No common characteristics
```

---

## 🗄️ Dataset — raw_skills.csv

15 job roles across AI, cloud, web, and systems domains:

| # | Job Role | Key Skills |
|---|----------|-----------|
| 1 | Data Scientist | Python, ML, Statistics, SQL, TensorFlow |
| 2 | Machine Learning Engineer | Python, TensorFlow, PyTorch, MLOps, Docker |
| 3 | Data Analyst | SQL, Power BI, Tableau, Statistics, Python |
| 4 | Backend Developer | Python, Java, REST APIs, SQL, Django |
| 5 | Frontend Developer | JavaScript, React, HTML, CSS, TypeScript |
| 6 | Full Stack Developer | JavaScript, React, Python, Node.js, SQL |
| 7 | DevOps Engineer | Docker, Kubernetes, AWS, CI/CD, Terraform |
| 8 | Cloud Engineer | AWS, Azure, GCP, Terraform, Kubernetes |
| 9 | Cybersecurity Analyst | Network Security, Linux, Python, SIEM |
| 10 | Mobile Developer | Flutter, Dart, React Native, Swift, Kotlin |
| 11 | AI Engineer | Python, ML, Deep Learning, NLP, LLMs |
| 12 | Data Engineer | Python, SQL, Spark, Kafka, Airflow |
| 13 | Systems Administrator | Linux, Networking, Bash, PowerShell |
| 14 | Software Architect | System Design, Microservices, AWS, Docker |
| 15 | NLP Engineer | Python, NLP, BERT, Transformers, LLMs |

---

## ✅ Project 3 Checklist

| Requirement                          | Status | Implementation |
|--------------------------------------|--------|----------------|
| **Take user input (choices/interests)** | ✅  | `input()` with min 3 skills validation |
| **Match preferences using logic/similarity** | ✅ | TF-IDF + Cosine Similarity |
| **Display recommended items**        | ✅     | Top-3 with % score, matched skills, full ranking table |
| **Content-Based Filtering**          | ✅     | User profile vs item vectors |
| **TF-IDF feature weighting**         | ✅     | `TfidfVectorizer` — penalizes generic terms |
| **4-step pipeline**                  | ✅     | Ingestion → Scoring → Sorting → Filtering |
| **Cold Start detection**             | ✅     | Zero-vector check with helpful hint |

---

## 📊 Sample Output

**Input:** `Python, Machine Learning, TensorFlow, Docker, SQL`

```
🥇  #1  AI Engineer
     Similarity : [██████████░░░░░░░░░░] 54.5%
     Matched    : ['Python', 'TensorFlow', 'Docker']

🥈  #2  Machine Learning Engineer
     Similarity : [██████████░░░░░░░░░░] 50.5%
     Matched    : ['Python', 'TensorFlow', 'Docker']

🥉  #3  Data Scientist
     Similarity : [████████░░░░░░░░░░░░] 40.4%
     Matched    : ['Python', 'TensorFlow', 'SQL']
```

---

## 🚀 How to Run

### Requirements
```bash
pip install pandas scikit-learn
```

### Run the Recommender
```bash
git clone https://github.com/KanwalAi/DecoRecommender-AI-Tech-Stack.git
cd DecoRecommender-AI-Tech-Stack
python recommend.py
```

### Example Interactions

| Skills Input | Top Result |
|---|---|
| `Python, Machine Learning, SQL` | Data Scientist |
| `AWS, Docker, Kubernetes, Terraform` | DevOps Engineer |
| `JavaScript, React, Node.js` | Full Stack Developer |
| `Python, NLP, BERT, Transformers` | NLP Engineer |

---

## 📂 Project Structure

```
project3_recommendation/
│
├── recommend.py        # Main recommender — full pipeline
├── raw_skills.csv      # Knowledge base — 15 roles × 116 skills
├── README.md           # This file
└── screenshots/        # Demo screenshots
```

---

## 🔬 Key Concepts Demonstrated

| Concept | Where Used |
|---|---|
| **Content-Based Filtering** | Matching user profile to item attributes |
| **TF-IDF Vectorization** | `TfidfVectorizer` — frequency-weighted features |
| **Cosine Similarity** | `cosine_similarity()` — magnitude-invariant matching |
| **Cold Start Detection** | Zero-vector check + user guidance |
| **Top-N Filtering** | Truncate output to prevent choice overload |
| **4-Step Pipeline** | Ingestion → Scoring → Sorting → Filtering |
| **Input Validation** | Minimum 3 skills enforced in loop |

---

## ⚠️ The Cold Start Problem

Content-based filtering is inherently robust to **Item Cold Start** — new roles can be recommended immediately via their metadata tags.

For **User Cold Start** (zero-skill input), the script detects a zero cosine similarity vector and prompts the user to try different skill spellings, bypassing the failure gracefully.

---

## 🎓 Learning Outcomes

- ✅ Built a content-based recommendation engine from scratch
- ✅ Applied TF-IDF to weight specific skills over generic ones
- ✅ Used cosine similarity for magnitude-invariant profile matching
- ✅ Implemented the 4-step ranking pipeline (Ingest → Score → Sort → Filter)
- ✅ Handled the Cold Start problem with detection and fallback

---

## 👩‍💻 Author

**Kanwal Fatima**
AI Student | Software Developer
📧 kanwal.ai.pk@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/kanwal-fatima-72a352357/)
🐙 [GitHub](https://github.com/KanwalAi)

---

## 🏢 About Decode Labs

**Decode Labs** — Your Digital Lab
🌐 [www.decodelabs.tech](https://www.decodelabs.tech)
✉️ hr@decodelabs.tech
📍 Greater Lucknow, India
✦ Govt. Registered Enterprise

---

*Project 3 of 3 — AI Recommendation Logic | Decode Labs AI Internship 2026*

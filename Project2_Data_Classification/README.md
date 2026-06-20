# 🧠 DecoClassifier — Data Classification Using AI
### Project 2 | Decode Labs AI Internship | Batch 2026

> *"We do not write the rules. We provide history, and the machine derives the logic."*
> — Decode Labs Industrial Training Kit

---

## 📌 Project Overview

**DecoClassifier** is a supervised machine learning classifier built as **Project 2** of the Decode Labs AI Internship (Batch 2026).

This project demonstrates the shift from rule-based logic (Project 1) to **Supervised Learning** — teaching a machine to recognize patterns in data and categorize new information based on what it has learned.

| Field         | Details                           |
|---------------|-----------------------------------|
| **Intern**    | Kanwal Fatima                     |
| **Track**     | Artificial Intelligence (AI)      |
| **Company**   | Decode Labs (`decodelabs.tech`)   |
| **Mode**      | Remote / Virtual                  |
| **Language**  | Python 3.12.3                     |
| **Dataset**   | Iris Benchmark                    |
| **Algorithm** | K-Nearest Neighbors (KNN, K=5)    |

---

## 🏗️ Architecture — The IPO Framework

```
INPUT          →       PROCESS        →      OUTPUT
   │                      │                     │
Load Iris            Scale Features        Confusion Matrix
Feature Scaling      Train/Test Split      F1 Score
                     KNN Train & Predict   Classification Report
```

### Phase 1 — Input (Load & Understand Dataset)
The **Iris Benchmark** dataset: 150 samples, 3 classes, 4 features.

| Property   | Value |
|------------|-------|
| Samples    | 150 (50 per class, balanced) |
| Classes    | 3 → Setosa, Versicolor, Virginica |
| Features   | 4 → Sepal Length, Sepal Width, Petal Length, Petal Width |

### Phase 2 — Process (Scale → Split → Train)

**Step 1 — Feature Scaling**
```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)   # mean=0, variance=1
```
Removes bias caused by different feature magnitudes.

**Step 2 — Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)
# → 120 training samples | 30 test samples
```

**Step 3 — KNN: Instantiate → Fit → Predict**
```python
model  = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Phase 3 — Output (Validate Results)
- **Confusion Matrix** — visualises TP, FP, FN, TN per class
- **Accuracy Score** — overall correct predictions
- **F1 Score (weighted)** — balances precision and recall
- **Classification Report** — per-class precision, recall, F1

---

## ✅ Project 2 Checklist

| Requirement                        | Status | Implementation |
|------------------------------------|--------|----------------|
| **Load and understand a dataset**  | ✅     | `load_iris()` with shape & feature info printed |
| **Split into train & test sets**   | ✅     | `train_test_split()` — 80/20, shuffled |
| **Apply classification algorithm** | ✅     | `KNeighborsClassifier(n_neighbors=5)` |
| **Feature Scaling**                | ✅     | `StandardScaler` — mean=0, var=1 |
| **Confusion Matrix**               | ✅     | Printed per class |
| **F1 Score**                       | ✅     | Weighted F1 score |

---

## 📊 Results

```
Confusion Matrix:
              setosa  versicolor  virginica
    setosa        10           0          0
versicolor         0           9          0
 virginica         0           0         11

Accuracy  : 100.00%
F1 Score  : 1.0000 (weighted)
```

---

## 🚀 How to Run

### Requirements
```bash
pip install scikit-learn
```

### Run the Classifier
```bash
git clone https://github.com/KanwalAi/DecoClassifier-Data-Classification.git
cd DecoClassifier-Data-Classification
python classify.py
```

---

## 📂 Project Structure

```
project2_classification/
│
├── classify.py       # Main classifier — all logic lives here
├── README.md         # This file
└── screenshots/      # Demo screenshots
```

---

## 🔬 Key Concepts Demonstrated

| Concept                  | Where Used |
|--------------------------|------------|
| **Supervised Learning**  | KNN trained on labelled Iris data |
| **Feature Scaling**      | `StandardScaler` — removes magnitude bias |
| **Train/Test Split**     | 80/20 split with shuffle |
| **KNN Algorithm**        | `KNeighborsClassifier(n_neighbors=5)` |
| **Confusion Matrix**     | TP / FP / FN / TN per class |
| **F1 Score**             | Weighted harmonic mean of precision & recall |
| **IPO Model**            | Input → Process → Output pipeline |

---

## 📊 Why KNN?

KNN works on the **Proximity Principle** — similar data points exist in close proximity.  
For each new sample, it finds the **K=5 nearest neighbours** and assigns the majority class.

| K Value | Risk |
|---------|------|
| K=1     | Overfitting (noise-sensitive) |
| **K=5** | **Optimal (The Elbow)** |
| K=100   | Underfitting (too generic) |

---

## 📸 Screenshots

> *See `/screenshots` folder in this repo.*

---

## 🎓 Learning Outcomes

- ✅ Loaded and explored a real-world benchmark dataset
- ✅ Applied feature scaling to remove magnitude bias
- ✅ Split data into training and testing sets
- ✅ Trained a KNN classifier using scikit-learn
- ✅ Evaluated model using Confusion Matrix and F1 Score
- ✅ Understood why accuracy alone can be misleading

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

*Project 2 of 3 — Data Classification Using AI | Decode Labs AI Internship 2026*

# 🤖 DecodeLabs AI Engineering Internship — 2026

### Kanwal Fatima | Artificial Intelligence Track

A complete portfolio of 4 hands-on AI projects built during the **DecodeLabs AI Engineering Internship (Batch 2026)** — progressing from rule-based logic to supervised learning, recommendation systems, and computer vision.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Internship-Completed-success)
![Projects](https://img.shields.io/badge/Projects-4%2F4-brightgreen)
![Certificate](https://img.shields.io/badge/Certificate-Verified-blue)
![Recommendation](https://img.shields.io/badge/Recommendation-Received-success)

---

## 📋 Project Index

| # | Project | Core Concept | Tech Stack |
|---|---------|---------------|------------|
| 1 | [Rule-Based AI Chatbot](./Project1_RuleBased_Chatbot) | Control flow & decision logic | Python, Dictionaries |
| 2 | [Data Classification](./Project2_Data_Classification) | Supervised learning (KNN) | Scikit-Learn, Iris Dataset |
| 3 | [AI Recommendation Logic](./Project3_Recommendation_System) | Content-based filtering | TF-IDF, Cosine Similarity |
| 4 | [Image & Text Recognition](./Project4_Image_Text_Recognition) | Computer vision & OCR | OpenCV, Tesseract, MobileNet-
SSD |

---

## 🗂️ Project Details

### 1️⃣ Rule-Based AI Chatbot
**Goal:** Build a chatbot that responds to predefined inputs using pure `if-else`/dictionary logic — no ML involved.

- Continuous `while True` input loop with a clean exit strategy
- Input sanitization (`.lower().strip()`)
- 20+ intent knowledge base with keyword-fallback matching
- Demonstrates: **Control flow, decision-making logic, basic AI concepts**

📁 [`Project1_RuleBased_Chatbot/`](./Project1_RuleBased_Chatbot) → `chatbot.py`

---

### 2️⃣ Data Classification Using AI
**Goal:** Train a supervised classification model on a real dataset and evaluate it properly.

- **Dataset:** Iris (150 samples, 3 classes, 4 features)
- **Pipeline:** Feature scaling (`StandardScaler`) → Train/Test split (80/20) → `KNeighborsClassifier` (K=5)
- **Evaluation:** Confusion matrix, F1 score, full classification report
- **Result:** 100% accuracy, F1 = 1.0000 on the test set
- Demonstrates: **Data handling, supervised learning basics, model training**

📁 [`Project2_Data_Classification/`](./Project2_Data_Classification) → `classify.py`

---

### 3️⃣ AI Recommendation Logic
**Goal:** Build a content-based recommendation engine matching user skills to career paths.

- **Dataset:** `raw_skills.csv` — 15 job roles × 116 unique skill tags (self-built)
- **Algorithm:** TF-IDF vectorization (penalizes generic skills, rewards specific ones) + Cosine Similarity (magnitude-invariant matching)
- **Pipeline:** Ingestion → Scoring → Sorting → Filtering (Top-3 output)
- Includes Cold Start detection for unmatched input
- Demonstrates: **Logic building, pattern matching, recommendation concepts**

📁 [`Project3_Recommendation_System/`](./Project3_Recommendation_System) → `recommend.py`

---

### 4️⃣ Image & Text Recognition (Optional Mastery Milestone)
**Goal:** Implement both Optical Character Recognition and Object Detection using pre-trained models.

- **Path 1 — OCR:** Full pre-processing pipeline (Grayscale → Gaussian Blur → Hough-line Deskew → Otsu Adaptive Threshold) feeding into `pytesseract`. Achieved 89.6%–93.8% confidence on noisy, skewed test scans.
- **Path 2 — Object Detection:** Transfer learning with pre-trained **MobileNet-SSD** (Caffe, VOC-trained) via `cv2.dnn`. Detected horse (100%), car (99.3%), and person (95.8%) on a multi-object test scene, with an 80% confidence gate correctly filtering out weaker detections.
- Demonstrates: **Using AI libraries, understanding model outputs, transfer learning**

📁 [`Project4_Image_Text_Recognition/`](./Project4_Image_Text_Recognition) → `ocr_recognition.py`, `object_detection.py`

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML / Data | Scikit-Learn, Pandas, NumPy |
| Computer Vision | OpenCV (`cv2.dnn`), Tesseract OCR (`pytesseract`) |
| Concepts | Control Flow, Supervised Learning (KNN), TF-IDF + Cosine Similarity, Transfer Learning (MobileNet-SSD) |

---

## 🚀 Getting Started

Each project folder is self-contained with its own `README.md` and run instructions. General setup:

```bash
git clone https://github.com/KanwalAi/DecodeLabs-AI-Internship-2026.git
cd DecodeLabs-AI-Internship-2026

# Install shared dependencies
pip install scikit-learn pandas numpy opencv-python pytesseract

# System dependency for Project 4 (OCR engine)
sudo apt-get install tesseract-ocr   # Linux
# brew install tesseract              # Mac
# or download from: https://github.com/UB-Mannheim/tesseract/wiki  (Windows)
```

Then `cd` into any project folder and follow its individual `README.md`.

---

## 🎓 Learning Journey

This internship moved progressively from **deterministic logic** → **statistical learning** → **similarity-based matching** → **perceptual AI**:

```
Project 1            Project 2              Project 3                Project 4
Rule-Based     →     Supervised      →      Content-Based      →     Computer Vision
(if-else)            Learning (KNN)         Filtering (TF-IDF)        (OCR + Detection)
```

Each milestone builds directly on the last — from teaching a machine to follow explicit rules, to teaching it to recognize patterns in structured data, to matching unstructured preferences, to finally interpreting raw pixels and scanned documents.

---

# 🏆 Recognition

The successful completion of this internship is recognized through the following official documents issued by **DecodeLabs**.

### 📜 AI Engineering Internship Certificate

Successfully completed the **AI Engineering Internship (Batch 2026)**, demonstrating practical skills in Artificial Intelligence through four hands-on projects covering rule-based systems, machine learning, recommendation systems, and computer vision.

📄 **View Certificate:**  
[AI Engineering Internship Certificate](./documents/AI_Engineering_Internship_Certificate.pdf)

---

### 💼 Letter of Recommendation

Received an official **Letter of Recommendation** from DecodeLabs in recognition of my dedication, technical performance, and successful completion of the internship program.

📄 **View Recommendation Letter:**  
[Letter of Recommendation](./documents/DecodeLabs_Recommendation_Letter.pdf)

---

## 👩‍💻 Author

**Kanwal Fatima**

BS Artificial Intelligence Student

Passionate about building intelligent systems that combine AI, robotics, computer vision, and embedded technologies.

📧 kanwal.ai.pk@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/kanwal-fatima-72a352357/

🐙 GitHub: https://github.com/KanwalAi

---

## 🏢 About DecodeLabs

**DecodeLabs** — Your Digital Lab
🌐 [www.decodelabs.tech](https://www.decodelabs.tech)
✉️ decodelabs.tech@gmail.com
📍 Greater Lucknow, India

---

*Completed as part of the DecodeLabs Artificial Intelligence Industrial Training Program, Batch 2026.*

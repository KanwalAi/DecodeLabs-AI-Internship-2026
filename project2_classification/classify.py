# ============================================================
# Project 2: Data Classification Using AI
# Intern   : Kanwal Fatima
# Company  : Decode Labs  |  Batch: 2026
# Track    : Artificial Intelligence (AI)
# Dataset  : Iris Benchmark (150 samples, 3 classes, 4 features)
# Algorithm: K-Nearest Neighbors (KNN)
# ============================================================

from sklearn.datasets        import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.metrics         import (classification_report,
                                     confusion_matrix,
                                     f1_score,
                                     accuracy_score)

# ── BANNER ──────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════╗
║      DecoClassifier — Data Classification Using AI  ║
║   Decode Labs  |  Project 2  |  Batch 2026          ║
║   Built by: Kanwal Fatima                           ║
╠══════════════════════════════════════════════════════╣
║  Dataset: Iris  |  Algorithm: KNN  |  Split: 80/20  ║
╚══════════════════════════════════════════════════════╝
"""

def separator(title=""):
    line = "─" * 54
    if title:
        print(f"\n  ┌{line}┐")
        print(f"  │  {title:<52}│")
        print(f"  └{line}┘")
    else:
        print(f"  {'─'*54}")

# ── PHASE 1: INPUT — Load & Understand the Dataset ──────────
def load_data():
    separator("PHASE 1 · INPUT — Loading the Iris Dataset")

    iris   = load_iris()
    X      = iris.data          # features: sepal/petal length & width
    y      = iris.target        # labels : 0=Setosa, 1=Versicolor, 2=Virginica
    names  = iris.target_names
    feats  = iris.feature_names

    print(f"\n  📦 Dataset     : Iris Benchmark")
    print(f"  📊 Samples     : {X.shape[0]}")
    print(f"  🔢 Features    : {X.shape[1]}")
    print(f"  🏷️  Classes     : {len(names)}  →  {list(names)}")
    print(f"\n  Features used:")
    for i, f in enumerate(feats):
        print(f"    [{i+1}] {f}")

    return X, y, names

# ── PHASE 2: PROCESS — Scale → Split → Train ────────────────
def process(X, y):
    separator("PHASE 2 · PROCESS — Scale  →  Split  →  Train")

    # Step 1: Feature Scaling (StandardScaler → mean=0, variance=1)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n  ✅ Scaling      : StandardScaler applied (mean=0, var=1)")

    # Step 2: Train-Test Split (80% train / 20% test, shuffled)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = 0.2,
        random_state = 42,
        shuffle      = True
    )
    print(f"  ✅ Split        : {len(X_train)} train  |  {len(X_test)} test  (80/20)")

    # Step 3: KNN Model — Instantiate → Fit → Predict
    model = KNeighborsClassifier(n_neighbors=5)   # K = 5 (The Elbow)
    model.fit(X_train, y_train)                   # FIT   — memorise the map
    y_pred = model.predict(X_test)                # PREDICT — apply logic
    print(f"  ✅ Algorithm    : KNeighborsClassifier  (K=5)")
    print(f"  ✅ Model trained and predictions generated")

    return y_test, y_pred

# ── PHASE 3: OUTPUT — Validate Results ──────────────────────
def output(y_test, y_pred, class_names):
    separator("PHASE 3 · OUTPUT — Confusion Matrix  +  F1 Score")

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  📊 Confusion Matrix:")
    print(f"  {'':>12}", end="")
    for n in class_names:
        print(f"  {n:>10}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>12}  ", end="")
        for val in row:
            print(f"  {val:>10}", end="")
        print()

    # Metrics
    print(f"\n  📈 Accuracy     : {acc * 100:.2f}%")
    print(f"  🎯 F1 Score     : {f1:.4f}  (weighted)")

    # Classification Report
    print(f"\n  📋 Classification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names)
    for line in report.splitlines():
        print(f"     {line}")

# ── ENTRY POINT ─────────────────────────────────────────────
def main():
    print(BANNER)

    X, y, class_names = load_data()
    y_test, y_pred    = process(X, y)
    output(y_test, y_pred, class_names)

    separator("DONE")
    print("\n  ✅ Project 2 complete — Model trained, tested & validated.")

if __name__ == "__main__":
    main()

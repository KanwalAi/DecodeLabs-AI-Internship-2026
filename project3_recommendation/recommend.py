# ============================================================
# Project 3: AI Recommendation Logic — Tech Stack Recommender
# Intern   : Kanwal Fatima
# Company  : Decode Labs  |  Batch: 2026
# Track    : Artificial Intelligence (AI)
# Dataset  : raw_skills.csv (15 job roles, skill tags)
# Algorithm: TF-IDF Vectorization + Cosine Similarity
# Pipeline : Ingestion → Scoring → Sorting → Filtering
# ============================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── CONSTANTS ────────────────────────────────────────────────
TOP_N      = 3    # Top-N recommendations to display
MIN_SKILLS = 3    # Minimum inputs for sufficient data density

BANNER = """
╔════════════════════════════════════════════════════════════╗
║    DecoRecommender — AI Tech Stack Recommender            ║
║    Decode Labs  |  Project 3  |  Batch 2026               ║
║    Built by: Kanwal Fatima                                ║
╠════════════════════════════════════════════════════════════╣
║    Algorithm  : TF-IDF + Cosine Similarity                ║
║    Method     : Content-Based Filtering                   ║
║    Pipeline   : Ingestion → Scoring → Sorting → Filtering ║
╚════════════════════════════════════════════════════════════╝
"""

def separator(title=""):
    line = "─" * 58
    if title:
        print(f"\n  ┌{line}┐")
        print(f"  │  {title:<56}│")
        print(f"  └{line}┘")
    else:
        print(f"  {'─'*58}")


# ── LOAD DATASET ─────────────────────────────────────────────
def load_dataset(path: str = "raw_skills.csv") -> pd.DataFrame:
    """Load the job roles knowledge base from CSV."""
    separator("DATASET — Loading raw_skills.csv")
    df = pd.read_csv(path)
    print(f"\n  ✅ Loaded     : {len(df)} job roles")
    print(f"  ✅ Columns    : {list(df.columns)}")
    print(f"\n  Roles available:")
    for i, role in enumerate(df['job_role'], 1):
        print(f"    [{i:>2}] {role}")
    return df


# ── PHASE 1: INPUT — Ingestion ────────────────────────────────
def ingest_user_profile() -> list[str]:
    """
    Pipeline Step 1: Ingestion.
    Capture the user state — minimum 3 skills required
    for sufficient data density in the similarity math.
    """
    separator("PHASE 1 · INPUT — Ingesting User Profile")

    print(f"\n  Enter your skills/technologies (comma-separated).")
    print(f"  Minimum {MIN_SKILLS} skills required for accurate matching.")
    print(f"  Example: Python, Machine Learning, SQL, Docker, AWS\n")

    while True:
        raw = input("  Your skills: ")
        skills = [s.strip() for s in raw.split(',') if s.strip()]

        if len(skills) < MIN_SKILLS:
            print(f"\n  ⚠️  Please enter at least {MIN_SKILLS} skills. "
                  f"You entered {len(skills)}.\n")
        else:
            print(f"\n  ✅ Profile ingested  : {len(skills)} skills captured")
            print(f"  ✅ Skills            : {skills}")
            return skills


# ── PHASE 2: PROCESS — Scoring ────────────────────────────────
def build_vectors_and_score(
    user_skills: list[str],
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Pipeline Steps 1-2: Build TF-IDF vectors + compute cosine similarity.

    Why TF-IDF over binary vectors?
      Binary vectors treat 'Python' and 'Kubernetes' equally.
      TF-IDF weights rare, specific tags more heavily —
      rewarding niche skill matches over common ones.

    Why Cosine Similarity over Euclidean distance?
      Cosine measures the angle between vectors, not their length.
      This makes it invariant to profile size — a user with
      3 skills gets a fair comparison against one with 10.
    """
    separator("PHASE 2 · PROCESS — TF-IDF Vectorization + Cosine Similarity")

    # Step 1: Fit TF-IDF on the entire job role corpus
    vectorizer  = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['skills'])
    print(f"\n  ✅ TF-IDF matrix    : shape {tfidf_matrix.shape}  "
          f"({tfidf_matrix.shape[0]} roles × {tfidf_matrix.shape[1]} unique terms)")

    # Step 2: Project user profile into the same vocabulary space
    user_profile_str = ' '.join(user_skills)
    user_vector      = vectorizer.transform([user_profile_str])
    print(f"  ✅ User vector      : projected into shared vocabulary space")

    # Step 3: Compute cosine similarity  (Scoring step of the pipeline)
    #   cos(θ) = (A · B) / (‖A‖ · ‖B‖)
    #   Score = 1.0 → perfect alignment | Score = 0 → no overlap
    scores = cosine_similarity(user_vector, tfidf_matrix)[0]
    df     = df.copy()
    df['similarity'] = scores
    print(f"  ✅ Cosine scores    : computed for all {len(df)} job roles")

    # Step 4: Sort descending by similarity score  (Sorting step)
    df_sorted = df.sort_values('similarity', ascending=False).reset_index(drop=True)
    print(f"  ✅ Roles sorted     : highest similarity first")

    return df_sorted


# ── PHASE 3: OUTPUT — Top-N Filtering + Display ───────────────
def display_recommendations(df_ranked: pd.DataFrame, user_skills: list[str]):
    """
    Pipeline Steps 3-4: Sort + Filter to generate Top-N list.

    Truncating to Top-N prevents choice overload —
    a core design principle from the IPO architecture.
    """
    separator(f"PHASE 3 · OUTPUT — Top {TOP_N} Career Path Recommendations")

    # Check for cold start (zero vector — no skill overlaps at all)
    if df_ranked['similarity'].max() == 0:
        print("\n  ⚠️  Cold Start detected: no skills matched the vocabulary.")
        print("  💡 Try different spellings — e.g. 'Machine Learning' not 'ML'.")
        return

    top_n  = df_ranked.head(TOP_N)   # Step 4: Filtering
    medals = ["🥇", "🥈", "🥉"]

    print(f"\n  Input skills : {user_skills}")
    print(f"\n  Your Top {TOP_N} recommended career paths:\n")

    for i, (_, row) in enumerate(top_n.iterrows()):
        score_pct  = row['similarity'] * 100
        bar_filled = int(score_pct / 5)
        bar        = "█" * bar_filled + "░" * (20 - bar_filled)

        # Find overlapping skills between user profile and this role
        role_skills = set(row['skills'].lower().split())
        user_set    = set(s.lower() for s in user_skills)
        matched     = [s for s in user_skills if s.lower() in role_skills]

        print(f"  {medals[i]}  #{i+1}  {row['job_role']}")
        print(f"       Similarity : [{bar}] {score_pct:.1f}%")
        print(f"       Matched    : {matched if matched else ['(weighted partial overlap)']}")
        print(f"       Full stack : {row['skills']}")
        print()

    # Show full ranking table
    print(f"\n  📊 Full similarity ranking:")
    print(f"  {'#':<4} {'Job Role':<30} {'Score':>8}")
    print(f"  {'─'*4} {'─'*30} {'─'*8}")
    for i, (_, row) in enumerate(df_ranked.iterrows(), 1):
        bar = "▓" * int(row['similarity'] * 20)
        print(f"  {i:<4} {row['job_role']:<30} {row['similarity']:>7.4f}  {bar}")


# ── ENTRY POINT ─────────────────────────────────────────────
def main():
    print(BANNER)

    # Load the knowledge base
    df = load_dataset("raw_skills.csv")

    # Run the 4-step recommendation pipeline
    user_skills = ingest_user_profile()
    df_ranked   = build_vectors_and_score(user_skills, df)
    display_recommendations(df_ranked, user_skills)

    separator("DONE")
    print("\n  ✅ Project 3 complete — Tech Stack Recommendations generated.\n")


if __name__ == "__main__":
    main()

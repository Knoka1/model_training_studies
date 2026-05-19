"""
Logistic regression to identify keywords that drive a PURSUE label.
Features: TF-IDF over Title + Description + Documents AI Summary
Target:   Label (DROP / PURSUE / REVIEW)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

# ── 1. Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv("datatsets/keywords_hg_labeled.csv")

TEXT_COLS = ["Title", "Description", "Documents AI Summary"]
df["text"] = (
    df[TEXT_COLS]
    .fillna("")
    .apply(lambda row: " ".join(row.values), axis=1)
    .str.strip()
)

X = df["text"]
y = df["Label"]

print(f"Dataset: {len(df)} rows | Labels: {y.value_counts().to_dict()}\n")

# ── 2. Pipeline ───────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )),
])

# ── 3. Cross-validation ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro")
print(f"5-fold CV  macro-F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n")

# ── 4. Fit on full dataset ────────────────────────────────────────────────────
pipeline.fit(X, y)
tfidf = pipeline.named_steps["tfidf"]
clf   = pipeline.named_steps["clf"]
feature_names = np.array(tfidf.get_feature_names_out())
classes       = clf.classes_          # ['DROP', 'PURSUE', 'REVIEW']

print("Classification report (train):")
print(classification_report(y, pipeline.predict(X)))

# ── 5. Top PURSUE keywords ────────────────────────────────────────────────────
pursue_idx  = list(classes).index("PURSUE")
coef_pursue = clf.coef_[pursue_idx]

n_top = 30
top_idx  = np.argsort(coef_pursue)[::-1][:n_top]
bot_idx  = np.argsort(coef_pursue)[:n_top]

top_df = pd.DataFrame({
    "keyword":     feature_names[top_idx],
    "coefficient": coef_pursue[top_idx],
    "direction":   "PURSUE",
})
bot_df = pd.DataFrame({
    "keyword":     feature_names[bot_idx],
    "coefficient": coef_pursue[bot_idx],
    "direction":   "AGAINST",
})

print("\n── Top 30 keywords → PURSUE ──────────────────────────────────────")
print(top_df.to_string(index=False))
print("\n── Top 30 keywords AGAINST PURSUE (toward DROP/REVIEW) ──────────")
print(bot_df.to_string(index=False))

# ── 6. Save keyword tables ────────────────────────────────────────────────────
all_keywords = pd.concat([top_df, bot_df], ignore_index=True)
all_keywords.to_csv("pursue_keywords.csv", index=False)
print("\nSaved: pursue_keywords.csv")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Logistic Regression — Keywords Driving PURSUE Label", fontsize=14)

# PURSUE-positive keywords
ax = axes[0]
colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in top_df["coefficient"]]
ax.barh(top_df["keyword"][::-1], top_df["coefficient"][::-1], color=colors[::-1])
ax.set_title("Top 30 Keywords → PURSUE")
ax.set_xlabel("Logistic Regression Coefficient")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

# PURSUE-negative keywords
ax = axes[1]
colors = ["#e74c3c" for _ in bot_df["coefficient"]]
ax.barh(bot_df["keyword"][::-1], bot_df["coefficient"][::-1], color=colors[::-1])
ax.set_title("Top 30 Keywords AGAINST PURSUE")
ax.set_xlabel("Logistic Regression Coefficient")
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig("pursue_keywords.png", dpi=150, bbox_inches="tight")
print("Saved: pursue_keywords.png")

# Confusion matrix
fig2, ax2 = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y, pipeline.predict(X), labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("Confusion Matrix (full-train fit)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved: confusion_matrix.png")

plt.show()

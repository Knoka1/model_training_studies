"""
Compare logistic regression keyword models for v1 vs v2 labels.
Also reports label noise (relabeled rows) for PURSUE and REVIEW.
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

TEXT_COLS = ["Title", "Description", "Documents AI Summary"]
CLASSES   = ["DROP", "PURSUE", "REVIEW"]
N_TOP     = 30

# ── helpers ───────────────────────────────────────────────────────────────────

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.9,
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )),
    ])

def prep_text(df):
    return df[TEXT_COLS].fillna("").apply(lambda r: " ".join(r.values), axis=1).str.strip()

def top_keywords(pipeline, n=N_TOP):
    tfidf = pipeline.named_steps["tfidf"]
    clf   = pipeline.named_steps["clf"]
    feat  = np.array(tfidf.get_feature_names_out())
    coef  = {cls: clf.coef_[i] for i, cls in enumerate(clf.classes_)}
    result = {}
    for cls in CLASSES:
        c = coef[cls]
        top = pd.DataFrame({"keyword": feat[np.argsort(c)[::-1][:n]], "coefficient": np.sort(c)[::-1][:n]})
        bot = pd.DataFrame({"keyword": feat[np.argsort(c)[:n]],       "coefficient": np.sort(c)[:n]})
        result[cls] = (top, bot)
    return result

def run_model(df, label_col, version_tag):
    X = prep_text(df)
    y = df[label_col]
    pipe = build_pipeline()
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
    pipe.fit(X, y)
    return pipe, scores, X, y

# ── load data ─────────────────────────────────────────────────────────────────

v1 = pd.read_csv("datatsets/keywords_hg_labeled.csv")
v2 = pd.read_csv("datatsets/keywords_hg_labeled_v2.csv")

# ── noise analysis ────────────────────────────────────────────────────────────

noise = v2.copy()
noise["changed"] = noise["Label"] != noise["Old_Label"]

def noise_stats(df_noise, label_col, version_tag):
    rows = []
    for lbl in ["PURSUE", "REVIEW"]:
        total   = (df_noise[label_col] == lbl).sum()
        changed = ((df_noise[label_col] == lbl) & df_noise["changed"]).sum()
        pct     = 100 * changed / total if total else 0
        rows.append({"version": version_tag, "label": lbl,
                     "total": total, "relabeled": changed, "noise_%": round(pct, 1)})
    return pd.DataFrame(rows)

noise_v1 = noise_stats(noise, "Old_Label", "V1")
noise_v2 = noise_stats(noise, "Label",     "V2")
noise_df  = pd.concat([noise_v1, noise_v2], ignore_index=True)

print("=" * 60)
print("LABEL NOISE (rows that changed between V1 → V2)")
print("=" * 60)
print(noise_df.to_string(index=False))

# breakdown of where changes went
print("\nV1 PURSUE relabeled to:")
print(noise[(noise["Old_Label"] == "PURSUE") & noise["changed"]]["Label"].value_counts().to_string())
print("\nV1 REVIEW relabeled to:")
print(noise[(noise["Old_Label"] == "REVIEW") & noise["changed"]]["Label"].value_counts().to_string())
print("\nV2 PURSUE sourced from (Old_Label):")
print(noise[(noise["Label"] == "PURSUE") & noise["changed"]]["Old_Label"].value_counts().to_string())
print("\nV2 REVIEW sourced from (Old_Label):")
print(noise[(noise["Label"] == "REVIEW") & noise["changed"]]["Old_Label"].value_counts().to_string())

# ── run models ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("MODEL V1")
print("=" * 60)
pipe_v1, scores_v1, X_v1, y_v1 = run_model(v1, "Label", "V1")
print(f"5-fold CV macro-F1: {scores_v1.mean():.3f} ± {scores_v1.std():.3f}")
print(classification_report(y_v1, pipe_v1.predict(X_v1)))

print("=" * 60)
print("MODEL V2")
print("=" * 60)
pipe_v2, scores_v2, X_v2, y_v2 = run_model(v2, "Label", "V2")
print(f"5-fold CV macro-F1: {scores_v2.mean():.3f} ± {scores_v2.std():.3f}")
print(classification_report(y_v2, pipe_v2.predict(X_v2)))

# ── keyword comparison ────────────────────────────────────────────────────────

kw_v1 = top_keywords(pipe_v1)
kw_v2 = top_keywords(pipe_v2)

for label in CLASSES:
    top1, _ = kw_v1[label]
    top2, _ = kw_v2[label]
    merged = top1.rename(columns={"coefficient": "coef_v1"}).merge(
             top2.rename(columns={"coefficient": "coef_v2"}), on="keyword", how="outer"
    ).fillna(0).sort_values("coef_v2", ascending=False)
    merged.to_csv(f"keywords_{label.lower()}_compare.csv", index=False)
    print(f"\n── Top keywords for {label} (V1 vs V2) ──────────────────")
    print(merged.head(20).to_string(index=False))

# ── save keyword CSV ──────────────────────────────────────────────────────────

pursue_top_v1, pursue_bot_v1 = kw_v1["PURSUE"]
pursue_top_v2, pursue_bot_v2 = kw_v2["PURSUE"]
pd.concat([
    pursue_top_v1.assign(direction="PURSUE", version="V1"),
    pursue_bot_v1.assign(direction="AGAINST", version="V1"),
    pursue_top_v2.assign(direction="PURSUE", version="V2"),
    pursue_bot_v2.assign(direction="AGAINST", version="V2"),
], ignore_index=True).to_csv("pursue_keywords_compare.csv", index=False)
print("\nSaved: pursue_keywords_compare.csv")

# ── plots ─────────────────────────────────────────────────────────────────────

# 1. Noise bar chart
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(2)
w = 0.3
v1_pct = noise_v1["noise_%"].values
v2_pct = noise_v2["noise_%"].values
bars1 = ax.bar(x - w/2, v1_pct, w, label="V1 (original)", color="#3498db")
bars2 = ax.bar(x + w/2, v2_pct, w, label="V2 (revised)",  color="#e67e22")
ax.set_xticks(x); ax.set_xticklabels(["PURSUE", "REVIEW"])
ax.set_ylabel("% relabeled"); ax.set_title("Label Noise: % of PURSUE/REVIEW rows that changed")
ax.legend()
for bar in bars1: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
for bar in bars2: ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("noise_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: noise_comparison.png")

# 2. PURSUE keyword comparison (side-by-side V1 vs V2)
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("PURSUE Keywords: V1 vs V2 — Top driving & against", fontsize=14)

for col_idx, (version, top, bot) in enumerate([
    ("V1", pursue_top_v1, pursue_bot_v1),
    ("V2", pursue_top_v2, pursue_bot_v2),
]):
    ax = axes[0][col_idx]
    ax.barh(top["keyword"][::-1], top["coefficient"][::-1], color="#2ecc71")
    ax.set_title(f"{version} — Top {N_TOP} → PURSUE"); ax.set_xlabel("Coefficient")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax = axes[1][col_idx]
    ax.barh(bot["keyword"][::-1], bot["coefficient"][::-1], color="#e74c3c")
    ax.set_title(f"{version} — Top {N_TOP} AGAINST PURSUE"); ax.set_xlabel("Coefficient")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

plt.tight_layout()
plt.savefig("pursue_keywords_compare.png", dpi=150, bbox_inches="tight")
print("Saved: pursue_keywords_compare.png")

# 3. CV score comparison
fig, ax = plt.subplots(figsize=(5, 4))
labels_bar = ["V1", "V2"]
means = [scores_v1.mean(), scores_v2.mean()]
errs  = [scores_v1.std(),  scores_v2.std()]
colors = ["#3498db", "#e67e22"]
bars = ax.bar(labels_bar, means, yerr=errs, capsize=6, color=colors, width=0.4)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, m + 0.005, f"{m:.3f}", ha="center", fontsize=10)
ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1 (5-fold CV)")
ax.set_title("Model Quality: V1 vs V2")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: model_comparison.png")

# 4. Confusion matrices side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, pipe, y, title in [
    (axes[0], pipe_v1, y_v1, "Confusion Matrix — V1"),
    (axes[1], pipe_v2, y_v2, "Confusion Matrix — V2"),
]:
    cm = confusion_matrix(y, pipe.predict(prep_text(v1 if title.endswith("V1") else v2)), labels=CLASSES)
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("confusion_matrices_compare.png", dpi=150, bbox_inches="tight")
print("Saved: confusion_matrices_compare.png")

print("\nDone.")

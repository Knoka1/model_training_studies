"""
Run all model combinations on the scope-only dataset.

Outputs go to outputs/hg_labeled_scope_only_final/ instead of the default outputs/.

Train/test split: stratified 80/20 hold-out (random_state=42).
Cross-validation: 5-fold on the TRAINING set only.

Models:
  tfidf    — TF-IDF (1–2 grams) + Logistic Regression
  semantic — Sentence-Transformer embeddings (all-MiniLM-L6-v2) + Logistic Regression
"""

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.config import CLASSES, SEMANTIC_MODEL, TEXT_COLS, LABEL_COL, TEST_SIZE, RANDOM_STATE, CV_FOLDS, N_TOP_KEYWORDS
from src.utils  import (
    load, split,
    evaluate_tfidf, evaluate_semantic,
    keywords_tfidf, keywords_semantic,
    plot_keywords, plot_confusion, plot_comparison,
)

ROOT = Path(__file__).parent

DATASET_PATH = ROOT / "datatsets/keywords_hg_labeled_scope_only.csv"

OUT_BASE     = ROOT / "outputs/hg_labeled_scope_only_final"
OUT_METRICS  = OUT_BASE / "metrics"
OUT_PLOTS    = OUT_BASE / "plots"
OUT_KEYWORDS = OUT_BASE / "keywords"

for p in [OUT_METRICS, OUT_PLOTS, OUT_KEYWORDS]:
    p.mkdir(parents=True, exist_ok=True)

DATASETS = {"scope_only": DATASET_PATH}

# ── Load sentence-transformer once ───────────────────────────────────────────
print("Loading sentence-transformer model...", flush=True)
st_model = SentenceTransformer(SEMANTIC_MODEL)

# ── Run all combinations ──────────────────────────────────────────────────────
summary_rows = []
confusion_results = {}

for ds_name, ds_path in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"DATASET: {ds_name}  ({ds_path.name})")
    print(f"{'='*60}")

    df = load(ds_path)
    train_idx, test_idx = split(df)

    print(f"  Total: {len(df)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"  Train label dist: {df[df.index.isin(train_idx)]['Label'].value_counts().to_dict()}")
    print(f"  Test  label dist: {df[df.index.isin(test_idx)]['Label'].value_counts().to_dict()}")

    # ── TF-IDF model ──────────────────────────────────────────────────────────
    print(f"\n  [tfidf] training...", flush=True)
    res_tfidf = evaluate_tfidf(df, train_idx, test_idx)
    tag_tfidf = f"{ds_name}_tfidf"

    print(f"  [tfidf] CV macro-F1: {res_tfidf['cv_f1']:.3f} ± {res_tfidf['cv_std']:.3f}")
    print(f"  [tfidf] Test macro-F1: {res_tfidf['test_f1']:.3f}")

    kw_tfidf = keywords_tfidf(res_tfidf)
    plot_keywords(kw_tfidf, tag_tfidf, OUT_PLOTS, model_type="tfidf")

    for cls in CLASSES:
        kw_tfidf[cls]["top"].assign(direction="positive", dataset=ds_name, model="tfidf", label=cls).to_csv(
            OUT_KEYWORDS / f"{tag_tfidf}_{cls.lower()}.csv", index=False
        )

    confusion_results[tag_tfidf] = res_tfidf
    summary_rows.append({
        "dataset": ds_name, "model": "tfidf",
        "cv_f1": res_tfidf["cv_f1"], "cv_std": res_tfidf["cv_std"],
        "test_f1": res_tfidf["test_f1"],
        **{f"{cls}_f1": res_tfidf["report"].get(cls, {}).get("f1-score", 0) for cls in CLASSES},
    })

    report_df = pd.DataFrame(res_tfidf["report"]).T
    report_df.to_csv(OUT_METRICS / f"{tag_tfidf}_report.csv")

    # ── Semantic model ────────────────────────────────────────────────────────
    print(f"\n  [semantic] training...", flush=True)
    res_sem = evaluate_semantic(df, train_idx, test_idx, st_model)
    tag_sem = f"{ds_name}_semantic"

    print(f"  [semantic] CV macro-F1: {res_sem['cv_f1']:.3f} ± {res_sem['cv_std']:.3f}")
    print(f"  [semantic] Test macro-F1: {res_sem['test_f1']:.3f}")

    df_train = df.iloc[train_idx].reset_index(drop=True)
    kw_sem = keywords_semantic(res_sem, df_train, st_model)
    plot_keywords(kw_sem, tag_sem, OUT_PLOTS, model_type="semantic")

    for cls in CLASSES:
        kw_sem[cls]["top"].assign(direction="positive", dataset=ds_name, model="semantic", label=cls).to_csv(
            OUT_KEYWORDS / f"{tag_sem}_{cls.lower()}.csv", index=False
        )

    confusion_results[tag_sem] = res_sem
    summary_rows.append({
        "dataset": ds_name, "model": "semantic",
        "cv_f1": res_sem["cv_f1"], "cv_std": res_sem["cv_std"],
        "test_f1": res_sem["test_f1"],
        **{f"{cls}_f1": res_sem["report"].get(cls, {}).get("f1-score", 0) for cls in CLASSES},
    })

    report_df = pd.DataFrame(res_sem["report"]).T
    report_df.to_csv(OUT_METRICS / f"{tag_sem}_report.csv")

# ── Summary table ─────────────────────────────────────────────────────────────
summary = pd.DataFrame(summary_rows)
summary = summary.sort_values(["dataset", "model"]).reset_index(drop=True)
summary.to_csv(OUT_METRICS / "summary.csv", index=False)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
cols = ["dataset", "model", "cv_f1", "cv_std", "test_f1"] + [f"{c}_f1" for c in CLASSES]
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── Comparison plot ───────────────────────────────────────────────────────────
plot_comparison(summary, OUT_PLOTS / "comparison.png")
print(f"\nSaved: {OUT_BASE}/plots/comparison.png")

# ── Confusion matrices ────────────────────────────────────────────────────────
for tag, res in confusion_results.items():
    plot_confusion({tag: res}, OUT_PLOTS / f"confusion_{tag}.png")
print("Saved: confusion matrices")

print("\nAll done.")

"""
Shared helpers: data loading, model building, evaluation, keyword extraction, plotting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline

from src.config import TEXT_COLS, LABEL_COL, CLASSES, RANDOM_STATE, TEST_SIZE, CV_FOLDS, N_TOP_KEYWORDS


# ── Data ──────────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    df["text"] = df[TEXT_COLS].fillna("").apply(lambda r: " ".join(r.values), axis=1).str.strip()
    return df


def split(df):
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=df[LABEL_COL].values, random_state=RANDOM_STATE
    )
    return train_idx, test_idx


# ── Model builders ────────────────────────────────────────────────────────────

def tfidf_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.9,
            sublinear_tf=True, stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def logreg():
    return LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=RANDOM_STATE,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_tfidf(df, train_idx, test_idx):
    X, y = df["text"], df[LABEL_COL]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test,  y_test  = X.iloc[test_idx],  y.iloc[test_idx]

    pipe = tfidf_pipeline()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "pipeline":  pipe,
        "cv_f1":     cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "test_f1":   test_f1,
        "report":    classification_report(y_test, y_pred, output_dict=True),
        "y_test":    y_test,
        "y_pred":    y_pred,
        "X_train":   X_train,
        "y_train":   y_train,
    }


def evaluate_semantic(df, train_idx, test_idx, st_model):
    """Pre-embed all texts (embedder is frozen), run LR + CV on embeddings."""
    y = df[LABEL_COL].values
    print("    Encoding texts...", flush=True)
    X_emb = st_model.encode(df["text"].tolist(), show_progress_bar=False, batch_size=64)

    X_train, y_train = X_emb[train_idx], y[train_idx]
    X_test,  y_test  = X_emb[test_idx],  y[test_idx]

    clf = logreg()
    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    return {
        "clf":      clf,
        "X_emb":    X_emb,
        "cv_f1":    cv_scores.mean(),
        "cv_std":   cv_scores.std(),
        "test_f1":  test_f1,
        "report":   classification_report(y_test, y_pred, output_dict=True),
        "y_test":   pd.Series(y_test),
        "y_pred":   y_pred,
        "X_train":  X_train,
        "y_train":  y_train,
    }


# ── Keyword extraction ────────────────────────────────────────────────────────

def keywords_tfidf(result, n=N_TOP_KEYWORDS):
    pipe  = result["pipeline"]
    feat  = np.array(pipe.named_steps["tfidf"].get_feature_names_out())
    clf   = pipe.named_steps["clf"]
    coef  = {cls: clf.coef_[i] for i, cls in enumerate(clf.classes_)}
    out = {}
    for cls in CLASSES:
        c = coef[cls]
        out[cls] = {
            "top": pd.DataFrame({"keyword": feat[np.argsort(c)[::-1][:n]], "coefficient": np.sort(c)[::-1][:n]}),
            "bot": pd.DataFrame({"keyword": feat[np.argsort(c)[:n]],       "coefficient": np.sort(c)[:n]}),
        }
    return out


def keywords_semantic(result, df_train, st_model, n=N_TOP_KEYWORDS):
    """
    Project TF-IDF vocabulary embeddings onto the LR weight vector for each class.
    Gives interpretable 'which words are most semantically pursue-like'.
    """
    clf = result["clf"]

    # Build vocabulary from train texts
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, stop_words="english")
    vec.fit(df_train["text"])
    vocab = list(vec.get_feature_names_out())

    print(f"    Embedding {len(vocab)} vocabulary terms...", flush=True)
    vocab_emb = st_model.encode(vocab, show_progress_bar=False, batch_size=256)  # (V, 384)

    out = {}
    for i, cls in enumerate(clf.classes_):
        scores = vocab_emb @ clf.coef_[i]                 # dot product → scalar per term
        top_idx = np.argsort(scores)[::-1][:n]
        bot_idx = np.argsort(scores)[:n]
        out[cls] = {
            "top": pd.DataFrame({"keyword": np.array(vocab)[top_idx], "score": scores[top_idx]}),
            "bot": pd.DataFrame({"keyword": np.array(vocab)[bot_idx], "score": scores[bot_idx]}),
        }
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_keywords(kw_dict, tag, out_dir, model_type="tfidf"):
    score_col = "coefficient" if model_type == "tfidf" else "score"
    for cls in CLASSES:
        top = kw_dict[cls]["top"]
        bot = kw_dict[cls]["bot"]
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Keywords → {cls}  [{tag}]", fontsize=13)

        axes[0].barh(top["keyword"][::-1], top[score_col][::-1], color="#2ecc71")
        axes[0].set_title(f"Top {N_TOP_KEYWORDS} → {cls}")
        axes[0].set_xlabel("Score"); axes[0].axvline(0, color="black", lw=0.8, ls="--")

        axes[1].barh(bot["keyword"][::-1], bot[score_col][::-1], color="#e74c3c")
        axes[1].set_title(f"Top {N_TOP_KEYWORDS} against {cls}")
        axes[1].set_xlabel("Score"); axes[1].axvline(0, color="black", lw=0.8, ls="--")

        plt.tight_layout()
        fname = out_dir / f"keywords_{tag}_{cls.lower()}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()


def plot_confusion(results_map, out_path):
    """results_map: {tag: result_dict} where result_dict has y_test, y_pred"""
    n = len(results_map)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (tag, res) in zip(axes, results_map.items()):
        cm = confusion_matrix(res["y_test"], res["y_pred"], labels=CLASSES)
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(tag)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(summary_df, out_path):
    datasets = summary_df["dataset"].unique()
    models   = summary_df["model"].unique()
    x = np.arange(len(datasets))
    w = 0.35
    colors = {"tfidf": "#3498db", "semantic": "#e67e22"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in [
        (axes[0], "cv_f1",   "5-Fold CV Macro-F1 (train set)"),
        (axes[1], "test_f1", "Test Macro-F1 (held-out 20%)"),
    ]:
        for i, model in enumerate(models):
            sub = summary_df[summary_df["model"] == model].set_index("dataset")
            vals = [sub.loc[d, metric] if d in sub.index else 0 for d in datasets]
            errs = [sub.loc[d, "cv_std"] if d in sub.index and metric == "cv_f1" else 0 for d in datasets]
            offset = (i - 0.5) * w
            bars = ax.bar(x + offset, vals, w, yerr=errs if metric == "cv_f1" else None,
                          capsize=5, label=model, color=colors[model])
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1")
        ax.set_title(title); ax.legend()
        ax.axhline(0.65, color="gray", lw=0.8, ls="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

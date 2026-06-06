from pathlib import Path

ROOT = Path(__file__).parent.parent

DATASETS = {
    "v1":      ROOT / "datatsets/keywords_hg_labeled.csv",
    "v2":      ROOT / "datatsets/keywords_hg_labeled_v2.csv",
    "v2_mega": ROOT / "datatsets/keyword_hg_labeled_v2_mega.csv",
}

TEXT_COLS       = ["Title", "Description", "Documents AI Summary"]
LABEL_COL       = "Label"
CLASSES         = ["DROP", "PURSUE", "REVIEW"]
SEMANTIC_MODEL  = "all-MiniLM-L6-v2"
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
CV_FOLDS        = 5
N_TOP_KEYWORDS  = 30

OUT_METRICS  = ROOT / "outputs/metrics"
OUT_PLOTS    = ROOT / "outputs/plots"
OUT_KEYWORDS = ROOT / "outputs/keywords"

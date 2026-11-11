from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / (
    "AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. "
    "httpsdoi.org10.1093databasebaae120"
) / "AneRBC_dataset"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

LABELS = {"healthy": 0, "anaemic": 1}
LABEL_ALIASES = {
    "Healthy_individuals": ("healthy", "h"),
    "Anemic_individuals": ("anaemic", "a"),
}

CBC_FEATURES = [
    "WBC", "RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW_CV",
    "PLT", "MPV", "Neut", "LYMP", "MONO", "EOS", "BASO",
    "NEUT_abs", "LYMP_abs", "MONO_abs", "EOS_abs", "BASO_abs",
]

FEATURE_SETS = {
    "core_with_hgb": ["RBC", "HGB", "HCT", "MCV", "MCH", "MCHC", "RDW_CV"],
    "core_without_hgb": ["RBC", "HCT", "MCV", "MCH", "MCHC", "RDW_CV"],
    "full_with_hgb": CBC_FEATURES,
    "full_without_hgb": [f for f in CBC_FEATURES if f != "HGB"],
}

MANIFEST_COLUMNS = [
    "case_id", "label", "label_id", "original_image_path",
    *[f"patch_{i}_path" for i in range(1, 13)],
    "cbc_report_path", "morphology_report_path", "split",
]

DEFAULT_SEED = 42

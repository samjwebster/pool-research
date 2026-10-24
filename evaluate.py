"""Evaluate pooling methods on anonymized DINOv3 patch features."""

import argparse
import csv
import inspect
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from discord_webhook import send_evaluation
from pooling_methods import POOLING_METHODS

ROOT = Path(__file__).resolve().parent
DEFAULT_FEATURE_DIR = ROOT / "data" / "features_unpooled"
DATASETS = {
    "Dataset_1": "FingerprintPAD_dinov3_features.h5",
    "Dataset_2": "FingerVeinPAD_dinov3_features.h5",
    "Dataset_3": "IDCardPAD_dinov3_features.h5",
    "Dataset_4": "IrisPAD_dinov3_features.h5",
    "Dataset_5": "SyntheticFace_dinov3_features.h5",
}
BASELINE_NAMES = ("GAP", "GMP", "GeM", "LSE", "LAE", "NWE", "GAP_GMP")
RESULTS_PATH = ROOT / "results.json"
PROGRESS_PATH = ROOT / "progress.csv"


def load_unpooled(path):
    with h5py.File(path, "r") as handle:
        features = np.asarray(handle["features"][:], dtype=np.float32)
        labels = handle["class_label"][:]
    labels = np.asarray(
        [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in labels]
    )
    if features.ndim == 4 and features.shape[1] == 1:
        features = features[:, 0]
    if features.ndim != 3:
        raise ValueError(f"{path} must contain [images, patches, dim], got {features.shape}")
    return features, labels


def evaluate_dataset(method, features, labels, seed=42):
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    train_features, test_features, y_train, y_test = train_test_split(
        features, y, test_size=0.2, random_state=seed, stratify=y
    )
    method.fit(torch.from_numpy(train_features))
    with torch.inference_mode():
        x_train = method.pool(torch.from_numpy(train_features)).cpu().numpy()
        x_test = method.pool(torch.from_numpy(test_features)).cpu().numpy()
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(x_train, y_train)
    return float(probe.score(x_test, y_test))


def _method_description(method_class):
    return inspect.getdoc(method_class) or getattr(method_class, "description", "")


def _read_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {"best_mean_accuracy": None, "results": []}


def _write_result(result):
    state = _read_results()
    state["results"].append(result)
    state["best_mean_accuracy"] = max(
        (item["mean_accuracy"] for item in state["results"]), default=None
    )
    RESULTS_PATH.write_text(json.dumps(state, indent=2) + "\n")
    exists = PROGRESS_PATH.exists()
    with PROGRESS_PATH.open("a", newline="") as handle:
        fields = ["method_name", "method_path", *DATASETS, "mean_accuracy", "all_time_best"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: result.get(field, "") for field in fields})


def run_method(name, feature_dir, prior_best=None):
    if name not in POOLING_METHODS:
        raise ValueError(f"Unknown method {name!r}; available: {sorted(POOLING_METHODS)}")
    method_class = POOLING_METHODS[name]
    scores = {}
    for dataset_name, filename in DATASETS.items():
        path = feature_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {dataset_name} feature file: {path}")
        features, labels = load_unpooled(path)
        # Fit state is dataset-specific; never allow one anonymized dataset to
        # influence another.
        scores[dataset_name] = evaluate_dataset(method_class(), features, labels)
    mean_accuracy = float(np.mean(list(scores.values())))
    all_time_best = prior_best is None or mean_accuracy > prior_best
    result = {
        "method_name": name,
        "method_path": str(Path(inspect.getfile(method_class)).relative_to(ROOT)),
        "description": _method_description(method_class),
        **scores,
        "mean_accuracy": mean_accuracy,
        "all_time_best": all_time_best,
    }
    _write_result(result)
    send_evaluation(result, baseline_or_best=prior_best)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Registered method to evaluate; omit to warm up baselines.")
    parser.add_argument("--feature-dir", default=os.environ.get("UNPOOLED_FEATURE_DIR", str(DEFAULT_FEATURE_DIR)))
    parser.add_argument("--force-baselines", action="store_true")
    args = parser.parse_args()
    feature_dir = Path(args.feature_dir).resolve()
    state = _read_results()
    names = BASELINE_NAMES if args.force_baselines or args.method is None else (args.method,)
    prior_best = state["best_mean_accuracy"]
    for name in names:
        result = run_method(name, feature_dir, prior_best)
        prior_best = max(prior_best or result["mean_accuracy"], result["mean_accuracy"])
        print(json.dumps(result))


if __name__ == "__main__":
    main()

"""Validate the fixed VFM feature inputs before starting autonomous search.

The large feature files are prepared externally. This command intentionally
does not download data or expose source dataset names to the search agent.
"""

from pathlib import Path

from evaluate import DATASETS, DEFAULT_FEATURE_DIR, load_unpooled


def main():
    feature_dir = Path(DEFAULT_FEATURE_DIR)
    for dataset_name, filename in DATASETS.items():
        path = feature_dir / filename
        features, labels = load_unpooled(path)
        print(f"{dataset_name}: {features.shape[0]} samples, {features.shape[1]} patches, {features.shape[2]} dims, {len(set(labels))} classes")
    print("All anonymized VFM features are ready.")


if __name__ == "__main__":
    main()

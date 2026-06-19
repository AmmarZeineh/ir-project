from pathlib import Path


def raw_dataset_dir(dataset: str) -> Path:
    return Path(f"data/raw/{dataset}")


def processed_dataset_dir(dataset: str) -> Path:
    return Path(f"data/processed/{dataset}")


def index_dataset_dir(dataset: str) -> Path:
    return Path(f"data/indexes/{dataset}")


def raw_docs_path(dataset: str) -> Path:
    raw_dir = raw_dataset_dir(dataset)
    docs_path = raw_dir / "docs.csv"
    if docs_path.exists():
        return docs_path
    return raw_dir / "docs_relevant.csv"

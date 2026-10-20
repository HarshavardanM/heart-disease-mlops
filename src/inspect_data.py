
from pathlib import Path

import pandas as pd


# Project root is the parent directory of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

DATASETS = {
    "Cleveland": "processed.cleveland.data",
    "Hungarian": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a UCI Heart Disease data file and normalize missing values."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )

    # Convert every column to numeric; invalid entries become missing.
    for column in COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    target_rows = []

    for dataset_name, filename in DATASETS.items():
        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        df = load_dataset(path)

        # Missingness and data-quality checks
        missing_by_column = df.isna().sum()
        missing_cells = int(missing_by_column.sum())

        # The target should be present for supervised learning.
        missing_targets = int(df["num"].isna().sum())

        # Exact duplicate rows within each source file
        duplicate_rows = int(df.duplicated().sum())

        audit_rows.append(
            {
                "dataset": dataset_name,
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_cells": missing_cells,
                "missing_target": missing_targets,
                "duplicate_rows": duplicate_rows,
                "missing_percentage": round(
                    100 * missing_cells / df.size, 2
                ),
                "ca_missing": int(df["ca"].isna().sum()),
                "thal_missing": int(df["thal"].isna().sum()),
                "slope_missing": int(df["slope"].isna().sum()),
            }
        )

        # Record original target counts and binary target counts.
        original_counts = df["num"].value_counts(
            dropna=False
        ).sort_index()

        binary_target = df["num"].map(
            lambda value: (
                int(value > 0) if pd.notna(value) else pd.NA
            )
        )

        binary_counts = binary_target.value_counts(
            dropna=False
        ).sort_index()

        for target_value, count in original_counts.items():
            target_rows.append(
                {
                    "dataset": dataset_name,
                    "target_type": "original_num",
                    "target_value": str(target_value),
                    "count": int(count),
                }
            )

        for target_value, count in binary_counts.items():
            target_rows.append(
                {
                    "dataset": dataset_name,
                    "target_type": "binary",
                    "target_value": str(target_value),
                    "count": int(count),
                }
            )

        print(f"\n--- {dataset_name} ---")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print(f"Missing cells: {missing_cells}")
        print(f"Duplicate rows: {duplicate_rows}")
        print("\nMissing values by column:")
        print(missing_by_column.to_string())
        print("\nOriginal target distribution:")
        print(original_counts.to_string())
        print("\nBinary target distribution (0 vs 1):")
        print(binary_counts.to_string())

    audit_df = pd.DataFrame(audit_rows)
    targets_df = pd.DataFrame(target_rows)

    audit_path = REPORT_DIR / "dataset_audit.csv"
    target_path = REPORT_DIR / "target_distribution.csv"

    audit_df.to_csv(audit_path, index=False)
    targets_df.to_csv(target_path, index=False)

    print("\n=== Dataset comparison ===")
    print(audit_df.to_string(index=False))

    print("\nAudit reports saved:")
    print(audit_path)
    print(target_path)


if __name__ == "__main__":
    main()
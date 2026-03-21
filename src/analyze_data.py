from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_preprocessing import DataPreprocessor


def analyze_dataset():
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "PPMI_Curated_Data_Cut_Public_20250714.csv"
    notebooks_dir = root_dir / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print("\nDataset Shape:", df.shape)

    if "COHORT" not in df.columns:
        raise ValueError("Dataset is missing required column: COHORT")

    print("\nCOHORT Distribution:")
    cohort_dist = df["COHORT"].value_counts(dropna=False)
    print(cohort_dist)

    preprocessor = DataPreprocessor()
    selected_features = list(preprocessor.core.selected_features)

    analysis_features = [
        col
        for col in selected_features
        if col in df.columns and col not in {"COHORT", "PATNO"}
    ]

    print(f"\nUsing {len(analysis_features)} available analysis features:")
    print(analysis_features)

    if not analysis_features:
        raise ValueError("No expected analysis features were found in the dataset.")

    print("\nKey Features Statistics:")
    print(df[analysis_features].describe(include="all").transpose())

    print("\nMissing Values Analysis:")
    missing = df[analysis_features].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_summary = pd.DataFrame(
        {
            "Missing Values": missing,
            "Percentage": missing_pct,
        }
    ).sort_values("Percentage", ascending=False)
    print(missing_summary[missing_summary["Missing Values"] > 0])

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="COHORT")
    plt.title("Distribution of Cohorts")
    plt.tight_layout()
    plt.savefig(notebooks_dir / "cohort_distribution.png")
    plt.close()

    numerical_features = (
        df[analysis_features].select_dtypes(include=["number"]).columns.tolist()
    )

    if numerical_features:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numerical_features].corr(numeric_only=True)
        sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(notebooks_dir / "correlation_matrix.png")
        plt.close()

        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 5 * n_rows))
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(data=df, x=feature, kde=True)
            plt.title(feature)
        plt.tight_layout()
        plt.savefig(notebooks_dir / "feature_distributions.png")
        plt.close()
    else:
        print("\nNo numerical features available for correlation/distribution plots.")

    print("\nFeature Types:")
    for feature in analysis_features:
        dtype = df[feature].dtype
        n_unique = df[feature].nunique(dropna=True)
        print(f"- {feature}: {dtype}, {n_unique} unique values")

    missing_expected = [
        col
        for col in selected_features
        if col not in df.columns and col not in {"COHORT", "PATNO"}
    ]
    if missing_expected:
        print("\nExpected features not found in dataset:")
        for feature in missing_expected:
            print(f"- {feature}")


if __name__ == "__main__":
    analyze_dataset()

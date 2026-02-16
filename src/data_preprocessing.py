import os

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class PPMIDataPreprocessor:
    """
    Clean, leak-proof preprocessing pipeline for PPMI dataset.
    Handles patient-level splitting, imputation, encoding, scaling.
    """

    def __init__(self):

        # -------------------------
        # Selected Features
        # -------------------------
        self.selected_features = [
            # Demographics
            "age", "SEX", "EDUCYRS", "race", "BMI",

            # Family History
            "fampd", "fampd_bin",

            # Motor symptoms
            "sym_tremor", "sym_rigid", "sym_brady", "sym_posins",

            # Non-motor
            "rem", "ess", "gds", "stai",

            # Cognitive
            "moca", "clockdraw", "bjlot",

            # Target
            "COHORT",

            # Patient ID
            "PATNO"
        ]

        # Features for KNN
        self.cognitive_cols = ["moca", "clockdraw", "bjlot", "rem", "gds", "ess", "stai"]

        # Categorical
        self.categorical_cols = ["SEX", "race", "fampd"]

        # Numeric
        self.numeric_cols = [
            "age", "EDUCYRS", "BMI",
            "fampd_bin",
            "sym_tremor", "sym_rigid", "sym_brady", "sym_posins",
        ]

        self.preprocessor = None

    # ------------------------------------------------------------
    def load_data(self, file_path):
        """Load and filter only necessary columns."""
        df = pd.read_csv(file_path)
        df = df[self.selected_features].dropna(subset=["COHORT", "PATNO"])
        return df

    # ------------------------------------------------------------
    def patient_split(self, df, test_size=0.2):
        """Split such that the same patient never appears in both sets."""
        gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
        train_idx, test_idx = next(gss.split(df, groups=df["PATNO"]))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        return train_df, test_df

    # ------------------------------------------------------------
    def build_preprocessor(self):
        """Builds leak-proof transformers."""

        numeric_transformer = Pipeline(steps=[
            ("imputer_mean", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        knn_transformer = Pipeline(steps=[
            ("imputer_knn", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer_mode", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer,
                 [c for c in self.numeric_cols if c not in self.cognitive_cols]),

                ("knn", knn_transformer, self.cognitive_cols),

                ("cat", categorical_transformer, self.categorical_cols),
            ],
            remainder="drop"
        )

        return self.preprocessor

    # ------------------------------------------------------------
    def prepare(self, file_path):
        """
        COMPLETE DATA PREPARATION PIPELINE
        Returns X_train, X_test, y_train, y_test
        """

        df = self.load_data(file_path)
        train_df, test_df = self.patient_split(df)

        X_train = train_df.drop(["COHORT", "PATNO"], axis=1)
        y_train = train_df["COHORT"]

        X_test = test_df.drop(["COHORT", "PATNO"], axis=1)
        y_test = test_df["COHORT"]

        # Build processor & fit only on training data
        pre = self.build_preprocessor()
        X_train_processed = pre.fit_transform(X_train)
        X_test_processed = pre.transform(X_test)

        return X_train_processed, X_test_processed, y_train.values, y_test.values


class DataPreprocessor:
    """Backwards-compatible wrapper around the new PPMI pipeline."""

    def __init__(self):
        self.core = PPMIDataPreprocessor()
        self.feature_names_ = None
        self.class_mapping_ = None
        self.preprocessor_ = None
        self.train_df_ = None
        self.test_df_ = None

    def _load_all_files(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        dataframes = []
        for path in file_paths:
            if not path:
                continue
            if not os.path.exists(path):
                print(f"[WARN] DataPreprocessor: '{path}' not found, skipping.")
                continue
            df = self.core.load_data(path)
            dataframes.append(df)

        if not dataframes:
            raise FileNotFoundError("No valid CSV files were provided to DataPreprocessor.")

        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        return combined

    def prepare_data(self, file_paths, test_size=0.2, use_patient_split=True):
        """Expose the legacy API expected by the training scripts."""

        df = self._load_all_files(file_paths)

        if use_patient_split:
            train_df, test_df = self.core.patient_split(df, test_size=test_size)
        else:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=42,
                stratify=df["COHORT"],
            )

        self.train_df_ = train_df.reset_index(drop=True)
        self.test_df_ = test_df.reset_index(drop=True)

        classes_sorted = np.sort(df["COHORT"].unique())
        self.class_mapping_ = {original: idx for idx, original in enumerate(classes_sorted)}

        X_train = train_df.drop(["COHORT", "PATNO"], axis=1)
        y_train = train_df["COHORT"].map(self.class_mapping_).values
        X_test = test_df.drop(["COHORT", "PATNO"], axis=1)
        y_test = test_df["COHORT"].map(self.class_mapping_).values

        pre = self.core.build_preprocessor()
        X_train_processed = pre.fit_transform(X_train)
        X_test_processed = pre.transform(X_test)
        self.preprocessor_ = pre

        try:
            self.feature_names_ = pre.get_feature_names_out(X_train.columns).tolist()
        except AttributeError:
            self.feature_names_ = None

        return X_train_processed, X_test_processed, y_train, y_test

    def get_feature_names(self):
        if self.feature_names_ is None:
            raise ValueError("Feature names are unavailable. Call prepare_data() first.")
        return self.feature_names_

    def get_preprocessor(self):
        if self.preprocessor_ is None:
            raise ValueError("Preprocessor is unavailable. Call prepare_data() first.")
        return self.preprocessor_

    def get_class_mapping(self):
        if self.class_mapping_ is None:
            raise ValueError("Class mapping is unavailable. Call prepare_data() first.")
        return self.class_mapping_

    def get_split_frames(self):
        if self.train_df_ is None or self.test_df_ is None:
            raise ValueError("Split frames are unavailable. Call prepare_data() first.")
        return self.train_df_.copy(), self.test_df_.copy()


# -------------------------------------------------------------------
# Script example (not executed when imported)
# -------------------------------------------------------------------
if __name__ == "__main__":
    file_path = "PPMI_Curated_Data.csv"
    prep = PPMIDataPreprocessor()
    X_train, X_test, y_train, y_test = prep.prepare(file_path)

    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

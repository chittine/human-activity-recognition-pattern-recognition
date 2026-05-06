
# ============================================================
# CSE 802 FINAL PROJECT CODE
# Pattern Classification of Human Activities Using Sensor Data
#
# Datasets:
#   1. HAR Smartphone Dataset (train.csv, test.csv)
#   2. WISDM Dataset (raw sensor .txt files)
#
# This code includes:
#   - Min-max normalization
#   - Z-score normalization
#   - PCA
#   - SFFS / Sequential Forward Feature Selection
#   - kNN, SVM, Random Forest
#   - Bayesian MAP classifier with multivariate Gaussian density
#   - Gaussian Naive Bayes
#   - Non-parametric KDE / Parzen classifier
#   - Confusion matrices
#   - Accuracy
#   - Class confusion analysis
#   - Stability analysis using repeated splits
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")


# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
N_STABILITY_RUNS = 10

# For speed. You can increase these if your machine can handle it.
PCA_COMPONENTS = 20
SFFS_FEATURES = 10

# KDE can be slow in high dimensions, so use reduced features for KDE.
KDE_MAX_FEATURES = 10
KDE_BANDWIDTH = 1.0


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def split_train_val_test(X, y, test_size=0.20, validation_size=0.20, random_state=42):
    """
    Creates train, validation, and test partitions.
    First split: train+validation vs test.
    Second split: train vs validation.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    val_relative_size = validation_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_scaler(normalization_name):
    """
    Returns the scaler object for the selected normalization method.
    """
    if normalization_name == "zscore":
        return StandardScaler()
    elif normalization_name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError("normalization_name must be either 'zscore' or 'minmax'")


def apply_normalization(X_train, X_val, X_test, normalization_name):
    """
    Fits scaler only on training data and transforms validation and test.
    """
    scaler = get_scaler(normalization_name)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def apply_pca(X_train, X_val, X_test, n_components=20):
    """
    Applies PCA feature extraction.
    PCA is fitted only on training data.
    """
    n_components = min(n_components, X_train.shape[1], X_train.shape[0] - 1)

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    explained = np.sum(pca.explained_variance_ratio_)

    return X_train_pca, X_val_pca, X_test_pca, explained


def apply_sffs(X_train, X_val, X_test, y_train, n_features=10):
    """
    Sequential Forward Feature Selection using kNN as the selection estimator.
    This identifies informative original features.
    """
    n_features = min(n_features, X_train.shape[1])

    estimator = KNeighborsClassifier(n_neighbors=5)

    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features,
        direction="forward",
        scoring="accuracy",
        cv=3,
        n_jobs=-1
    )

    selector.fit(X_train, y_train)

    X_train_sffs = selector.transform(X_train)
    X_val_sffs = selector.transform(X_val)
    X_test_sffs = selector.transform(X_test)

    selected_indices = np.where(selector.get_support())[0]

    return X_train_sffs, X_val_sffs, X_test_sffs, selected_indices


def analyze_confusions(cm, labels, top_k=5):
    """
    Prints the most common class confusions from a confusion matrix.
    """
    confusions = []

    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((labels[i], labels[j], cm[i, j]))

    confusions = sorted(confusions, key=lambda x: x[2], reverse=True)

    if len(confusions) == 0:
        print("No major class confusions.")
        return

    print("\nMost frequent class confusions:")
    for true_label, pred_label, count in confusions[:top_k]:
        print(f"  True: {true_label}  -->  Predicted: {pred_label}  Count: {count}")


def print_evaluation(name, y_test, y_pred, labels):
    """
    Prints accuracy, confusion matrix, classification report, and confusion analysis.
    """
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    analyze_confusions(cm, labels)

    return acc, cm


# ============================================================
# 4. BAYESIAN CLASSIFIERS
# ============================================================

class MultivariateGaussianMAP:
    """
    Bayesian MAP classifier using multivariate Gaussian class-conditional densities.

    Decision rule:
        choose class w_i that maximizes:
        p(x | w_i) P(w_i)

    Each class density is modeled as:
        p(x | w_i) = N(mu_i, Sigma_i)

    A small regularization term is added to the covariance matrix to avoid singularity.
    """

    def __init__(self, reg=1e-4):
        self.reg = reg
        self.classes_ = None
        self.means_ = {}
        self.covs_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for c in self.classes_:
            X_c = X[y == c]

            self.means_[c] = np.mean(X_c, axis=0)

            cov = np.cov(X_c, rowvar=False)

            if cov.ndim == 0:
                cov = np.array([[cov]])

            cov = cov + self.reg * np.eye(cov.shape[0])
            self.covs_[c] = cov

            self.priors_[c] = X_c.shape[0] / X.shape[0]

        return self

    def _log_gaussian_density(self, X, mean, cov):
        X = np.asarray(X)
        d = X.shape[1]

        sign, logdet = np.linalg.slogdet(cov)

        if sign <= 0:
            cov = cov + self.reg * np.eye(cov.shape[0])
            sign, logdet = np.linalg.slogdet(cov)

        inv_cov = np.linalg.pinv(cov)

        diff = X - mean
        mahalanobis = np.sum(diff @ inv_cov * diff, axis=1)

        log_density = -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)

        return log_density

    def predict(self, X):
        X = np.asarray(X)

        scores = []

        for c in self.classes_:
            log_likelihood = self._log_gaussian_density(
                X,
                self.means_[c],
                self.covs_[c]
            )
            log_prior = np.log(self.priors_[c])
            scores.append(log_likelihood + log_prior)

        scores = np.vstack(scores).T
        best_indices = np.argmax(scores, axis=1)

        return self.classes_[best_indices]


class KDEParzenMAP:
    """
    Non-parametric Bayesian classifier using KDE / Parzen density estimation.

    For each class:
        p(x | w_i) is estimated using KernelDensity.

    MAP rule:
        choose class w_i that maximizes:
        log p(x | w_i) + log P(w_i)
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.classes_ = None
        self.kdes_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        for c in self.classes_:
            X_c = X[y == c]

            kde = KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel
            )

            kde.fit(X_c)

            self.kdes_[c] = kde
            self.priors_[c] = X_c.shape[0] / X.shape[0]

        return self

    def predict(self, X):
        X = np.asarray(X)

        scores = []

        for c in self.classes_:
            log_density = self.kdes_[c].score_samples(X)
            log_prior = np.log(self.priors_[c])
            scores.append(log_density + log_prior)

        scores = np.vstack(scores).T
        best_indices = np.argmax(scores, axis=1)

        return self.classes_[best_indices]


# ============================================================
# 5. CLASSIFIER RUNNERS
# ============================================================

def get_standard_models():
    """
    Existing software-package classifiers required by the proposal.
    """
    return {
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE
        )
    }


def get_bayesian_models():
    """
    Bayesian classifiers required by the proposal.
    """
    return {
        "Bayesian MAP - Multivariate Gaussian": MultivariateGaussianMAP(reg=1e-4),
        "Gaussian Naive Bayes": GaussianNB(),
        "Bayesian MAP - KDE / Parzen": KDEParzenMAP(
            bandwidth=KDE_BANDWIDTH,
            kernel="gaussian"
        )
    }


def run_models_on_feature_set(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    labels,
    dataset_name,
    normalization_name,
    feature_set_name
):
    """
    Runs software classifiers and Bayesian classifiers on one feature representation.
    """
    results = []

    all_models = {}
    all_models.update(get_standard_models())
    all_models.update(get_bayesian_models())

    for model_name, model in all_models.items():

        # KDE is expensive in high dimensions. Use only first few features for KDE.
        if "KDE" in model_name:
            Xtr = X_train[:, :min(KDE_MAX_FEATURES, X_train.shape[1])]
            Xva = X_val[:, :min(KDE_MAX_FEATURES, X_val.shape[1])]
            Xte = X_test[:, :min(KDE_MAX_FEATURES, X_test.shape[1])]
        else:
            Xtr, Xva, Xte = X_train, X_val, X_test

        model.fit(Xtr, y_train)

        y_val_pred = model.predict(Xva)
        val_acc = accuracy_score(y_val, y_val_pred)

        y_test_pred = model.predict(Xte)
        test_acc, cm = print_evaluation(
            name=f"{dataset_name} | {normalization_name} | {feature_set_name} | {model_name}",
            y_test=y_test,
            y_pred=y_test_pred,
            labels=labels
        )

        results.append({
            "dataset": dataset_name,
            "normalization": normalization_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "validation_accuracy": val_acc,
            "test_accuracy": test_acc
        })

    return results


def run_full_experiment(X, y, dataset_name):
    """
    Full pipeline:
    - train/validation/test split
    - z-score and min-max normalization
    - original features
    - PCA features
    - SFFS features
    - kNN, SVM, RF
    - Bayesian classifiers
    """
    print("\n" + "#" * 90)
    print(f"RUNNING FULL EXPERIMENT FOR: {dataset_name}")
    print("#" * 90)

    labels = sorted(np.unique(y))

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X,
        y,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    all_results = []

    for normalization_name in ["zscore", "minmax"]:
        print("\n" + "-" * 90)
        print(f"Normalization: {normalization_name}")
        print("-" * 90)

        X_train_scaled, X_val_scaled, X_test_scaled = apply_normalization(
            X_train,
            X_val,
            X_test,
            normalization_name
        )

        # ----------------------------------------------------
        # A. Original normalized features
        # ----------------------------------------------------
        original_results = run_models_on_feature_set(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            y_val,
            y_test,
            labels,
            dataset_name,
            normalization_name,
            "Original Features"
        )
        all_results.extend(original_results)

        # ----------------------------------------------------
        # B. PCA features
        # ----------------------------------------------------
        X_train_pca, X_val_pca, X_test_pca, explained = apply_pca(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            n_components=PCA_COMPONENTS
        )

        print(f"\nPCA explained variance using {X_train_pca.shape[1]} components: {explained:.4f}")

        pca_results = run_models_on_feature_set(
            X_train_pca,
            X_val_pca,
            X_test_pca,
            y_train,
            y_val,
            y_test,
            labels,
            dataset_name,
            normalization_name,
            f"PCA Features ({X_train_pca.shape[1]} components)"
        )
        all_results.extend(pca_results)

        # ----------------------------------------------------
        # C. SFFS features
        # ----------------------------------------------------
        X_train_sffs, X_val_sffs, X_test_sffs, selected_indices = apply_sffs(
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train,
            n_features=SFFS_FEATURES
        )

        print(f"\nSFFS selected feature indices: {selected_indices}")

        sffs_results = run_models_on_feature_set(
            X_train_sffs,
            X_val_sffs,
            X_test_sffs,
            y_train,
            y_val,
            y_test,
            labels,
            dataset_name,
            normalization_name,
            f"SFFS Features ({len(selected_indices)} features)"
        )
        all_results.extend(sffs_results)

    results_df = pd.DataFrame(all_results)

    print("\n" + "#" * 90)
    print(f"SUMMARY RESULTS FOR: {dataset_name}")
    print("#" * 90)
    print(results_df.sort_values("test_accuracy", ascending=False).to_string(index=False))

    return results_df


# ============================================================
# 6. STABILITY ANALYSIS
# ============================================================

def stability_analysis(
    X,
    y,
    dataset_name,
    model,
    model_name,
    normalization_name="zscore",
    feature_method="original",
    n_runs=10
):
    """
    Repeats training/testing using different random partitions.
    Computes mean and variance of classification error rate.
    """
    print("\n" + "#" * 90)
    print(f"STABILITY ANALYSIS: {dataset_name} | {model_name} | {normalization_name} | {feature_method}")
    print("#" * 90)

    errors = []
    accuracies = []

    for seed in range(n_runs):
        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
            X,
            y,
            test_size=TEST_SIZE,
            validation_size=VALIDATION_SIZE,
            random_state=seed
        )

        X_train_scaled, X_val_scaled, X_test_scaled = apply_normalization(
            X_train,
            X_val,
            X_test,
            normalization_name
        )

        if feature_method == "pca":
            X_train_final, X_val_final, X_test_final, explained = apply_pca(
                X_train_scaled,
                X_val_scaled,
                X_test_scaled,
                n_components=PCA_COMPONENTS
            )
        elif feature_method == "sffs":
            X_train_final, X_val_final, X_test_final, selected_indices = apply_sffs(
                X_train_scaled,
                X_val_scaled,
                X_test_scaled,
                y_train,
                n_features=SFFS_FEATURES
            )
        else:
            X_train_final, X_val_final, X_test_final = (
                X_train_scaled,
                X_val_scaled,
                X_test_scaled
            )

        model.fit(X_train_final, y_train)
        y_pred = model.predict(X_test_final)

        acc = accuracy_score(y_test, y_pred)
        error = 1.0 - acc

        accuracies.append(acc)
        errors.append(error)

        print(f"Run {seed + 1}: Accuracy = {acc:.4f}, Error = {error:.4f}")

    mean_error = np.mean(errors)
    variance_error = np.var(errors)
    mean_accuracy = np.mean(accuracies)

    print("\nStability Summary:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Error Rate: {mean_error:.4f}")
    print(f"Variance of Error Rate: {variance_error:.6f}")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "normalization": normalization_name,
        "feature_method": feature_method,
        "mean_accuracy": mean_accuracy,
        "mean_error_rate": mean_error,
        "variance_error_rate": variance_error
    }


# ============================================================
# 7. LOAD HAR SMARTPHONE DATASET
# ============================================================

def load_har_dataset(train_path="train.csv", test_path="test.csv"):
    """
    Loads HAR dataset from Kaggle/UCI train.csv and test.csv.
    The last column should be Activity.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    df = pd.concat([train, test], axis=0).reset_index(drop=True)

    if "Activity" not in df.columns:
        raise ValueError("HAR dataset must contain an 'Activity' column.")

    X = df.drop(columns=["Activity"])

    # Remove subject column if it exists because it is an identifier, not a sensor feature.
    if "subject" in X.columns:
        X = X.drop(columns=["subject"])

    y = df["Activity"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    return X.values, y.values


# ============================================================
# 8. LOAD AND PROCESS WISDM DATASET
# ============================================================

def read_wisdm_file(file_path):
    """
    Reads one WISDM raw sensor file.

    Expected format:
        user, activity, timestamp, x, y, z;

    Some files may have semicolons at the end.
    """
    rows = []

    with open(file_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if len(line) == 0:
                continue

            line = line.replace(";", "")
            parts = line.split(",")

            if len(parts) != 6:
                continue

            try:
                user = parts[0]
                activity = parts[1]
                timestamp = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])

                rows.append([user, activity, timestamp, x, y, z])

            except:
                continue

    df = pd.DataFrame(
        rows,
        columns=["user", "activity", "timestamp", "x", "y", "z"]
    )

    return df


def extract_wisdm_window_features(df, window_size=100, step_size=100):
    """
    Extracts statistical features from WISDM sensor windows.
    Uses x, y, z axes and magnitude.
    """
    feature_rows = []

    df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

    for user in df["user"].unique():
        user_df = df[df["user"] == user]

        for activity in user_df["activity"].unique():
            act_df = user_df[user_df["activity"] == activity].reset_index(drop=True)

            for start in range(0, len(act_df) - window_size + 1, step_size):
                window = act_df.iloc[start:start + window_size]

                if len(window) < window_size:
                    continue

                x = window["x"].values
                y = window["y"].values
                z = window["z"].values
                mag = np.sqrt(x**2 + y**2 + z**2)

                row = {
                    "user": user,
                    "activity": activity,

                    "x_mean": np.mean(x),
                    "y_mean": np.mean(y),
                    "z_mean": np.mean(z),
                    "mag_mean": np.mean(mag),

                    "x_std": np.std(x),
                    "y_std": np.std(y),
                    "z_std": np.std(z),
                    "mag_std": np.std(mag),

                    "x_min": np.min(x),
                    "y_min": np.min(y),
                    "z_min": np.min(z),
                    "mag_min": np.min(mag),

                    "x_max": np.max(x),
                    "y_max": np.max(y),
                    "z_max": np.max(z),
                    "mag_max": np.max(mag),

                    "x_median": np.median(x),
                    "y_median": np.median(y),
                    "z_median": np.median(z),
                    "mag_median": np.median(mag),

                    "x_energy": np.sum(x**2) / len(x),
                    "y_energy": np.sum(y**2) / len(y),
                    "z_energy": np.sum(z**2) / len(z),
                    "mag_energy": np.sum(mag**2) / len(mag),

                    "x_iqr": np.percentile(x, 75) - np.percentile(x, 25),
                    "y_iqr": np.percentile(y, 75) - np.percentile(y, 25),
                    "z_iqr": np.percentile(z, 75) - np.percentile(z, 25),
                    "mag_iqr": np.percentile(mag, 75) - np.percentile(mag, 25),
                }

                feature_rows.append(row)

    feature_df = pd.DataFrame(feature_rows)

    return feature_df


def load_wisdm_dataset(raw_root="raw", max_files=None):
    """
    Loads WISDM raw dataset.

    This function tries to read all files under:
        raw/phone/accel
        raw/phone/gyro
        raw/watch/accel
        raw/watch/gyro

    If those folders do not exist, it searches all .txt files under raw/.

    To keep the experiment manageable, this implementation combines available files
    into one WISDM feature dataset. If you only have phone accelerometer files,
    the code still works and uses that subset.
    """
    possible_dirs = [
        os.path.join(raw_root, "phone", "accel"),
        os.path.join(raw_root, "phone", "gyro"),
        os.path.join(raw_root, "watch", "accel"),
        os.path.join(raw_root, "watch", "gyro"),
    ]

    files = []

    for d in possible_dirs:
        if os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "*.txt")))

    if len(files) == 0:
        files = glob.glob(os.path.join(raw_root, "**", "*.txt"), recursive=True)

    files = sorted(files)

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise ValueError("No WISDM .txt files found. Check your raw dataset path.")

    print(f"Number of WISDM files found: {len(files)}")

    all_feature_dfs = []

    for file_path in files:
        sensor_df = read_wisdm_file(file_path)

        if sensor_df.empty:
            continue

        feature_df = extract_wisdm_window_features(
            sensor_df,
            window_size=100,
            step_size=100
        )

        if feature_df.empty:
            continue

        # Add sensor source information from path.
        path_lower = file_path.lower()

        if "phone" in path_lower:
            device = "phone"
        elif "watch" in path_lower:
            device = "watch"
        else:
            device = "unknown_device"

        if "gyro" in path_lower:
            sensor = "gyro"
        elif "accel" in path_lower:
            sensor = "accel"
        else:
            sensor = "unknown_sensor"

        feature_df["device"] = device
        feature_df["sensor"] = sensor

        all_feature_dfs.append(feature_df)

    if len(all_feature_dfs) == 0:
        raise ValueError("WISDM files were found, but no valid windows were extracted.")

    wisdm_features = pd.concat(all_feature_dfs, axis=0).reset_index(drop=True)

    # One-hot encode device and sensor so the classifier can use them.
    wisdm_features = pd.get_dummies(
        wisdm_features,
        columns=["device", "sensor"],
        drop_first=False
    )

    X = wisdm_features.drop(columns=["activity", "user"])
    y = wisdm_features["activity"]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    print("WISDM feature matrix shape:", X.shape)
    print("WISDM class distribution:")
    print(y.value_counts())

    return X.values, y.values, wisdm_features


# ============================================================
# 9. MAIN EXPERIMENT EXECUTION
# ============================================================

# ----------------------------
# A. HAR DATASET
# ----------------------------

X_har, y_har = load_har_dataset(
    train_path="train.csv",
    test_path="test.csv"
)

print("HAR shape:", X_har.shape)
print("HAR classes:", np.unique(y_har))

har_results = run_full_experiment(
    X_har,
    y_har,
    dataset_name="HAR Smartphone"
)


# ----------------------------
# B. WISDM DATASET
# ----------------------------

# Set max_files=None to use all available WISDM files.
# If runtime is too long, temporarily use max_files=20 while testing.
X_wisdm, y_wisdm, wisdm_feature_df = load_wisdm_dataset(
    raw_root="raw",
    max_files=None
)

print("WISDM shape:", X_wisdm.shape)
print("WISDM classes:", np.unique(y_wisdm))

wisdm_results = run_full_experiment(
    X_wisdm,
    y_wisdm,
    dataset_name="WISDM"
)


# ============================================================
# 10. COMBINED RESULTS COMPARISON
# ============================================================

combined_results = pd.concat(
    [har_results, wisdm_results],
    axis=0
).reset_index(drop=True)

print("\n" + "#" * 90)
print("COMBINED HAR VS WISDM RESULTS")
print("#" * 90)

print(
    combined_results.sort_values(
        ["dataset", "test_accuracy"],
        ascending=[True, False]
    ).to_string(index=False)
)

combined_results.to_csv("final_project_results_har_wisdm.csv", index=False)

print("\nSaved combined results to final_project_results_har_wisdm.csv")


# ============================================================
# 11. STABILITY ANALYSIS
# ============================================================

stability_results = []

# Use strong baseline model for stability analysis.
# You can add more models if needed.

stability_results.append(
    stability_analysis(
        X_har,
        y_har,
        dataset_name="HAR Smartphone",
        model=RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        model_name="Random Forest",
        normalization_name="zscore",
        feature_method="original",
        n_runs=N_STABILITY_RUNS
    )
)

stability_results.append(
    stability_analysis(
        X_har,
        y_har,
        dataset_name="HAR Smartphone",
        model=KNeighborsClassifier(n_neighbors=5),
        model_name="kNN",
        normalization_name="zscore",
        feature_method="pca",
        n_runs=N_STABILITY_RUNS
    )
)

stability_results.append(
    stability_analysis(
        X_wisdm,
        y_wisdm,
        dataset_name="WISDM",
        model=RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE),
        model_name="Random Forest",
        normalization_name="zscore",
        feature_method="original",
        n_runs=N_STABILITY_RUNS
    )
)

stability_results.append(
    stability_analysis(
        X_wisdm,
        y_wisdm,
        dataset_name="WISDM",
        model=KNeighborsClassifier(n_neighbors=5),
        model_name="kNN",
        normalization_name="zscore",
        feature_method="pca",
        n_runs=N_STABILITY_RUNS
    )
)

stability_df = pd.DataFrame(stability_results)

print("\n" + "#" * 90)
print("STABILITY ANALYSIS SUMMARY")
print("#" * 90)
print(stability_df.to_string(index=False))

stability_df.to_csv("stability_analysis_results.csv", index=False)

print("\nSaved stability analysis results to stability_analysis_results.csv")


# ============================================================
# 12. SHORT TEXT FOR REPORT
# ============================================================

print("\n" + "#" * 90)
print("REPORT-READY SUMMARY POINTS")
print("#" * 90)

print("""
This implementation evaluates human activity recognition on both the HAR Smartphone
dataset and the WISDM dataset. For preprocessing, both z-score normalization and
min-max normalization are tested. Dimensionality reduction is performed using PCA,
while feature selection is performed using Sequential Forward Feature Selection.

The software-package classifiers include k-nearest neighbors, support vector machines,
and random forests. The Bayesian classifiers include a MAP classifier using
multivariate Gaussian class-conditional densities, Gaussian Naive Bayes, and a
non-parametric KDE/Parzen MAP classifier.

Each model is evaluated using classification accuracy, confusion matrices, and
classification reports. Class confusion analysis is printed by identifying the most
frequent off-diagonal entries in the confusion matrix. Stability is analyzed by
repeating train/validation/test partitioning over multiple random seeds and reporting
the mean and variance of the error rate.
""")

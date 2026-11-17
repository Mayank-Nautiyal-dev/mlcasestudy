"""Train on earlier filing years, tune on a later cohort, and test on the next year."""
import argparse
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import DATA_OUT, FIGURES_OUT, MODELS_OUT, RANDOM_SEED, STATE_CODE, TABLES_OUT


CAT_FEATURES = ["court_name", "case_type", "year"]
TARGET = "delay_over_3_years"


def make_preprocessor():
    categories = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=25,
        max_categories=2000, sparse_output=True,
    )
    acts = CountVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False,
        binary=True, min_df=10, max_features=5000,
    )
    return ColumnTransformer([
        ("categorical", categories, CAT_FEATURES),
        ("act_section", acts, "act_section_tokens"),
    ], sparse_threshold=1.0)


def best_f1_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics_row(name, y, probabilities, threshold, train_n, test_n, scope, threshold_source="validation_year"):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "training_n": train_n,
        "test_n": test_n,
        "test_scope": scope,
        "threshold_from_validation": threshold,
        "threshold_source": threshold_source,
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "f1": f1_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "precision": precision_score(y, predicted, zero_division=0),
        "accuracy": accuracy_score(y, predicted),
        "brier_score": brier_score_loss(y, probabilities),
    }, predicted


def historical_rates(train_frame, frame, alpha=100):
    global_rate = float(train_frame[TARGET].mean())
    court = train_frame.groupby("court_name")[TARGET].agg(["sum", "count"])
    court_rate = (court["sum"] + alpha * global_rate) / (court["count"] + alpha)
    keys = train_frame["court_name"].astype(str) + "\x1f" + train_frame["case_type"].astype(str)
    combo = pd.DataFrame({"key": keys, "y": train_frame[TARGET].astype(int)}).groupby("key")["y"].agg(["sum", "count"])
    combo_rate = (combo["sum"] + alpha * global_rate) / (combo["count"] + alpha)
    query_keys = frame["court_name"].astype(str) + "\x1f" + frame["case_type"].astype(str)
    combo_scores = query_keys.map(combo_rate)
    court_scores = frame["court_name"].map(court_rate)
    return combo_scores.fillna(court_scores).fillna(global_rate).to_numpy(), global_rate


def stratified_sample(X, y, n, seed):
    if len(y) <= n:
        return np.arange(len(y))
    sample_idx, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        np.arange(len(y)), train_size=n, random_state=seed, stratify=y
    )
    return sample_idx


def run(year_start=2010, train_end=2012, validation_year=2013, test_year=2014, state_code=STATE_CODE):
    state_code = str(state_code).zfill(2)
    source = DATA_OUT / f"temporal_features_state{state_code}_{year_start}_{test_year}.csv"
    if not source.exists():
        raise FileNotFoundError(f"Run build_temporal_features.py first: {source}")

    columns = ["ddl_case_id", "year", "court_name", "case_type", "act_section_tokens", TARGET]
    # Keep the in-memory training/validation/test samples bounded. Sampling is
    # independent within each streamed chunk and filing-year role.
    fractions = {"train": 0.40, "validation": 0.20, "test": 0.20}
    pieces = {key: [] for key in fractions}
    reader = pd.read_csv(
        source, usecols=columns,
        dtype={"ddl_case_id": str, "year": str, "court_name": str, "case_type": str, "act_section_tokens": str},
        chunksize=250_000, low_memory=False,
    )
    for chunk_no, part in enumerate(reader):
        part = part.loc[part[TARGET].notna()].copy()
        if part.empty:
            continue
        year_num = pd.to_numeric(part["year"], errors="coerce")
        role_masks = {
            "train": year_num.between(year_start, train_end),
            "validation": year_num.eq(validation_year),
            "test": year_num.eq(test_year),
        }
        for role, mask in role_masks.items():
            selected = part.loc[mask]
            if len(selected):
                pieces[role].append(selected.sample(frac=fractions[role], random_state=RANDOM_SEED + chunk_no))
    frame_parts = {key: pd.concat(value, ignore_index=True) for key, value in pieces.items() if value}
    for role in fractions:
        if role not in frame_parts:
            raise ValueError(f"No labeled rows found for {role} years.")
    for key, sample in frame_parts.items():
        sample[TARGET] = sample[TARGET].astype(int)
        for col in ["court_name", "case_type", "act_section_tokens"]:
            sample[col] = sample[col].fillna("").astype(str)

    train_frame = frame_parts["train"]
    validation_frame = frame_parts["validation"]
    test_frame = frame_parts["test"]
    if min(len(train_frame), len(validation_frame), len(test_frame)) == 0:
        raise ValueError("Train, validation, and test years must each have labeled cases.")
    for name, part in [("train", train_frame), ("validation", validation_frame), ("test", test_frame)]:
        if part[TARGET].nunique() != 2:
            raise ValueError(f"The {name} cohort must contain both outcome classes.")

    X_train = train_frame[CAT_FEATURES + ["act_section_tokens"]]
    X_val = validation_frame[CAT_FEATURES + ["act_section_tokens"]]
    X_test = test_frame[CAT_FEATURES + ["act_section_tokens"]]
    y_train = train_frame[TARGET].to_numpy()
    y_val = validation_frame[TARGET].to_numpy()
    y_test = test_frame[TARGET].to_numpy()

    print(f"Fitting feature encoder on {len(train_frame):,} training cases", flush=True)
    preprocessor = make_preprocessor()
    X_train_sparse = preprocessor.fit_transform(X_train)
    X_val_sparse = preprocessor.transform(X_val)
    X_test_sparse = preprocessor.transform(X_test)
    print(f"Encoded dimensions: {X_train_sparse.shape[1]:,}; train positives: {y_train.mean():.1%}", flush=True)

    models = {
        "logistic_regression_balanced": LogisticRegression(
            max_iter=500, class_weight="balanced", solver="liblinear",
            random_state=RANDOM_SEED,
        ),
        "decision_tree_balanced": DecisionTreeClassifier(
            max_depth=14, min_samples_leaf=500, class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "multinomial_naive_bayes": MultinomialNB(alpha=0.5),
    }
    rows = []
    test_curves = {}
    prediction_table = pd.DataFrame({
        "ddl_case_id": test_frame["ddl_case_id"].to_numpy(),
        "filing_year": test_frame["year"].to_numpy(),
        "y_true": y_test,
    })

    baseline = DummyClassifier(strategy="prior").fit(X_train_sparse, y_train)
    base_test = baseline.predict_proba(X_test_sparse)[:, 1]
    rows.append(metrics_row(
        "prevalence_baseline", y_test, base_test, 0.5, len(y_train), len(y_test),
        "20% random sample of test year", threshold_source="fixed 0.5",
    )[0])
    test_curves["prevalence_baseline"] = (y_test, base_test)
    prediction_table["prevalence_baseline_probability"] = base_test

    hist_val, global_rate = historical_rates(train_frame, validation_frame)
    hist_test, _ = historical_rates(train_frame, test_frame)
    hist_threshold = best_f1_threshold(y_val, hist_val)
    hist_row, hist_prediction = metrics_row("historical_court_case_type_rate", y_test, hist_test, hist_threshold, len(y_train), len(y_test), "20% random sample of test year")
    rows.append(hist_row)
    test_curves["historical_court_case_type_rate"] = (y_test, hist_test)
    prediction_table["historical_rate_probability"] = hist_test
    prediction_table["historical_rate_prediction"] = hist_prediction

    for name, estimator in models.items():
        print(f"Fitting {name}...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name == "multinomial_naive_bayes":
                weights = np.where(y_train == 1, 0.5 / max(y_train.mean(), 1e-9), 0.5 / max(1 - y_train.mean(), 1e-9))
                estimator.fit(X_train_sparse, y_train, sample_weight=weights)
            else:
                estimator.fit(X_train_sparse, y_train)
        val_probability = estimator.predict_proba(X_val_sparse)[:, 1]
        test_probability = estimator.predict_proba(X_test_sparse)[:, 1]
        threshold = best_f1_threshold(y_val, val_probability)
        row, predicted = metrics_row(name, y_test, test_probability, threshold, len(y_train), len(y_test), "20% random sample of test year")
        rows.append(row)
        test_curves[name] = (y_test, test_probability)
        prediction_table[name + "_probability"] = test_probability
        prediction_table[name + "_prediction"] = predicted
        joblib.dump(
            {"preprocessor": preprocessor, "estimator": estimator, "features": CAT_FEATURES + ["act_section_tokens"]},
            MODELS_OUT / f"temporal_{name}_state{state_code}_{year_start}_{train_end}_test{test_year}.joblib",
        )

    # k-NN is expensive in a high-dimensional sparse representation. Evaluate
    # it on stratified subsets while retaining the same chronological split.
    knn_train_idx = stratified_sample(X_train_sparse, y_train, 20_000, RANDOM_SEED)
    knn_val_idx = stratified_sample(X_val_sparse, y_val, 5_000, RANDOM_SEED + 1)
    knn_test_idx = stratified_sample(X_test_sparse, y_test, 5_000, RANDOM_SEED + 2)
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="brute", metric="cosine", n_jobs=-1)
    print(f"Fitting k-NN on {len(knn_train_idx):,} sampled training cases", flush=True)
    knn.fit(X_train_sparse[knn_train_idx], y_train[knn_train_idx])
    knn_val_probability = knn.predict_proba(X_val_sparse[knn_val_idx])[:, 1]
    knn_test_probability = knn.predict_proba(X_test_sparse[knn_test_idx])[:, 1]
    knn_threshold = best_f1_threshold(y_val[knn_val_idx], knn_val_probability)
    knn_row, knn_prediction = metrics_row(
        "knn_sampled", y_test[knn_test_idx], knn_test_probability,
        knn_threshold, len(knn_train_idx), len(knn_test_idx), "stratified 5,000-case test subset",
    )
    rows.append(knn_row)
    test_curves["knn_sampled"] = (y_test[knn_test_idx], knn_test_probability)
    knn_predictions = test_frame.iloc[knn_test_idx][["ddl_case_id", "year"]].copy()
    knn_predictions["y_true"] = y_test[knn_test_idx]
    knn_predictions["probability"] = knn_test_probability
    knn_predictions["prediction"] = knn_prediction
    knn_predictions.to_csv(DATA_OUT / f"temporal_knn_test_sample_state{state_code}_{year_start}_{test_year}.csv", index=False)
    joblib.dump(
        {"preprocessor": preprocessor, "estimator": knn, "features": CAT_FEATURES + ["act_section_tokens"], "training_sample_n": len(knn_train_idx)},
        MODELS_OUT / f"temporal_knn_sampled_state{state_code}_{year_start}_{train_end}_test{test_year}.joblib",
    )

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_path = TABLES_OUT / f"temporal_model_metrics_state{state_code}_{year_start}_{train_end}_val{validation_year}_test{test_year}.csv"
    metrics.to_csv(metrics_path, index=False)
    prediction_table.to_csv(DATA_OUT / f"temporal_test_predictions_state{state_code}_{year_start}_{test_year}.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (labels, probability) in test_curves.items():
        RocCurveDisplay.from_predictions(labels, probability, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(f"Temporal test ROC curves: filing year {test_year}")
    fig.tight_layout()
    fig.savefig(FIGURES_OUT / f"temporal_roc_state{state_code}_{year_start}_{test_year}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (labels, probability) in test_curves.items():
        PrecisionRecallDisplay.from_predictions(labels, probability, name=name, ax=ax)
    ax.set_title(f"Temporal test precision-recall: filing year {test_year}")
    fig.tight_layout()
    fig.savefig(FIGURES_OUT / f"temporal_precision_recall_state{state_code}_{year_start}_{test_year}.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (labels, probability) in test_curves.items():
        CalibrationDisplay.from_predictions(labels, probability, n_bins=10, strategy="quantile", name=name, ax=ax)
    ax.set_title(f"Temporal test calibration: filing year {test_year}")
    fig.tight_layout()
    fig.savefig(FIGURES_OUT / f"temporal_calibration_state{state_code}_{year_start}_{test_year}.png", dpi=160)
    plt.close(fig)

    best = metrics.iloc[0]["model"]
    if best != "knn_sampled":
        best_prob = test_curves[best][1]
        threshold = float(metrics.loc[metrics["model"].eq(best), "threshold_from_validation"].iloc[0])
        best_pred = (best_prob >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(y_test, best_pred, cmap="Blues")
        plt.title(f"Temporal test confusion matrix: {best}")
        plt.tight_layout()
        plt.savefig(FIGURES_OUT / f"temporal_confusion_matrix_state{state_code}_{year_start}_{test_year}.png", dpi=160)
        plt.close()

    year_summary = (
        pd.concat(frame_parts.values(), ignore_index=True).groupby("year")[TARGET]
        .agg(cases="size", delayed="sum", delay_rate="mean")
        .reset_index()
    )
    year_summary.to_csv(TABLES_OUT / f"temporal_prevalence_by_year_state{state_code}_{year_start}_{test_year}.csv", index=False)
    info = {
        "state_code": state_code,
        "train_years": f"{year_start}-{train_end}",
        "validation_year": validation_year,
        "test_year": test_year,
        "train_n": int(len(train_frame)),
        "validation_n": int(len(validation_frame)),
        "test_n": int(len(test_frame)),
        "train_prevalence": float(train_frame[TARGET].mean()),
        "validation_prevalence": float(validation_frame[TARGET].mean()),
        "test_prevalence": float(test_frame[TARGET].mean()),
        "global_training_rate_for_baselines": global_rate,
        "feature_encoding": "one-hot with rare-category grouping plus binary act-section tokens; fitted on training years only",
        "threshold_selection": "maximize F1 on validation year; fixed before test scoring",
        "knn_scope": "20,000 training cases, 5,000 validation cases, 5,000 test cases, stratified samples",
        "temporal_sample_fractions": fractions,
        "label_rule": "valid disposal date after third calendar anniversary; pending cases with DDL disposition-missing code and plausible last-list date after anniversary are positive; other unresolved cases censored/excluded",
        "last_list_plausibility_ceiling": "2021-12-31 (data-quality ceiling, not source cutoff)",
        "best_model_by_roc_auc": str(best),
    }
    (TABLES_OUT / f"temporal_run_info_state{state_code}_{year_start}_{train_end}_val{validation_year}_test{test_year}.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(f"Temporal metrics written: {metrics_path}")
    print(metrics.to_string(index=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-start", type=int, default=2010)
    parser.add_argument("--train-end", type=int, default=2012)
    parser.add_argument("--validation-year", type=int, default=2013)
    parser.add_argument("--test-year", type=int, default=2014)
    parser.add_argument("--state-code", default=STATE_CODE)
    args = parser.parse_args()
    run(args.year_start, args.train_end, args.validation_year, args.test_year, args.state_code)

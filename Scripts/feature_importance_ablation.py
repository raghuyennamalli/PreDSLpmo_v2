#!/usr/bin/env python3

from __future__ import annotations
import gc
import json
import os
import random
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import check_random_state

ROOT = Path(
    "/media/ragothaman/yennamalli/Sweety/LPMO_NEW/New_Version/"
    "Cluster_Aware_Spliting_Changed"
)
FEAT_ROOT = ROOT / "Training_Validation_Final" / "Feature_Extraction"
CV_ROOT = FEAT_ROOT / "Cluster_Aware_CV"

TRAIN_CSV = (
    FEAT_ROOT / "combined_dataset" / "multiclass_training"
    / "PredLPMO_multiclass_training_clean.csv"
)
FOLD_CSV = CV_ROOT / "fold_assignments" / "multiclass_training_cluster_aware_5fold.csv"
EFS_TXT = FEAT_ROOT / "EFS" / "multiclass_results" / "multiclass_features_gt_0.5.txt"

NN_MODEL_PATH = CV_ROOT / "Feature_NN_FINAL" / "feature_nn_final.keras"
NN_SCALER_PATH = CV_ROOT / "Feature_NN_FINAL" / "feature_nn_scaler.joblib"
NN_METADATA_PATH = CV_ROOT / "Feature_NN_FINAL" / "feature_nn_metadata.json"
NN_HISTORY_PATH = CV_ROOT / "Feature_NN_FINAL" / "final_training_history.csv"

SGD_MODELS = {
    "SGD_log_loss": CV_ROOT / "SGD_FINAL_MODELS" / "multiclass_SGD_log_loss.joblib",
    "SGD_hinge": CV_ROOT / "SGD_FINAL_MODELS" / "multiclass_SGD_hinge.joblib",
    "SGD_modified_huber": CV_ROOT / "SGD_FINAL_MODELS" / "multiclass_SGD_modified_huber.joblib",
}

OUT_DIR = CV_ROOT / "Feature_Importance_Ablation_Results"
RANDOM_SEED = 42
PERMUTATION_REPEATS = 5
FALLBACK_EPOCHS = 30
BATCH_SIZE = 32
VERBOSE_NN = 0
N_CLASSES = 9

ID_COL = "Sequence ID_#"
LABEL_COL = "Class"
FOLD_COL = "Fold"
CLUSTER_COL = "Cluster_ID"

CLASS_NAMES = [
    "AA9", "AA10", "AA11", "AA13", "AA14",
    "AA15", "AA16", "AA17", "Non-LPMO"
]

EXPECTED_FEATURES = [
    "CKSAAP_Feature_1182_YC.gap2",
    "CKSAAP_Feature_1862_QC.gap4",
    "APAAC_Feature_11_Pc1.L",
    "TPC_Feature_7289_WFK",
    "CKSAAP_Feature_1993_YP.gap4",
    "CTDD_Feature_66_hydrophobicity_CASG920101.2.residue0",
    "TPC_Feature_6034_SCQ",
    "PAAC_Feature_11_Xc1.L",
    "CTDD_Feature_156_charge.2.residue0",
    "CKSAAP_Feature_434_CQ.gap1",
    "DDE_Feature_34_CQ",
    "DPC_Feature_246_PG",
]


def log(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def read_feature_list(path: Path) -> list[str]:
    """Read feature names from the EFS text file, with safe fallbacks."""
    if not path.exists():
        warnings.warn(f"EFS list not found: {path}. Using embedded 12-feature list.")
        return EXPECTED_FEATURES.copy()

    text = path.read_text(encoding="utf-8", errors="replace")
    found = []
    for raw in text.splitlines():
        line = raw.strip().strip('"').strip("'")
        if not line or line.startswith("#"):
            continue
        # Supports plain one-feature-per-line and simple indexed output.
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            candidates = parts[::-1] + parts
        elif "," in line and not line.startswith(("CKSAAP_", "APAAC_", "TPC_", "CTDD_", "PAAC_", "DDE_", "DPC_")):
            candidates = [p.strip() for p in line.split(",")]
        else:
            candidates = [line]
        match = next((p for p in candidates if p in EXPECTED_FEATURES), None)
        if match and match not in found:
            found.append(match)

    # If parsing doesn't recover all 12 exact names, use the known validated list.
    if set(found) != set(EXPECTED_FEATURES):
        log("Could not parse all 12 feature names from EFS file; using the embedded validated list.")
        return EXPECTED_FEATURES.copy()
    # Preserve the order defined in the EFS file.
    return found


def macro_metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def load_inputs():
    for path in [TRAIN_CSV, FOLD_CSV]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    features = read_feature_list(EFS_TXT)
    data = pd.read_csv(TRAIN_CSV, low_memory=False)
    folds = pd.read_csv(FOLD_CSV, low_memory=False)

    for col in [ID_COL, LABEL_COL]:
        if col not in data.columns:
            raise KeyError(f"Training CSV must contain column {col!r}. Columns include: {list(data.columns[:20])}")
    for col in [ID_COL, LABEL_COL, FOLD_COL, CLUSTER_COL]:
        if col not in folds.columns:
            raise KeyError(f"Fold file must contain column {col!r}. Found: {list(folds.columns)}")

    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise KeyError("These EFS features are missing from the training CSV:\n" + "\n".join(missing_features))

    if data[ID_COL].duplicated().any():
        n = int(data[ID_COL].duplicated(keep=False).sum())
        raise ValueError(
            f"Training CSV has {n} rows with duplicated sequence IDs. "
            "Resolve/inspect duplicate IDs before joining folds; a many-to-many merge would invalidate CV."
        )
    if folds[ID_COL].duplicated().any():
        n = int(folds[ID_COL].duplicated(keep=False).sum())
        raise ValueError(f"Fold-assignment file has duplicate IDs ({n} rows); cannot safely merge.")

    # Keep only the fold metadata needed; do not merge the Class column twice.
    fold_meta = folds[[ID_COL, LABEL_COL, FOLD_COL, CLUSTER_COL]].copy()
    merged = data.merge(
        fold_meta, on=ID_COL, how="inner", suffixes=("_data", "_fold"),
        validate="one_to_one"
    )
    if merged.empty:
        raise ValueError("No IDs matched between training CSV and fold assignments.")

    label_data = pd.to_numeric(merged[f"{LABEL_COL}_data"], errors="coerce")
    label_fold = pd.to_numeric(merged[f"{LABEL_COL}_fold"], errors="coerce")
    if label_data.isna().any() or label_fold.isna().any():
        raise ValueError("Class labels must be numeric integers 0–8 in both files.")
    if not np.array_equal(label_data.to_numpy(dtype=int), label_fold.to_numpy(dtype=int)):
        mismatch = merged.loc[label_data.to_numpy(dtype=int) != label_fold.to_numpy(dtype=int), ID_COL].head(10).tolist()
        raise ValueError(f"Training labels disagree with fold labels for IDs such as: {mismatch}")

    if len(merged) != len(data) or len(merged) != len(folds):
        log(f"Rows: training={len(data)}, fold assignments={len(folds)}, matched={len(merged)}.")
        log("Only IDs present in both files are used; check this row-count difference before interpreting results.")

    merged[LABEL_COL] = label_data.astype(int)
    merged[FOLD_COL] = pd.to_numeric(merged[FOLD_COL], errors="raise").astype(int)
    if not set(merged[LABEL_COL].unique()).issubset(set(range(N_CLASSES))):
        raise ValueError("Unexpected class label; expected integer classes 0–8.")
    if merged[features].isna().any().any():
        bad = merged[features].columns[merged[features].isna().any()].tolist()
        raise ValueError(f"Missing values found in selected features: {bad}")
    merged[features] = merged[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(merged[features].to_numpy(dtype=float)).all():
        raise ValueError("NaN/inf values found in the selected feature matrix.")

    return merged, features


def get_nn_epochs() -> int:
    if NN_HISTORY_PATH.exists():
        try:
            hist = pd.read_csv(NN_HISTORY_PATH)
            if len(hist) > 0:
                # Match the duration of the saved training history.
                return int(len(hist))
        except Exception as exc:
            warnings.warn(f"Could not read NN training history ({exc}); using {FALLBACK_EPOCHS} epochs.")
    return FALLBACK_EPOCHS


def load_model_templates():
    import tensorflow as tf

    for path in [NN_MODEL_PATH, NN_SCALER_PATH, *SGD_MODELS.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Required saved model/scaler not found: {path}")

    nn_template = tf.keras.models.load_model(NN_MODEL_PATH)
    nn_scaler = joblib.load(NN_SCALER_PATH)
    sgd_templates = {name: joblib.load(path) for name, path in SGD_MODELS.items()}

    expected_n = len(EXPECTED_FEATURES)
    try:
        model_input_n = int(nn_template.input_shape[-1])
        if model_input_n != expected_n:
            raise ValueError(
                f"Saved Feature NN expects {model_input_n} input features, but this analysis uses "
                f"{expected_n}. Confirm the selected-feature/model version before proceeding."
            )
    except AttributeError:
        raise ValueError("Could not determine the saved Feature NN input shape; inspect the saved model.")

    for name, estimator in sgd_templates.items():
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is not None and int(n_in) != expected_n:
            raise ValueError(f"{name} expects {n_in} features, not {expected_n}.")

    return tf, nn_template, nn_scaler, sgd_templates


def clone_nn_for_features(tf, original, n_features: int):
    try:
        inp = tf.keras.Input(shape=(n_features,))
        model = tf.keras.models.clone_model(original, input_tensors=inp)
    except Exception as first_error:
        try:
            if isinstance(original, tf.keras.Sequential):
                rebuilt = tf.keras.Sequential()
                rebuilt.add(tf.keras.layers.Input(shape=(n_features,)))
                for layer in original.layers:
                    if isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    cfg = layer.get_config()
                    for key in ("batch_input_shape", "batch_shape", "input_shape"):
                        cfg.pop(key, None)
                    rebuilt.add(layer.__class__.from_config(cfg))
                model = rebuilt
            else:
                raise first_error
        except Exception as fallback_error:
            raise RuntimeError(
                f"Could not clone Feature NN for {n_features} input features. "
                "The architecture must be rebuilt to support reduced inputs. "
                f"Original clone error: {first_error}; fallback error: {fallback_error}"
            ) from fallback_error

    try:
        optimizer_cfg = tf.keras.optimizers.serialize(original.optimizer)
        optimizer = tf.keras.optimizers.deserialize(optimizer_cfg)
    except Exception:
        optimizer = tf.keras.optimizers.Adam()
    loss = original.loss if getattr(original, "loss", None) is not None else "sparse_categorical_crossentropy"
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    try:
        _ = model(np.zeros((1, n_features), dtype=np.float32), training=False)
    except Exception as exc:
        raise RuntimeError(f"Cloned Feature NN cannot accept {n_features} features: {exc}") from exc
    return model


def fit_sgd(template, X_train, y_train, X_test):
    estimator = clone(template)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    return estimator, np.asarray(pred, dtype=int)


def fit_nn(tf, nn_template, scaler_template, X_train, y_train, X_test, epochs, seed):
    scaler = clone(scaler_template)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    model = clone_nn_for_features(tf, nn_template, Xtr.shape[1])

    tf.keras.utils.set_random_seed(seed)
    model.fit(
        Xtr, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE_NN,
        shuffle=True,
    )
    probs = model.predict(Xte, verbose=0)
    pred = np.argmax(probs, axis=1).astype(int)
    return (model, scaler), pred, Xte


def predict_nn_cached(model, scaler, X):
    Xs = scaler.transform(X)
    probs = model.predict(Xs, verbose=0)
    return np.argmax(probs, axis=1).astype(int)


def cross_validated_baselines(data, features, folds, tf, nn_template, nn_scaler, sgd_templates, epochs):
    results = []
    cache = {name: [] for name in ["Feature_NN", *sgd_templates.keys()]}

    for fold in folds:
        train = data[data[FOLD_COL] != fold]
        test = data[data[FOLD_COL] == fold]
        Xtr = train[features].to_numpy(dtype=np.float32)
        Xte = test[features].to_numpy(dtype=np.float32)
        ytr = train[LABEL_COL].to_numpy(dtype=int)
        yte = test[LABEL_COL].to_numpy(dtype=int)
        log(f"Baseline fold {fold}: train={len(train)}, test={len(test)}")

        (nn_model, nn_fold_scaler), nn_pred, _ = fit_nn(
            tf, nn_template, nn_scaler, Xtr, ytr, Xte, epochs,
            RANDOM_SEED + int(fold)
        )
        row = {"model": "Feature_NN", "fold": int(fold), **macro_metrics(yte, nn_pred)}
        results.append(row)
        cache["Feature_NN"].append({
            "fold": int(fold), "X_test": Xte, "y_test": yte,
            "model": nn_model, "scaler": nn_fold_scaler,
            "baseline_pred": nn_pred,
        })

        for name, template in sgd_templates.items():
            fitted, pred = fit_sgd(template, Xtr, ytr, Xte)
            row = {"model": name, "fold": int(fold), **macro_metrics(yte, pred)}
            results.append(row)
            cache[name].append({
                "fold": int(fold), "X_test": Xte, "y_test": yte,
                "model": fitted, "baseline_pred": pred,
            })

    return pd.DataFrame(results), cache


def compute_permutation_importance(features, cache):
    """Permutation importance on held-out fold predictions; no model refitting."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []

    for model_name, fold_items in cache.items():
        log(f"Permutation importance: {model_name}")
        for item in fold_items:
            Xtest = item["X_test"]
            ytest = item["y_test"]
            baseline = f1_score(ytest, item["baseline_pred"], average="macro", zero_division=0)

            for j, feature in enumerate(features):
                drops = []
                for rep in range(PERMUTATION_REPEATS):
                    Xperm = Xtest.copy()
                    perm_index = rng.permutation(len(Xperm))
                    Xperm[:, j] = Xperm[perm_index, j]
                    if model_name == "Feature_NN":
                        pred = predict_nn_cached(item["model"], item["scaler"], Xperm)
                    else:
                        pred = np.asarray(item["model"].predict(Xperm), dtype=int)
                    perm_score = f1_score(ytest, pred, average="macro", zero_division=0)
                    drops.append(baseline - perm_score)
                rows.append({
                    "model": model_name,
                    "fold": item["fold"],
                    "feature": feature,
                    "baseline_macro_f1": baseline,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std_repeats": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
                })

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance_mean", "mean"),
            std_across_folds=("importance_mean", "std"),
            mean_repeat_sd=("importance_std_repeats", "mean"),
        )
    )
    summary["std_across_folds"] = summary["std_across_folds"].fillna(0.0)
    summary["rank"] = summary.groupby("model")["mean_importance"].rank(
        ascending=False, method="min"
    ).astype(int)
    return fold_df, summary.sort_values(["model", "rank"])


def evaluate_ablation_configurations(data, features, folds, tf, nn_template, nn_scaler, sgd_templates, epochs, importance_summary):
    """Retrain each model on each fold for each feature subset."""
    configurations = [("baseline_all_12", features.copy())]
    for feature in features:
        configurations.append((f"leave_out__{feature}", [f for f in features if f != feature]))

    avg_rank = (
        importance_summary.groupby("feature", as_index=False)["rank"].mean()
        .sort_values("rank", ascending=True)
    )
    top5 = avg_rank["feature"].head(5).tolist()
    bottom5 = avg_rank["feature"].tail(5).tolist()
    configurations.append(("remove_top5_mean_rank", [f for f in features if f not in top5]))
    configurations.append(("remove_bottom5_mean_rank", [f for f in features if f not in bottom5]))

    rows = []
    model_names = ["Feature_NN", *sgd_templates.keys()]
    for config_index, (config_name, selected) in enumerate(configurations):
        log(f"Ablation {config_index + 1}/{len(configurations)}: {config_name} ({len(selected)} features)")
        if not selected:
            raise RuntimeError(f"Configuration {config_name} has no features.")

        for fold in folds:
            train = data[data[FOLD_COL] != fold]
            test = data[data[FOLD_COL] == fold]
            Xtr = train[selected].to_numpy(dtype=np.float32)
            Xte = test[selected].to_numpy(dtype=np.float32)
            ytr = train[LABEL_COL].to_numpy(dtype=int)
            yte = test[LABEL_COL].to_numpy(dtype=int)

            _, nn_pred, _ = fit_nn(
                tf, nn_template, nn_scaler, Xtr, ytr, Xte, epochs,
                RANDOM_SEED + 1000 + config_index * 10 + int(fold)
            )
            rows.append({
                "configuration": config_name,
                "n_features": len(selected),
                "model": "Feature_NN",
                "fold": int(fold),
                **macro_metrics(yte, nn_pred),
                "features_used": ";".join(selected),
            })

            for name, template in sgd_templates.items():
                _, pred = fit_sgd(template, Xtr, ytr, Xte)
                rows.append({
                    "configuration": config_name,
                    "n_features": len(selected),
                    "model": name,
                    "fold": int(fold),
                    **macro_metrics(yte, pred),
                    "features_used": ";".join(selected),
                })

    fold_results = pd.DataFrame(rows)
    metric_cols = ["macro_f1", "weighted_f1", "accuracy", "macro_precision", "macro_recall"]
    summary = (
        fold_results.groupby(["configuration", "n_features", "model"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part) != "")
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.reset_index(drop=True)
    return fold_results, summary, {"top5_mean_rank": top5, "bottom5_mean_rank": bottom5}


def make_plots(importance_summary: pd.DataFrame, ablation_summary: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"matplotlib unavailable; CSV outputs are still saved. Plot error: {exc}")
        return

    for model_name, sub in importance_summary.groupby("model"):
        sub = sub.sort_values("mean_importance", ascending=True)
        plt.figure(figsize=(10, 6))
        plt.barh(sub["feature"], sub["mean_importance"], xerr=sub["std_across_folds"])
        plt.axvline(0, linewidth=0.8)
        plt.xlabel("Permutation importance (baseline Macro-F1 − permuted Macro-F1)")
        plt.ylabel("EFS-selected feature")
        plt.title(f"Permutation feature importance: {model_name}")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"permutation_importance_{model_name}.png", dpi=300)
        plt.close()

    macro_mean_col = "macro_f1_mean"
    if macro_mean_col in ablation_summary.columns:
        for model_name, sub in ablation_summary.groupby("model"):
            sub = sub.sort_values(macro_mean_col, ascending=True)
            plt.figure(figsize=(12, max(6, 0.34 * len(sub))))
            plt.barh(sub["configuration"], sub[macro_mean_col], xerr=sub.get("macro_f1_std"))
            plt.xlabel("Cross-validated Macro-F1 (mean across folds)")
            plt.ylabel("Feature configuration")
            plt.title(f"Feature ablation: {model_name}")
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"ablation_macro_f1_{model_name}.png", dpi=300)
            plt.close()


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, features = load_inputs()
    folds = sorted(data[FOLD_COL].unique().tolist())
    if len(folds) < 2:
        raise ValueError(f"Need at least 2 folds; found {folds}")

    log(f"Using {len(features)} selected features and folds {folds}.")
    log(f"Matched training rows: {len(data)}; class counts: {data[LABEL_COL].value_counts().sort_index().to_dict()}")

    tf, nn_template, nn_scaler, sgd_templates = load_model_templates()
    epochs = get_nn_epochs()
    log(f"Feature NN retraining epochs: {epochs}")

    # Explicitly verify reduced-dimension NN cloning before the full run.
    log("Testing Feature NN cloning at 12 and 11 input features...")
    _ = clone_nn_for_features(tf, nn_template, len(features))
    _ = clone_nn_for_features(tf, nn_template, len(features) - 1)
    log("Feature NN cloning test passed.")

    baseline_df, cache = cross_validated_baselines(
        data, features, folds, tf, nn_template, nn_scaler, sgd_templates, epochs
    )
    baseline_df.to_csv(OUT_DIR / "cv_baseline_metrics_by_fold.csv", index=False)
    baseline_summary = (
        baseline_df.groupby("model", as_index=False)[
            ["macro_f1", "weighted_f1", "accuracy", "macro_precision", "macro_recall"]
        ].agg(["mean", "std"])
    )
    baseline_summary.columns = [
        "_".join(str(part) for part in col if str(part) != "")
        if isinstance(col, tuple) else col
        for col in baseline_summary.columns
    ]
    baseline_summary.to_csv(OUT_DIR / "cv_baseline_metrics_summary.csv", index=False)

    perm_fold, perm_summary = compute_permutation_importance(features, cache)
    perm_fold.to_csv(OUT_DIR / "permutation_importance_by_fold.csv", index=False)
    perm_summary.to_csv(OUT_DIR / "permutation_importance_summary.csv", index=False)

    # Release cached fold models before retraining many ablation configurations.
    for model_items in cache.values():
        for item in model_items:
            if item.get("model") is not None:
                del item["model"]
    del cache
    gc.collect()

    ablation_fold, ablation_summary, groups = evaluate_ablation_configurations(
        data, features, folds, tf, nn_template, nn_scaler, sgd_templates, epochs, perm_summary
    )
    ablation_fold.to_csv(OUT_DIR / "ablation_metrics_by_fold.csv", index=False)
    ablation_summary.to_csv(OUT_DIR / "ablation_metrics_summary.csv", index=False)
    with open(OUT_DIR / "top_bottom_feature_groups.json", "w", encoding="utf-8") as handle:
        json.dump(groups, handle, indent=2)

    make_plots(perm_summary, ablation_summary)

    log("Finished successfully.")
    log(f"Results directory: {OUT_DIR}")


if __name__ == "__main__":
    main()

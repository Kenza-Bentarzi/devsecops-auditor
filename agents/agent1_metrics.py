"""
Agent 1 - Code Metrics Evaluator (Random Forest) | NASA/PROMISE
Usage (run from the project root):
    python agents/agent1_metrics.py train --data data/pc1.csv
    python -c "from agents.agent1_metrics import predict_struct_risk; print(predict_struct_risk({...}))"
Requires: pandas numpy scikit-learn imbalanced-learn joblib
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline  # SMOTE is applied ONLY on training folds
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)

ARTIFACT = Path("models/agent1_rf.joblib")
SEED = 42
LABEL_MAP = {"true": 1, "yes": 1, "y": 1, "1": 1, "false": 0, "no": 0, "n": 0, "0": 0}


# ---------------------------------------------------------------- 1. Loading
def load_data(path):
    """Load a NASA/PROMISE CSV (PC1, JM1, KC1...). '?' = missing value."""
    df = pd.read_csv(path, na_values=["?", "", "NA"])
    df.columns = [c.strip().lower().replace("(", "").replace(")", "") for c in df.columns]
    label_col = next(c for c in ("defects", "defective", "label", "bug") if c in df.columns)
    y = df.pop(label_col).astype(str).str.strip().str.lower().map(LABEL_MAP)
    if y.isna().any():
        raise ValueError("Unknown label values: inspect the dataset's label encoding.")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates()          # NASA data has many duplicates -> leakage risk
    return df, y.loc[df.index].astype(int)


# --------------------------------------------------- 2. Feature engineering
def add_features(df):
    """Deterministic, label-free ratios (same code at train and inference).
    NOTE: 'bug density' must NOT be computed from the label (that is leakage);
    we use complexity/size densities instead."""
    d = df.copy()
    g = lambda c: d[c] if c in d.columns else None
    loc, vg, e = g("loc"), g("vg"), g("e")
    if loc is not None:
        if vg is not None:
            d["complexity_density"] = vg / (loc + 1)
        if e is not None:
            d["effort_per_loc"] = e / (loc + 1)
        if g("locomment") is not None:
            d["comment_ratio"] = d["locomment"] / (loc + 1)
        if g("branchcount") is not None:
            d["branch_density"] = d["branchcount"] / (loc + 1)
    if g("uniq_opnd") is not None and g("total_opnd") is not None:
        d["operand_reuse"] = d["total_opnd"] / (d["uniq_opnd"] + 1)
    return d.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- 3. Training
def build_pipeline():
    # No scaling: Random Forest is invariant to monotonic feature scaling.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("smote", SMOTE(random_state=SEED)),
        ("rf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])


def choose_threshold(model, X_tr, y_tr):
    """Pick the decision threshold from out-of-fold predictions on the TRAIN set only
    (never from the test set -> no leakage). F2 favors recall over precision, because
    missing a defect is worse than a false alarm."""
    oof = cross_val_predict(
        model, X_tr, y_tr,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f2 = 5 * prec[:-1] * rec[:-1] / (4 * prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[int(np.argmax(f2))])


def train(data_path):
    X_raw, y = load_data(data_path)
    raw_cols = list(X_raw.columns)
    X = add_features(X_raw)
    feature_cols = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"Samples: {len(y)} | defect rate: {y.mean():.2%}")

    search = RandomizedSearchCV(
        build_pipeline(),
        {
            "rf__n_estimators": [200, 300, 500],
            "rf__max_depth": [None, 6, 10, 15],
            "rf__min_samples_leaf": [1, 2, 5],
            "rf__class_weight": [None, "balanced"],
            "smote__k_neighbors": [3, 5],
        },
        n_iter=20, scoring="average_precision",
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    model = search.best_estimator_
    print("Best params:", search.best_params_)

    threshold = choose_threshold(model, X_tr, y_tr)
    print(f"Chosen threshold (train CV, F2): {threshold:.3f}")

    # Baseline for comparison (required by the checklist)
    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    print("Baseline recall(defect):", recall_score(y_te, base.predict(X_te), zero_division=0))

    proba = model.predict_proba(X_te)[:, 1]
    for name, t in (("default 0.5", 0.5), (f"chosen {threshold:.3f}", threshold)):
        pred = (proba >= t).astype(int)
        print(f"\n=== Threshold: {name} ===")
        print(classification_report(y_te, pred, digits=3, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    print("\nROC-AUC:", round(roc_auc_score(y_te, proba), 3))

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "raw_cols": raw_cols, "feature_cols": feature_cols,
                 "threshold": threshold}, ARTIFACT)
    ARTIFACT.with_suffix(".meta.json").write_text(json.dumps(
        {"raw_cols": raw_cols, "feature_cols": feature_cols, "dataset": str(data_path),
         "seed": SEED, "threshold": threshold, "best_params": search.best_params_},
        indent=2, default=str))
    print("Saved ->", ARTIFACT)


# --------------------------------------------------------------- 4. Inference
_bundle = None


def _error(reason):
    return {"struct_risk_score": None, "high_risk": None, "status": "error", "reason": reason}


def predict_struct_risk(metrics: dict) -> dict:
    """Standardized output for Agent 3. Never raises: on any problem (missing model,
    missing/invalid feature) returns status='error' and score=None, so the gatekeeper
    must route to REVIEW (fail-safe), never to PASS.
    NOTE: struct_risk_score is an uncalibrated model score, not a calibrated probability."""
    global _bundle
    try:
        if _bundle is None:
            _bundle = joblib.load(ARTIFACT)
        raw_cols = _bundle["raw_cols"]
        normalized = {str(k).strip().lower().replace("(", "").replace(")", ""): v
                      for k, v in metrics.items()}
        missing = [c for c in raw_cols if c not in normalized]
        if missing:
            return _error(f"missing features: {missing}")
        row = pd.DataFrame([{c: normalized[c] for c in raw_cols}]).apply(
            pd.to_numeric, errors="coerce")
        if row.isna().any().any():  # do NOT silently impute at inference time
            bad = row.columns[row.isna().iloc[0]].tolist()
            return _error(f"invalid or missing numeric values: {bad}")
        X = add_features(row)[_bundle["feature_cols"]]
        score = float(_bundle["model"].predict_proba(X)[0, 1])
        threshold = float(_bundle.get("threshold", 0.5))
        return {"struct_risk_score": round(score, 4), "high_risk": score >= threshold,
                "threshold": round(threshold, 4), "status": "ok", "reason": ""}
    except Exception as exc:  # model file missing, corrupted, etc.
        return _error(str(exc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train"])
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    train(args.data)
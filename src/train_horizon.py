"""Experiment: train the models of ``train.py`` on the 3-year horizon label instead of the status label.

Not the app model: nothing in ``app/`` loads what this writes. It answers the team's questions before
anyone switches the label (root CLAUDE.md, "Open decisions"):

1. How well does a model trained on "terminated within 3 years" (start 2008-2018) work on later trials,
   before COVID (valid = start 2019) and after it (test = start 2020 up to the follow-up cutoff)?
2. Is that better than the status label? Two references are scored against the same horizon labels:
   the current app model as it is (``models/gbm.joblib``: status label, start 2008-2016) and the same
   GBM refitted on the status label of the 2008-2018 trials that have ended (same years, other label).

Same pipelines, grids, 5-fold CV, isotonic calibration and F1-threshold rule as ``train.py``; only the
label, the rows and the split differ. Input: ``data/processed/trials_3y.parquet`` (``build_dataset.py``).
Output: ``models/horizon_3y/`` (``models/horizon_3y_phase1-3/`` with ``--phases 1-3``): ``gbm.joblib``,
``gbm_uncalibrated.joblib``, ``baseline_lr.joblib``, ``score_reference.npy`` (all git-ignored) and
``metrics.json`` (commit it). Every number in the JSON comes from train-only fitting; valid and test are
each scored once, after all models are saved.

Usage:
    python src/train_horizon.py               # all phases; about 10-15 minutes on 4 cores
    python src/train_horizon.py --phases 1-3  # only Phase 1, 2, 3, 1/2 and 2/3 trials
"""

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
import horizon  # noqa: E402
import train  # noqa: E402
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS,  # noqa: E402
                      NON_FEATURE_COLUMNS, NUMERIC_FEATURES)
from predict import load_model, to_model_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "trials_3y.parquet"
MODELS_DIR = ROOT / "models"
APP_MODEL_PATH = MODELS_DIR / "gbm.joblib"
PHASES_1_3 = ["PHASE1", "PHASE2", "PHASE3", "PHASE1/PHASE2", "PHASE2/PHASE3"]
ENDED = ["COMPLETED", "TERMINATED"]
DRIFT_TOP = 12  # features listed in the printed drift table


def output_dir(phases: str) -> Path:
    """Where one variant of the experiment writes its files."""
    return MODELS_DIR / ("horizon_3y" if phases == "all" else "horizon_3y_phase1-3")


def check_dataset(df: pd.DataFrame) -> pd.Timestamp:
    """Fail unless the file follows the horizon rules; returns the follow-up cutoff.

    Insurance against training on a stale file or one that carries outcome information.
    """
    if set(FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS):
        raise ValueError("Meta/display columns listed as features")
    leaked = set(df.columns) & set(horizon.OUTCOME_COLUMNS)
    if leaked:
        raise ValueError(f"Outcome columns in a training file: {sorted(leaked)}")
    missing = [c for c in FEATURE_COLUMNS + ["label", "start_date", "split", "overall_status"] if c not in df]
    if missing:
        raise ValueError(f"trials_3y.parquet lacks columns: {missing}")
    if "followup_cutoff" not in df.attrs:
        raise ValueError("trials_3y.parquet has no follow-up cutoff; rebuild it with src/build_dataset.py")
    cutoff = pd.Timestamp(df.attrs["followup_cutoff"])
    if df["label"].isna().any() or not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0/1 everywhere")
    if not set(df["overall_status"]) <= set(horizon.LABELLED_STATUSES):
        raise ValueError(f"Unexpected statuses: {set(df['overall_status']) - set(horizon.LABELLED_STATUSES)}")
    if (df["label"].eq(1) & df["overall_status"].ne("TERMINATED")).any():
        raise ValueError("label 1 on a trial that was not terminated")
    if df["phase"].isna().any():
        raise ValueError("rows without a phase; rebuild with src/build_dataset.py")
    for split, (first, last) in horizon.SPLITS.items():
        in_window = df["start_date"].between(first, last if last is not None else cutoff)
        if not (in_window == df["split"].eq(split)).all():
            raise ValueError(f"split='{split}' rows do not match their start-date window; rebuild the file")
    return cutoff


def part(df: pd.DataFrame, split: str) -> tuple:
    """(X, y, rows) of one split; X holds only FEATURE_COLUMNS."""
    rows = df[df["split"] == split]
    return to_model_frame(rows), rows["label"].to_numpy(dtype=int), rows


def per_start_year(y, proba, years) -> list:
    """Size, label rate, mean prediction and ranking quality per start year."""
    out = []
    for year in sorted(set(years)):
        mask = np.asarray(years) == year
        y_year, p_year = np.asarray(y)[mask], np.asarray(proba)[mask]
        both = len(set(y_year)) == 2
        out.append({"start_year": int(year), "n": int(mask.sum()), "rate": float(y_year.mean()),
                    "mean_predicted": float(p_year.mean()),
                    "roc_auc": float(roc_auc_score(y_year, p_year)) if both else None,
                    "pr_auc": float(average_precision_score(y_year, p_year)) if both else None})
    return out


def drift(train_rows: pd.DataFrame, test_rows: pd.DataFrame) -> list:
    """How far each feature moved between the train and test years, largest first.

    Numbers and flags: standardised mean difference (test minus train, in pooled standard deviations).
    Categories: total variation distance (half the summed absolute difference of category shares; 0 = same
    mix, 1 = no overlap). Both read "bigger = more different".
    """
    rows = []
    for col in NUMERIC_FEATURES + BINARY_FEATURES:
        a = pd.to_numeric(train_rows[col], errors="coerce").astype(float)
        b = pd.to_numeric(test_rows[col], errors="coerce").astype(float)
        sd = np.sqrt((a.var() + b.var()) / 2)
        rows.append({"feature": col, "kind": "standardised mean difference", "train": float(a.mean()),
                     "test": float(b.mean()), "difference": float((b.mean() - a.mean()) / sd) if sd > 0 else 0.0})
    for col in CATEGORICAL_FEATURES:
        a = train_rows[col].fillna("missing").value_counts(normalize=True)
        b = test_rows[col].fillna("missing").value_counts(normalize=True)
        tvd = 0.5 * a.sub(b, fill_value=0).abs().sum()
        rows.append({"feature": col, "kind": "total variation distance", "train": str(a.idxmax()),
                     "test": str(b.idxmax()), "difference": float(tvd)})
    return sorted(rows, key=lambda r: -abs(r["difference"]))


def split_summary(df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Window, size and label rate of each split, for the metrics JSON."""
    out = {"rule": "by start_date, never random", "followup_cutoff": f"{cutoff:%Y-%m-%d}"}
    for split, (first, last) in horizon.SPLITS.items():
        rows = df[df["split"] == split]
        out[split] = {"start_from": f"{first:%Y-%m-%d}", "start_to": f"{(last or cutoff):%Y-%m-%d}",
                      "n_trials": int(len(rows)), "n_terminated_3y": int(rows["label"].sum()),
                      "rate": float(rows["label"].mean()) if len(rows) else None,
                      "n_still_running": int(rows["overall_status"].isin(horizon.STILL_RUNNING).sum())}
    return out


def status_reference(train_rows: pd.DataFrame, params: dict):
    """The tuned GBM refitted on the status label (TERMINATED vs COMPLETED) of the train trials that ended."""
    ended = train_rows[train_rows["overall_status"].isin(ENDED)]
    y = ended["overall_status"].eq("TERMINATED").astype(int).to_numpy()
    model = train.build_gbm_pipeline().set_params(**{f"clf__{k}": v for k, v in params.items()})
    return model.fit(to_model_frame(ended), y), {"n_trials": int(len(ended)), "terminated_rate": float(y.mean())}


def run(data_path: Path = DATA_PATH, out_dir: Path = None, phases: str = "all",
        app_model_path: Path = APP_MODEL_PATH) -> dict:
    """Tune, fit and save the horizon models, then score valid and test once. Returns the metrics dict."""
    if not Path(data_path).exists():
        raise SystemExit(f"{data_path} is missing: run `python src/fetch_data.py --restart`, then "
                         "`python src/build_dataset.py`.")
    df = pd.read_parquet(data_path)
    if phases == "1-3":
        df = df[df["phase"].isin(PHASES_1_3)]
    cutoff = check_dataset(df)
    out_dir = Path(out_dir or output_dir(phases))
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, train_rows = part(df, "train")
    print(f"Horizon label ({df.attrs.get('label', '')}), phases: {phases}")
    for split in horizon.SPLITS:
        rows = df[df["split"] == split]
        print(f"  {split}: {len(rows):,} trials, {rows['label'].mean():.2%} terminated within "
              f"{horizon.HORIZON_YEARS} years")

    cv = train.make_cv()
    models, info = {}, {}
    for name in train.BUILDERS:
        t0 = time.time()
        search = train.tune(name, X_train, y_train, cv)
        cv_info = train.cv_summary(search)
        _, threshold = train.out_of_fold(search.best_estimator_, X_train, y_train, cv)
        saved = train.SAVED_NAMES[name]
        joblib.dump(search.best_estimator_, out_dir / f"{saved}.joblib", compress=3)
        models[saved] = search.best_estimator_
        info[saved] = {"cv": cv_info, "threshold": threshold, **train.model_details(search.best_estimator_)}
        print(f"{saved}: CV ROC-AUC {cv_info['best_roc_auc_mean']:.4f}, CV PR-AUC "
              f"{cv_info['best_pr_auc_mean']:.4f} ({time.time() - t0:.0f}s)")

    t0 = time.time()
    calibrated = train.build_calibrated(models["gbm_uncalibrated"])
    oof, threshold = train.out_of_fold(calibrated, X_train, y_train, cv)
    calibrated.fit(X_train, y_train)
    joblib.dump(calibrated, out_dir / "gbm.joblib", compress=3)
    np.save(out_dir / "score_reference.npy", np.sort(oof))
    models["gbm"] = calibrated
    info["gbm"] = {"cv": {**info["gbm_uncalibrated"]["cv"], "note": "tuned on gbm_uncalibrated"},
                   "threshold": threshold, **train.model_details(calibrated)}
    print(f"gbm (calibrated): OOF Brier {threshold['oof_brier']:.4f} ({time.time() - t0:.0f}s)")

    # References, scored against the same horizon labels: same years with the status label, and the app model.
    params = info["gbm_uncalibrated"]["cv"]["best_params"]
    references, reference_info = {}, {}
    references["status_gbm_2008_2018"], reference_info["status_gbm_2008_2018"] = status_reference(train_rows, params)
    reference_info["status_gbm_2008_2018"]["what"] = ("gbm_uncalibrated pipeline and hyperparameters, refitted on "
                                                      "TERMINATED vs COMPLETED of the train trials that ended; "
                                                      "uncalibrated, so compare ranking only")
    if Path(app_model_path).exists():
        references["app_model"] = load_model(app_model_path)
        reference_info["app_model"] = {"what": "models/gbm.joblib as deployed: status label, start 2008-2016, "
                                               "calibrated on that label"}

    report = {
        "experiment": "3-year horizon label (src/horizon.py); not the app model",
        "trained_on": date.today().isoformat(), "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(), "phases": phases,
        "label": df.attrs.get("label"), "data_snapshot": df.attrs.get("snapshot"),
        "split": split_summary(df, cutoff),
        "features": {"all": FEATURE_COLUMNS, "categorical": CATEGORICAL_FEATURES,
                     "binary": BINARY_FEATURES, "numeric": NUMERIC_FEATURES},
        "models": info, "references": reference_info,
    }
    for split in ("valid", "test"):
        X, y, rows = part(df, split)
        years = rows["start_date"].dt.year.to_numpy()
        scores = {}
        for name, model in {**models, **references}.items():
            proba = model.predict_proba(X)[:, 1]
            if name in models:
                entry = evaluate.classification_metrics(y, proba, info[name]["threshold"]["value"])
            else:
                entry = evaluate.ranking_metrics(y, proba)
            entry["per_start_year"] = per_start_year(y, proba, years)
            if name == "gbm":
                entry["calibration_bins"] = evaluate.calibration_bins(y, proba)
            scores[name] = entry
        scores["majority_baseline"] = evaluate.majority_baseline(y)
        report[split] = scores
    report["drift_train_vs_test"] = drift(train_rows, df[df["split"] == "test"])
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print_summary(report)
    print(f"\nSaved models and metrics.json to {out_dir}")
    return report


def print_summary(report: dict) -> None:
    """Valid (pre-COVID) and test (post-COVID) ranking per model, per start year, and the biggest drifts."""
    names = {"gbm": "GBM, horizon label, calibrated", "gbm_uncalibrated": "GBM, horizon label, uncalibrated",
             "baseline_lr": "Logistic regression, horizon label",
             "status_gbm_2008_2018": "GBM, status label, same years (reference)",
             "app_model": "Current app model, status label 2008-2016 (reference)"}
    rows = []
    for name, label in names.items():
        if name in report["test"]:
            valid, test = report["valid"][name], report["test"][name]
            rows.append({"model": label, "valid ROC-AUC": valid["roc_auc"], "valid PR-AUC": valid["pr_auc"],
                         "test ROC-AUC": test["roc_auc"], "test PR-AUC": test["pr_auc"],
                         "test Brier": test["brier"], "test mean pred": test["mean_predicted"]})
    base = report["test"]["majority_baseline"]
    rows.append({"model": "Majority (never stopped)", "valid PR-AUC": report["valid"]["majority_baseline"]["pr_auc"],
                 "test ROC-AUC": 0.5, "test PR-AUC": base["pr_auc"], "test Brier": base["brier_constant_rate"],
                 "test mean pred": base["terminated_rate"]})
    split = report["split"]
    print(f"\n=== Valid = start 2019 (before COVID), test = start {split['test']['start_from'][:7]} to "
          f"{split['test']['start_to'][:7]} (after) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(pd.DataFrame(rows).set_index("model").round(4).to_string(na_rep="-"))
        years = report["valid"]["gbm"]["per_start_year"] + report["test"]["gbm"]["per_start_year"]
        print("\nGBM (horizon label) per start year:")
        print(pd.DataFrame(years).set_index("start_year").round(4).to_string(na_rep="-"))
        print("\nFeatures that moved most between train and test years:")
        print(pd.DataFrame(report["drift_train_vs_test"][:DRIFT_TOP]).round(3).to_string(index=False))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phases", choices=["all", "1-3"], default="all",
                        help="all phases (default) or only Phase 1, 2, 3, 1/2 and 2/3")
    run(phases=parser.parse_args().phases)


if __name__ == "__main__":
    main()

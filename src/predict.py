"""Score trials with a saved model. This is the module the Streamlit app imports.

Typical use in the app::

    from features import build_features
    from predict import percentile, score

    risk = score(build_features(study_json))   # calibrated P(TERMINATED), 0-1
    rank = percentile(risk)                     # "riskier than rank% of trials"
    drivers = explain(build_features(study_json))  # what pushes it up / down

``models/gbm.joblib`` is the gradient boosting pipeline wrapped in isotonic
calibration (``CalibratedClassifierCV``, fitted on the train split), so
``score`` returns a calibrated probability: across many trials scored 0.20,
about 20% were terminated. ``percentile`` compares a score with the
out-of-fold calibrated scores of the 2008-2016 training trials.

``explain`` lists the factors that push one trial's risk up or down, measured on
the uncalibrated GBM (``models/gbm_uncalibrated.joblib``) in log-odds. SHAP's
TreeExplainer was tried first and rejected: it does not follow the native
categorical splits of ``HistGradientBoostingClassifier`` (its values missed the
model's log-odds by up to 2.2 on real trials), so ``explain`` sets one feature at
a time to the typical training trial and measures the change instead.

This module deliberately imports only pandas, numpy, joblib, ``features`` and
``labels`` so the app does not pull in training or plotting dependencies.
"""

import sys
from functools import lru_cache
from pathlib import Path

import json

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES,  # noqa: E402
                      FEATURE_COLUMNS, NUMERIC_FEATURES)
from labels import feature_name, feature_value  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "models/gbm.joblib"
# The calibrated model's isotonic step is flat over wide ranges, so small pushes
# vanish in it; the GBM it wraps gives smooth log-odds for explanations.
EXPLAIN_MODEL_PATH = "models/gbm_uncalibrated.joblib"
SCORE_REFERENCE_PATH = "models/score_reference.npy"
EXPLAIN_REFERENCE_PATH = "models/explain_reference.json"
# Columns that only make sense together: moving one to the typical trial moves
# the others too, and they are reported as one factor under the first name.
# (No maximum age is stored as 120 plus has_max_age = 0; either alone would
# describe a design that cannot exist.)
LINKED_FEATURES = {"max_age_years": ["max_age_years", "has_max_age"]}


def _resolve(path) -> Path:
    """Resolve a relative path against the repository root, not the cwd."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def model_features(model) -> list:
    """The input columns a fitted model was trained on, in order.

    A model saved before a feature was added keeps working: it is simply fed
    its own, shorter list. Falls back to ``FEATURE_COLUMNS`` for estimators
    that do not record their input names.
    """
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(FEATURE_COLUMNS)


def to_model_frame(records, columns=None) -> pd.DataFrame:
    """Turn feature dicts (or a DataFrame) into the exact frame the models expect.

    Training and the app both go through this function, so a trial scored live
    gets the same dtypes as the rows the model was fitted on: categoricals as
    Python objects with ``np.nan`` for missing, flags and numbers as float64.
    Only ``FEATURE_COLUMNS`` are kept; meta and display keys (``nct_id``,
    ``label``, dates, titles, ``split``) are dropped here, which is what keeps
    them out of the model.

    Args:
        records: a list of ``build_features`` dicts, or a DataFrame holding at
            least the ``FEATURE_COLUMNS``.
        columns: the columns to return, in order (``model_features(model)``);
            defaults to ``FEATURE_COLUMNS``.

    Returns:
        DataFrame with exactly ``columns``, in that order.

    Raises:
        ValueError: if any feature key is absent. A silently missing column
            would be imputed and give a plausible but wrong score.
    """
    columns = list(FEATURE_COLUMNS if columns is None else columns)
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    out = {}
    for col in columns:
        if col in CATEGORICAL_FEATURES:
            values = frame[col].astype(object)
            # None, pd.NA and NaN all become np.nan, the one marker sklearn's
            # imputers and encoders treat as missing in object columns.
            out[col] = values.where(values.notna(), np.nan)
        elif col in BINARY_FEATURES or col in NUMERIC_FEATURES:
            # Nullable Int8/Int32 from the parquet and None from a live dict
            # both end up as float64 with NaN.
            out[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
        else:
            raise ValueError(f"Unknown feature column: {col}")
    return pd.DataFrame(out, index=frame.index)[columns]


@lru_cache(maxsize=None)
def _load_cached(resolved_path: str):
    """Load a joblib file once per process; Streamlit reruns then reuse it."""
    return joblib.load(resolved_path)


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a saved model from ``models/``.

    Args:
        path: path to a ``.joblib`` file. A relative path is resolved against
            the repository root, so the app works whatever directory Streamlit
            is started from.

    Returns:
        The fitted estimator (a Pipeline, or the calibrated wrapper around one);
        call ``predict_proba`` on the output of ``to_model_frame``.
    """
    return _load_cached(str(_resolve(path)))


def score(features: dict, model=None) -> float:
    """Calibrated predicted termination risk for one trial.

    Args:
        features: the dict returned by ``build_features`` for one API study
            record. Extra keys (nct_id, start_date, titles, ...) are ignored.
        model: an already loaded model; defaults to ``models/gbm.joblib``
            (calibrated). Passing another model returns that model's own
            probability, which is uncalibrated for the other saved models.

    Returns:
        Probability of TERMINATED in [0, 1].
    """
    return float(score_frame([features], model)[0])


def score_frame(records, model=None) -> np.ndarray:
    """Calibrated predicted termination risk for many trials in one model call.

    Same numbers as ``score`` row by row; use it for whole tables (the game,
    the How the model works page), where thousands of single calls would take minutes.
    Each model is fed only the columns it was fitted on, so a table holding
    extra feature columns (e.g. the v2 condition flags) scores correctly.

    Args:
        records: a DataFrame with at least the model's feature columns, or a
            list of ``build_features`` dicts.
        model: an already loaded model; defaults to ``models/gbm.joblib``.

    Returns:
        Array of P(TERMINATED), one per row, in input order.
    """
    model = model if model is not None else load_model()
    return model.predict_proba(to_model_frame(records, model_features(model)))[:, 1]


@lru_cache(maxsize=None)
def _score_reference(resolved_path: str) -> np.ndarray:
    """Sorted reference scores, loaded once per process."""
    return np.sort(np.load(resolved_path))


def percentile(prob: float, reference_path=SCORE_REFERENCE_PATH) -> float:
    """Share of historical training trials with a lower risk score, in percent.

    The reference holds the out-of-fold calibrated scores of the 114,600
    train-split trials (start 2008-2016): each one was scored by a model that
    had not seen it, so the reference is as honest as a live score. Use it as
    ``percentile(score(features))``; only scores from ``models/gbm.joblib`` are
    on the same scale.

    Args:
        prob: a calibrated probability from ``score``.
        reference_path: the ``.npy`` file written by ``src/train.py``.

    Returns:
        A value in [0, 100]: ``percentile(p) == 73.0`` means "riskier than 73%
        of historical trials". Ties count as not lower, so a trial is never
        called riskier than trials with exactly its own score.
    """
    reference = _score_reference(str(_resolve(reference_path)))
    return float(100.0 * np.searchsorted(reference, prob, side="left") / len(reference))


def reference_size(reference_path=SCORE_REFERENCE_PATH) -> int:
    """Number of historical trials behind ``percentile`` (for captions)."""
    return int(len(_score_reference(str(_resolve(reference_path)))))


@lru_cache(maxsize=None)
def _explain_reference(resolved_path: str) -> dict:
    """The typical training trial written by ``src/train.py``, loaded once per process."""
    return json.loads(Path(resolved_path).read_text(encoding="utf-8"))["values"]


def explain(features: dict, top_n=5, model=None, reference=None) -> list:
    """The factors that push one trial's predicted risk up or down the most.

    Method (perturbation to the typical trial): for each input of the model,
    replace this trial's value with the one of a typical training trial (the
    train median for numbers and flags, the most common value for categories,
    from ``models/explain_reference.json``) and measure how much the model's
    log-odds of TERMINATED change. A factor that is already typical moves
    nothing and is left out. The model is the uncalibrated GBM, whose log-odds
    are smooth; calibration only rescales them monotonically, so the order and
    direction of the factors hold for the app's calibrated score too.

    The sizes are one-at-a-time effects, not shares: with interactions they do
    not add up to the trial's distance from the typical trial.

    Args:
        features: the dict returned by ``build_features`` (or
            ``features_from_design``) for one trial.
        top_n: how many factors to return at most.
        model: a fitted pipeline with ``decision_function``; defaults to
            ``models/gbm_uncalibrated.joblib``.
        reference: ``{column: typical value}``; defaults to the saved one.

    Returns:
        Up to ``top_n`` tuples ``(friendly_name, value, direction, size)``,
        largest first: the plain name (``labels.feature_name``), this trial's
        value in plain words (``labels.feature_value``), ``"up"`` if the value
        raises the risk compared with the typical trial or ``"down"`` if it
        lowers it, and the size of that change in log-odds (always >= 0).
    """
    model = model if model is not None else load_model(EXPLAIN_MODEL_PATH)
    if reference is None:
        reference = _explain_reference(str(_resolve(EXPLAIN_REFERENCE_PATH)))
    columns = model_features(model)
    base = to_model_frame([features], columns)

    linked = {c for group in LINKED_FEATURES.values() for c in group[1:]}
    factors, rows = [], []
    for col in columns:
        if col in linked:
            continue  # moved together with its lead column
        group = [c for c in LINKED_FEATURES.get(col, [col]) if c in columns]
        row = base.iloc[0].copy()
        for c in group:
            row[c] = reference[c]
        factors.append(col)
        rows.append(row)
    perturbed = to_model_frame(pd.DataFrame(rows).reset_index(drop=True), columns)

    log_odds = model.decision_function(pd.concat([base, perturbed], ignore_index=True))
    effects = log_odds[0] - log_odds[1:]  # > 0: this trial's value raises the risk
    order = np.argsort(-np.abs(effects), kind="stable")
    out = []
    for i in order[:top_n]:
        if abs(effects[i]) < 1e-9:
            break  # the rest equal the typical trial and push nothing
        col = factors[i]
        out.append((feature_name(col), feature_value(col, base.iloc[0][col]),
                    "up" if effects[i] > 0 else "down", float(abs(effects[i]))))
    return out

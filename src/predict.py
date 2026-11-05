"""Score trials with a saved model. This is the module the Streamlit app imports.

Typical use in the app::

    from features import build_features
    from predict import percentile, score

    risk = score(build_features(study_json))   # calibrated P(TERMINATED), 0-1
    rank = percentile(risk)                     # "riskier than rank% of trials"

``models/gbm.joblib`` is the gradient boosting pipeline wrapped in isotonic
calibration (``CalibratedClassifierCV``, fitted on the train split), so
``score`` returns a calibrated probability: across many trials scored 0.20,
about 20% were terminated. ``percentile`` compares a score with the
out-of-fold calibrated scores of the 2008-2016 training trials.

The app's pages load the model and its metrics only through ``get_model`` and
``load_metrics``: each file is read once per process, and both return None
while the file is missing so a page can show a message instead.

This module deliberately imports only pandas, numpy, joblib and ``features`` so
the app does not pull in training or plotting dependencies.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (BINARY_FEATURES, CATEGORICAL_FEATURES,  # noqa: E402
                      FEATURE_COLUMNS, NUMERIC_FEATURES)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "models/gbm.joblib"
DEFAULT_METRICS_PATH = "models/gbm_metrics.json"
SCORE_REFERENCE_PATH = "models/score_reference.npy"


def _resolve(path) -> Path:
    """Resolve a relative path against the repository root, not the cwd."""
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def to_model_frame(records) -> pd.DataFrame:
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

    Returns:
        DataFrame with exactly ``FEATURE_COLUMNS``, in that order.

    Raises:
        ValueError: if any feature key is absent. A silently missing column
            would be imputed and give a plausible but wrong score.
    """
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    columns = {}
    for col in CATEGORICAL_FEATURES:
        values = frame[col].astype(object)
        # None, pd.NA and NaN all become np.nan, the one marker sklearn's
        # imputers and encoders treat as missing in object columns.
        columns[col] = values.where(values.notna(), np.nan)
    for col in BINARY_FEATURES + NUMERIC_FEATURES:
        # Nullable Int8/Int32 from the parquet and None from a live dict both
        # end up as float64 with NaN.
        columns[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return pd.DataFrame(columns, index=frame.index)[FEATURE_COLUMNS]


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


def get_model(path=DEFAULT_MODEL_PATH):
    """The app's model, or None when it cannot be loaded, so a page can show a message instead of a risk.

    Shares ``load_model``'s cache: one load per process for every page and
    session. A failed load is not cached, so a model file that appears later is
    picked up without a restart.

    Args:
        path: as for ``load_model``.

    Returns:
        The fitted estimator, or None if the file is missing or a library it
        needs to unpickle (scikit-learn) is not installed.
    """
    try:
        return load_model(path)
    except (ImportError, FileNotFoundError):
        return None


@lru_cache(maxsize=None)
def _read_json(resolved_path: str) -> dict:
    """Parse a JSON file once per process. Raises on failure, so a missing file is never cached."""
    with open(resolved_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_metrics(path=DEFAULT_METRICS_PATH):
    """Metrics of the app model (``models/gbm_metrics.json``), read once per process.

    Every page that quotes a metric reads it here, so they all show the same
    numbers from one read of the file.

    Args:
        path: a ``<model>_metrics.json`` written by ``src/train.py`` and
            ``src/evaluate.py``; a relative path is resolved against the
            repository root.

    Returns:
        The parsed JSON, shared by every caller (treat it as read-only), or
        None if the file is missing or not valid JSON.
    """
    try:
        return _read_json(str(_resolve(path)))
    except (OSError, ValueError):
        return None


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
    model = model if model is not None else load_model()
    return float(model.predict_proba(to_model_frame([features]))[0, 1])


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

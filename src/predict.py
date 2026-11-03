"""Score trials with a saved model. This is the module the Streamlit app imports.

Typical use in the app::

    from features import build_features
    from predict import score

    risk = score(build_features(study_json))   # float in [0, 1]

``score`` returns the model's raw probability of TERMINATED. Both baseline models
are trained with ``class_weight='balanced'``, which pushes probabilities upwards
(see ``models/calibration_curve.png``). Treat the number as a relative risk score
for ranking and comparing trials, not as a calibrated chance of termination.

This module deliberately imports only pandas, numpy, joblib and ``features`` so
the app does not pull in training or plotting dependencies.
"""

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


def to_model_frame(records) -> pd.DataFrame:
    """Turn feature dicts (or a DataFrame) into the exact frame the models expect.

    Training and the app both go through this function, so a trial scored live
    gets the same dtypes as the rows the model was fitted on: categoricals as
    Python objects with ``np.nan`` for missing, flags and numbers as float64.
    Only ``FEATURE_COLUMNS`` are kept; meta keys (``nct_id``, ``label``, dates,
    ``split``) are dropped here, which is what keeps them out of the model.

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
    """Load a saved sklearn Pipeline from ``models/``.

    Args:
        path: path to a ``.joblib`` file. A relative path is resolved against
            the repository root, so the app works whatever directory Streamlit
            is started from.

    Returns:
        The fitted Pipeline (preprocessing + classifier); call
        ``predict_proba`` on the output of ``to_model_frame``.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return _load_cached(str(resolved))


def score(features: dict, model=None) -> float:
    """Predicted termination risk for one trial.

    Args:
        features: the dict returned by ``build_features`` for one API study
            record. Extra keys (nct_id, start_date, ...) are ignored.
        model: an already loaded Pipeline; defaults to ``models/gbm.joblib``.

    Returns:
        Probability of TERMINATED in [0, 1] (uncalibrated, see module docstring).
    """
    model = model if model is not None else load_model()
    return float(model.predict_proba(to_model_frame([features]))[0, 1])

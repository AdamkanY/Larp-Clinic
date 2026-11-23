# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Score trials with a saved model. This is the file the Streamlit app imports.

How the app uses it:

    features = build_features(study_json)   # from features.py
    risk = score(features)                   # probability of "stopped early", between 0 and 1
    rank = percentile(risk)                  # "riskier than rank% of past trials"
    drivers = explain(features)              # what pushes the risk up or down

models/gbm.joblib is the gradient boosting model with calibration on top. Calibrated means: of many trials that
get 0.20, about 20% were really stopped.

We tried SHAP for the explanations first. It did not work with the way our model splits on categories (its
values were off by up to 2.2 log-odds on real trials). So explain() uses a simpler idea: set one feature at a
time to the value of a typical trial and look how much the prediction moves.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# The other files of src/ are imported by their plain name, so this works from the app and from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from labels import feature_name, feature_value

ROOT = Path(__file__).resolve().parents[1]  # the project folder
DEFAULT_MODEL_PATH = "models/gbm.joblib"
# The explanations use the model before calibration. Calibration is a step function, so small changes would
# disappear in it. The order and the direction of the factors are the same for both models.
EXPLAIN_MODEL_PATH = "models/gbm_uncalibrated.joblib"
SCORE_REFERENCE_PATH = "models/score_reference.npy"
EXPLAIN_REFERENCE_PATH = "models/explain_reference.json"

# Columns that only make sense together. "No maximum age" is stored as 120 years plus has_max_age = 0, so
# explain() changes both at once and reports them as one factor under the first name.
LINKED_FEATURES = {"max_age_years": ["max_age_years", "has_max_age"]}

# Files we already loaded, so every file is read from disk only once (Streamlit reruns the script very often).
LOADED_FILES = {}


def full_path(path):
    """A path like "models/gbm.joblib" is meant from the project folder, wherever the app was started."""
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def load_model(path=DEFAULT_MODEL_PATH):
    """Load a saved model from models/ (only the first time, afterwards from memory)."""
    key = str(full_path(path))
    if key not in LOADED_FILES:
        LOADED_FILES[key] = joblib.load(key)
    return LOADED_FILES[key]


def load_score_reference(path=SCORE_REFERENCE_PATH):
    """The sorted risk scores of the past trials that percentile() compares with."""
    key = str(full_path(path))
    if key not in LOADED_FILES:
        LOADED_FILES[key] = np.sort(np.load(key))
    return LOADED_FILES[key]


def load_explain_reference(path=EXPLAIN_REFERENCE_PATH):
    """The "typical training trial" that train.py saved, as a dictionary {column: typical value}."""
    key = str(full_path(path))
    if key not in LOADED_FILES:
        text = Path(key).read_text(encoding="utf-8")
        LOADED_FILES[key] = json.loads(text)["values"]
    return LOADED_FILES[key]


def model_features(model):
    """The input columns a model was trained on, in order.

    An older model that was trained with fewer columns keeps working, because it gets its own shorter list.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return list(FEATURE_COLUMNS)
    return list(names)


def to_model_frame(records, columns=None):
    """Turn feature dictionaries (or a DataFrame) into exactly the table the models expect.

    Training and the app both use this function, so a trial scored in the app has the same data types as the
    rows the model learned from. Only the feature columns are kept. Ids, titles, dates and the label are dropped
    here, and that is what keeps them away from the model.

    A missing feature column raises an error. If we filled it in silently, the score would look fine but be wrong.
    """
    if columns is None:
        columns = FEATURE_COLUMNS
    columns = list(columns)

    # The app passes a list of dictionaries, training passes a whole DataFrame.
    if isinstance(records, pd.DataFrame):
        frame = records
    else:
        frame = pd.DataFrame(list(records))

    missing = []
    for column in columns:
        if column not in frame.columns:
            missing.append(column)
    if missing:
        raise ValueError(f"Missing feature keys: {missing}")

    out = {}
    for column in columns:
        if column in CATEGORICAL_FEATURES:
            # Categories stay text. Every kind of "missing" (None, NA, NaN) becomes np.nan, the one that
            # scikit-learn understands.
            values = frame[column].astype(object)
            out[column] = values.where(values.notna(), np.nan)
        elif column in BINARY_FEATURES or column in NUMERIC_FEATURES:
            # Flags and numbers become decimal numbers, missing values become NaN.
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        else:
            raise ValueError(f"Unknown feature column: {column}")
    return pd.DataFrame(out, index=frame.index)[columns]


def score_frame(records, model=None):
    """Predicted risk of stopping early for many trials at once (one number per row, same order).

    Use this for whole tables (the game, the model page). Calling score() thousands of times would take minutes.
    Without a model the calibrated app model is used.
    """
    if model is None:
        model = load_model()
    table = to_model_frame(records, model_features(model))
    # predict_proba gives two columns: P(completed) and P(terminated). We want the second one.
    return model.predict_proba(table)[:, 1]


def score(features, model=None):
    """Predicted risk of stopping early for ONE trial, a number between 0 and 1.

    features is the dictionary from build_features(). Extra keys such as the id or the title are ignored.
    """
    return float(score_frame([features], model)[0])


def percentile(prob, reference_path=SCORE_REFERENCE_PATH):
    """How many percent of the past trials have a LOWER risk score than prob.

    percentile(p) == 73.0 means "riskier than 73% of past trials". The past trials are the training trials that
    started 2009-2016 (95,500). Each of them was scored by a model that had not seen it, so the comparison is
    as honest as a new score. Trials with exactly the same score do not count as lower.
    """
    reference = load_score_reference(reference_path)
    lower = np.searchsorted(reference, prob, side="left")  # the scores are sorted, so this is a quick search
    return float(100.0 * lower / len(reference))


def reference_size(reference_path=SCORE_REFERENCE_PATH):
    """How many past trials percentile() compares with (for the captions in the app)."""
    return int(len(load_score_reference(reference_path)))


def explain(features, top_n=5, model=None, reference=None):
    """The factors that push the predicted risk of one trial up or down the most.

    Idea: take the trial and replace ONE value by the value of a typical training trial (the median for numbers
    and flags, the most common value for categories). Then look how much the model's prediction changes. A value
    that is already typical changes nothing and is left out. The changes are measured in log-odds on the model
    before calibration. They are one-at-a-time effects, so they do not add up to the full difference.

    Returns a list of at most top_n tuples, biggest effect first:
    (plain name, this trial's value in words, "up" or "down", size of the effect)
    """
    if model is None:
        model = load_model(EXPLAIN_MODEL_PATH)
    if reference is None:
        reference = load_explain_reference()
    columns = model_features(model)
    trial = to_model_frame([features], columns)

    # Columns that are changed together with another column do not get their own turn.
    partner_columns = []
    for group in LINKED_FEATURES.values():
        partner_columns += group[1:]

    # One copy of the trial per factor, with that factor set to the typical value.
    factors = []
    copies = []
    for column in columns:
        if column in partner_columns:
            continue
        changed = trial.iloc[0].copy()
        for linked_column in LINKED_FEATURES.get(column, [column]):
            if linked_column in columns:
                changed[linked_column] = reference[linked_column]
        factors.append(column)
        copies.append(changed)
    copies = to_model_frame(pd.DataFrame(copies).reset_index(drop=True), columns)

    # Score the trial and all copies in one call. Row 0 is the real trial.
    log_odds = model.decision_function(pd.concat([trial, copies], ignore_index=True))
    effects = log_odds[0] - log_odds[1:]  # above 0: this trial's value makes the risk higher

    # Biggest effects first, whatever the direction.
    order = np.argsort(-np.abs(effects), kind="stable")
    result = []
    for i in order[:top_n]:
        effect = effects[i]
        if abs(effect) < 1e-9:
            break  # everything after this equals the typical trial
        column = factors[i]
        if effect > 0:
            direction = "up"
        else:
            direction = "down"
        value_in_words = feature_value(column, trial.iloc[0][column])
        result.append((feature_name(column), value_in_words, direction, float(abs(effect))))
    return result

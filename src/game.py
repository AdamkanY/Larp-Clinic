"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

The logic of the game "guess the stopped trial": draw pairs of trials, let the model pick, count how often
the model is right.

There is no Streamlit code in this file, the page app/pages/5_Game.py only calls these functions. The game uses
only the test split (trials that started 2017-2019). The model has never seen these trials, so its picks are as
honest as a score for a new trial.
"""
import sys  # sys.path: lets us import the other files of src/ by their plain name
from pathlib import Path  # file paths that work on every computer

import numpy as np  # random numbers and arrays
import pandas as pd  # tables (DataFrame)

# The other files of src/ are imported by their plain name, so this works from the app and from the terminal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DISPLAY_COLS, FEATURE_COLUMNS  # the column lists of the data
from predict import score_frame  # scores many trials with the saved model

ROOT = Path(__file__).resolve().parents[1]  # the project folder
TRIALS_PATH = ROOT / "data" / "processed" / "trials.parquet"
# How many pairs the page draws for the line "the model gets it right x% of the time".
ACCURACY_PAIRS = 2000


def load_test_trials(path=TRIALS_PATH):
    """Load the trials the game can use: the test split, only trials with a phase, numbered from 0.

    We read only the columns the game needs (what is shown, the model inputs, the label and the split), so the
    page stays fast.

    Input: path is the location of trials.parquet (a Path or text); the default is the project's own file.

    Output: a DataFrame with one row per usable trial and a new index 0, 1, 2, ...
    """
    columns = DISPLAY_COLS + FEATURE_COLUMNS + ["label", "split"]
    trials = pd.read_parquet(path, columns=columns)
    # A trial without a phase cannot be used, because the two trials of a pair must have the same phase.
    trials = trials[(trials["split"] == "test") & trials["phase_label"].notna()]
    return trials.reset_index(drop=True)


def sample_pairs(trials, n, seed=None):
    """Draw n pairs: one TERMINATED and one COMPLETED trial with the same phase_label.

    Both trials have the same phase on purpose. Otherwise a player could win just by knowing that early phases
    are stopped more often, and would not have to look at the design at all.

    Input: trials is the table from load_test_trials(); n is how many pairs to draw (a whole number); seed is a
    number (or random generator) so the same seed gives the same pairs again, or None for chance.

    Output: a DataFrame with the columns "terminated" and "completed", one pair per row. The values are row
    numbers of trials. If there are fewer than n terminated trials that can be paired, fewer pairs come back.
    """
    rng = np.random.default_rng(seed)
    terminated = trials[trials["label"] == 1]
    completed = trials[trials["label"] == 0]

    # For every phase: the completed trials in a random order. The partners are taken from these lists.
    completed_pool = {}
    for phase, group in completed.groupby("phase_label"):
        completed_pool[phase] = list(rng.permutation(group.index.to_numpy()))

    # Draw the terminated trials, each one only once. A phase without completed trials cannot be used.
    usable = terminated[terminated["phase_label"].isin(list(completed_pool))]
    n_pairs = min(n, len(usable))
    chosen = rng.choice(usable.index.to_numpy(), size=n_pairs, replace=False)

    # Give every terminated trial the next completed trial from the list of its phase, so no completed trial
    # comes twice. Only if a phase has fewer completed than terminated trials, its list starts again from the
    # beginning (this is what the "%" does).
    taken = {}
    for phase in completed_pool:
        taken[phase] = 0
    partners = []
    for row_number in chosen:
        phase = trials.at[row_number, "phase_label"]
        pool = completed_pool[phase]
        partners.append(pool[taken[phase] % len(pool)])
        taken[phase] += 1
    return pd.DataFrame({"terminated": chosen, "completed": partners})


def score_rows(trials, model=None):
    """Score the risk of stopping early for every row of trials (without a model: the calibrated app model).

    This only calls score_frame() from predict.py, which gives the model nothing but its own input columns.
    The game and its tests use this name.

    Input: trials is a DataFrame with the feature columns; model is a saved model, or None for the app model.

    Output: a numpy array with one risk (0 to 1) per row, in the same order.
    """
    return score_frame(trials, model)


def model_pick(terminated_score, completed_score, rng=None):
    """Decide which trial of a pair the model thinks was stopped: "terminated" (model is right) or "completed".

    The trial with the higher risk score wins. With exactly the same score we flip a coin. This is rare, but it
    happens, because the calibrated scores come in steps.

    Input: terminated_score and completed_score are the risk scores (numbers) of the two trials; rng is a seed or
    a random generator for the coin flip (or None).

    Output: the text "terminated" if the model picks the really terminated trial, otherwise "completed".
    """
    if terminated_score != completed_score:
        if terminated_score > completed_score:
            return "terminated"
        return "completed"

    # The coin flip. rng can be a seed or a random generator, so a whole game can be repeated.
    if np.random.default_rng(rng).random() < 0.5:
        return "terminated"
    return "completed"


def pairwise_accuracy(trials, pairs, model=None):
    """Work out in which share of the pairs the model gives the terminated trial the higher score.

    This is the number the player plays against. A tie counts as half a point. (It is the same as the ROC-AUC,
    measured on pairs with the same phase.)

    Input: trials is the table from load_test_trials(); pairs is the table from sample_pairs(); model is a saved
    model, or None for the app model.

    Output: a float between 0 and 1.
    """
    # Score every trial once, then pick out the two scores of every pair.
    scores = score_rows(trials, model)
    terminated_scores = scores[pairs["terminated"].to_numpy()]
    completed_scores = scores[pairs["completed"].to_numpy()]

    # 1 point if the model is right, half a point for a tie, 0 if it is wrong.
    points = (terminated_scores > completed_scores) + 0.5 * (terminated_scores == completed_scores)
    return float(points.mean())


def new_game(trials, rounds, model=None, seed=None):
    """Decide everything for one game in advance: the pairs, which trial is shown left, and the model's picks.

    Input: trials is the table from load_test_trials(); rounds is how many pairs the game has; model is a saved
    model, or None for the app model; seed makes the game repeatable (or None).

    Output: a list with one dictionary per round: "terminated" and "completed" (row numbers in trials), "left"
    (which of the two is shown as trial A) and "model_pick" ("terminated" or "completed").
    """
    # One random generator for everything, so the same seed gives exactly the same game.
    rng = np.random.default_rng(seed)
    pairs = sample_pairs(trials, rounds, seed=rng)

    # Score both trials of every pair.
    terminated_scores = score_rows(trials.loc[pairs["terminated"]], model)
    completed_scores = score_rows(trials.loc[pairs["completed"]], model)

    # Build one dictionary per round.
    game = []
    for i in range(len(pairs)):
        # Which trial is on the left is random. Otherwise the player would learn that the left one is always
        # the stopped trial.
        if rng.random() < 0.5:
            left = "terminated"
        else:
            left = "completed"
        game.append({
            "terminated": int(pairs["terminated"].iloc[i]),
            "completed": int(pairs["completed"].iloc[i]),
            "left": left,
            "model_pick": model_pick(terminated_scores[i], completed_scores[i], rng),
        })
    return game

"""Logic of the "guess the stopped trial" game: pair sampling, model picks and the model's pairwise accuracy.

Pure pandas/numpy plus the saved model: no Streamlit. Only the test split (start 2017-2019) is used, so the model has
never seen these trials and its picks are as honest as a live score.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import DISPLAY_COLS, FEATURE_COLUMNS  # noqa: E402
from predict import score_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRIALS_PATH = ROOT / "data" / "processed" / "trials.parquet"
ACCURACY_PAIRS = 2000  # pairs in the sample behind the "model gets it right x% of the time" line


def load_test_trials(path=TRIALS_PATH) -> pd.DataFrame:
    """Test-split trials that can be paired: label 0/1 and a phase label, with a fresh 0..n-1 index.

    Only the display columns, the model features, the label and the split are read, to keep the page light.
    """
    columns = list(dict.fromkeys(DISPLAY_COLS + FEATURE_COLUMNS + ["label", "split"]))
    trials = pd.read_parquet(path, columns=columns)
    trials = trials[(trials["split"] == "test") & trials["phase_label"].notna()]
    return trials.reset_index(drop=True)


def sample_pairs(trials: pd.DataFrame, n: int, seed=None) -> pd.DataFrame:
    """Draw `n` pairs of one TERMINATED and one COMPLETED trial that share the same `phase_label`.

    The phase is matched so the user cannot win by knowing that, say, early-phase trials stop more often; the
    comparison has to come from the design. Terminated trials are drawn without replacement and the completed ones
    are taken from a shuffled pool per phase, so no trial appears twice unless a phase runs out of completed trials.

    Args:
        trials: output of `load_test_trials` (needs `label` 0/1 and `phase_label`, index 0..n-1).
        n: number of pairs wanted; fewer come back if there are not enough terminated trials.
        seed: int or numpy Generator for reproducible draws.

    Returns:
        DataFrame with columns `terminated` and `completed`: row positions in `trials`, one pair per row.
    """
    rng = np.random.default_rng(seed)
    terminated = trials[trials["label"] == 1]
    completed = trials[trials["label"] == 0]
    completed_pool = {phase: list(rng.permutation(group.index.to_numpy()))
                      for phase, group in completed.groupby("phase_label")}
    usable = terminated[terminated["phase_label"].isin(completed_pool)]
    chosen = rng.choice(usable.index.to_numpy(), size=min(n, len(usable)), replace=False)

    taken = {phase: 0 for phase in completed_pool}
    partners = []
    for idx in chosen:
        phase = trials.at[idx, "phase_label"]
        pool = completed_pool[phase]
        partners.append(pool[taken[phase] % len(pool)])  # wraps around only if a phase has fewer completed trials
        taken[phase] += 1
    return pd.DataFrame({"terminated": chosen, "completed": partners})


def score_rows(trials: pd.DataFrame, model=None) -> np.ndarray:
    """Calibrated P(TERMINATED) for every row: `predict.score_frame`, which feeds the model only its own inputs.

    Kept as a name of its own so the game's tests and callers read naturally; the scoring rules live in predict.
    """
    return score_frame(trials, model)


def model_pick(terminated_score: float, completed_score: float, rng=None) -> str:
    """Which trial the model says was terminated: "terminated" or "completed" (the true outcome of the pick).

    The higher risk score wins. An exact tie (rare, the calibration is a step function) is a coin flip.
    """
    if terminated_score != completed_score:
        return "terminated" if terminated_score > completed_score else "completed"
    return "terminated" if np.random.default_rng(rng).random() < 0.5 else "completed"


def pairwise_accuracy(trials: pd.DataFrame, pairs: pd.DataFrame, model=None) -> float:
    """Share of pairs in which the model gives the terminated trial the higher score; a tie counts as half.

    This is the number a user is playing against. It is the ROC-AUC on phase-matched pairs.
    """
    scores = score_rows(trials, model)
    term = scores[pairs["terminated"].to_numpy()]
    comp = scores[pairs["completed"].to_numpy()]
    return float(((term > comp) + 0.5 * (term == comp)).mean())


def new_game(trials: pd.DataFrame, rounds: int, model=None, seed=None) -> list:
    """Everything a game needs, decided up front: `rounds` pairs with display order and the model's pick.

    Returns a list of dicts with `terminated` and `completed` (row positions in `trials`), `left` (which outcome is
    shown as trial A) and `model_pick` ("terminated" or "completed").
    """
    rng = np.random.default_rng(seed)
    pairs = sample_pairs(trials, rounds, seed=rng)
    term_scores = score_rows(trials.loc[pairs["terminated"]], model)
    comp_scores = score_rows(trials.loc[pairs["completed"]], model)
    game = []
    for term, comp, term_score, comp_score in zip(pairs["terminated"], pairs["completed"], term_scores, comp_scores):
        game.append({
            "terminated": int(term), "completed": int(comp),
            "left": "terminated" if rng.random() < 0.5 else "completed",
            "model_pick": model_pick(term_score, comp_score, rng),
        })
    return game

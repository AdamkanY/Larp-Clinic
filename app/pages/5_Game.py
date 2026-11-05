"""Game: two past trials of the same phase, one stopped early. Guess which, then see whether the model agrees."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Pages run with app/ on the path, so add the repo root to import src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # app/, for ui
from src import game  # noqa: E402
from src.labels import label  # noqa: E402
from ui import chip, esc, page_header, require_router  # noqa: E402

require_router("pages/5_Game.py")

ROUNDS = 10
SIDES = ("A", "B")
OUTCOME_CHIP = {"terminated": ("Stopped early", "red"), "completed": ("Completed", "green")}


@st.cache_resource(show_spinner=False)
def get_trials():
    """Test-split trials, read once per server (a resource, so reruns do not copy the table); None if the file is missing."""
    try:
        return game.load_test_trials()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Checking the model on 2,000 pairs ...")
def model_accuracy():
    """The model's pairwise accuracy on a fixed sample of 2,000 test pairs, computed once per server."""
    trials = get_trials()
    pairs = game.sample_pairs(trials, game.ACCURACY_PAIRS, seed=0)
    return game.pairwise_accuracy(trials, pairs)


def flag(value):
    """True for a 1 in a 0/1 column; a missing value counts as no."""
    return bool(pd.notna(value) and value == 1)


def age_range(row):
    """Eligible ages in words, for example '18 to 65 years' or '18 years and older'."""
    low, high = row["min_age_years"], row["max_age_years"]
    has_low, has_high = pd.notna(low), flag(row["has_max_age"]) and pd.notna(high)
    if has_low and has_high:
        return f"{low:g} to {high:g} years"
    if has_low:
        return f"{low:g} years and older"
    return f"Up to {high:g} years" if has_high else "Not stated"


def card_html(row, name, outcome=None):
    """HTML of one trial card. Only design facts; the outcome chip appears once `outcome` is given (after the guess)."""
    facts = [
        ("Conditions", row["conditions"] if pd.notna(row["conditions"]) and row["conditions"] else "Not stated"),
        ("Sponsor", row["lead_sponsor_name"] if pd.notna(row["lead_sponsor_name"]) else "Not stated"),
        ("Blinding", label("masking", row["masking"] if pd.notna(row["masking"]) else None)),
        ("Allocation", label("allocation", row["allocation"] if pd.notna(row["allocation"]) else None)),
        ("Number of arms", f"{int(row['n_arms'])}" if pd.notna(row["n_arms"]) and row["n_arms"] else "Not stated"),
        ("Healthy volunteers", "Accepted" if flag(row["healthy_volunteers"]) else "Not accepted"),
        ("Eligible ages", age_range(row)),
    ]
    chips = chip(f"Trial {name}") + chip(label("phase", row["phase_label"]))
    if outcome:
        chips += chip(*OUTCOME_CHIP[outcome])
    lines = "".join(f'<div class="card-sub"><b>{esc(k)}:</b> {esc(v)}</div>' for k, v in facts)
    return f'{chips}<div class="card-title">{esc(row["brief_title"])}</div>{lines}'


def start_game():
    """Draw 10 new rounds and reset the score."""
    st.session_state.game_rounds = game.new_game(get_trials(), ROUNDS)
    st.session_state.game_round = 0
    st.session_state.game_picks = []  # the user's pick per finished round: "terminated" or "completed"
    st.session_state.game_revealed = False


def pick(outcome):
    """Store the user's guess (the outcome of the trial they think was stopped) and show the answer."""
    st.session_state.game_picks.append(outcome)
    st.session_state.game_revealed = True


def next_round():
    """Move on to the next round."""
    st.session_state.game_round += 1
    st.session_state.game_revealed = False


def scores():
    """(user points, model points) over the finished rounds."""
    rounds, picks = st.session_state.game_rounds, st.session_state.game_picks
    return (sum(p == "terminated" for p in picks),
            sum(r["model_pick"] == "terminated" for r in rounds[:len(picks)]))


page_header("Guess the stopped trial",
            "Two past trials from the same phase. One was stopped early, one was completed. Which was which?")

trials = get_trials()
if trials is None:
    st.error("The trial data is missing. Run `python src/fetch_data.py` and `python src/build_dataset.py`, then reload.")
    st.stop()
try:
    accuracy = model_accuracy()
except FileNotFoundError:
    st.error("The trained model is missing (`models/gbm.joblib`). Run `python src/train.py`, then reload.")
    st.stop()

if "game_rounds" not in st.session_state:
    start_game()
rounds = st.session_state.game_rounds
index = st.session_state.game_round
you, model_points = scores()

st.markdown(f"On 2,000 random pairs the model picked the stopped trial **{accuracy:.0%}** of the time. "
            "Guessing gets 50%. These trials started in 2017-2019, after the period the model learned from.")

if index >= len(rounds):  # ---- final scoreboard
    st.markdown("### Final score")
    left, middle, right = st.columns(3)
    left.metric("You", f"{you} / {len(rounds)}")
    middle.metric("Model", f"{model_points} / {len(rounds)}")
    right.metric("Model on 2,000 pairs", f"{accuracy:.0%}")
    if you > model_points:
        st.markdown("**You beat the model.**")
    elif you < model_points:
        st.markdown("**The model wins this time.**")
    else:
        st.markdown("**A draw.**")
    table = pd.DataFrame({
        "Stopped early": [trials.at[r["terminated"], "brief_title"] for r in rounds],
        "You": ["Right" if p == "terminated" else "Wrong" for p in st.session_state.game_picks],
        "Model": ["Right" if r["model_pick"] == "terminated" else "Wrong" for r in rounds],
    }, index=range(1, len(rounds) + 1))
    table.index.name = "Round"
    st.table(table)
    st.button("Play again", type="primary", on_click=start_game, icon=":material/replay:")
    st.stop()

# ---- one round
current = rounds[index]
revealed = st.session_state.game_revealed
order = [current["left"], "completed" if current["left"] == "terminated" else "terminated"]
st.markdown(f"**Round {index + 1} of {len(rounds)}** &nbsp;·&nbsp; You {you} &nbsp;·&nbsp; Model {model_points}",
            unsafe_allow_html=True)

columns = st.columns(2)
for column, name, outcome in zip(columns, SIDES, order):
    row = trials.loc[current[outcome]]
    with column, st.container(key=f"card_game_{name}"):
        st.markdown(card_html(row, name, outcome if revealed else None), unsafe_allow_html=True)
        if revealed:
            st.link_button("Open on ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{row['nct_id']}",
                           icon=":material/open_in_new:")
        else:
            st.button(f"Trial {name} was stopped early", key=f"pick_{index}_{name}", on_click=pick, args=(outcome,),
                      type="primary", use_container_width=True)

if revealed:
    truth = SIDES[order.index("terminated")]
    user_side = SIDES[order.index(st.session_state.game_picks[-1])]
    model_side = SIDES[order.index(current["model_pick"])]
    st.markdown(f"**Trial {truth} was stopped early.** You picked {user_side}: "
                f"{'right' if user_side == truth else 'wrong'}. The model picked {model_side}: "
                f"{'right' if model_side == truth else 'wrong'}.")
    last = index + 1 >= len(rounds)
    st.button("See the final score" if last else "Next round", type="primary", on_click=next_round,
              icon=":material/arrow_forward:")

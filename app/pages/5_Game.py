"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Game: two past trials of the same phase, one of them was stopped early. Guess which one, then see whether the
model picked the same.

The rules of the game (drawing the pairs, the model's pick, the accuracy) are in src/game.py. This page only shows
the cards and keeps the score in st.session_state.

Outline of the page code at the bottom of this file:
    1. Header and loading
    2. Game state and score
    3. After the last round: the final score
    4. One round: the two cards
    5. After the guess: the answer
"""

import sys  # changes the import path
from pathlib import Path  # builds file paths that work on every computer

import pandas as pd  # tables (DataFrame) and missing-value checks
import streamlit as st  # the web app framework

# Streamlit runs a page as its own script, so we add the project folder (for src/) and app/ (for ui) by hand.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import game  # the rules of the game: drawing pairs, the model's pick, the accuracy
from src.labels import label  # plain-English names for raw codes
from ui import chip, esc, page_header, require_router  # shared look: chips, safe html, header, router

# Send the page through Home.py if it was opened directly.
require_router("pages/5_Game.py")

# One game has ten rounds. The two cards are called A and B.
ROUNDS = 10
SIDES = ["A", "B"]
# The chip that shows the real outcome after the guess: text and colour.
OUTCOME_CHIP = {"terminated": ("Stopped early", "red"), "completed": ("Completed", "green")}


# ---------------------------------------------------------------------------------------------------------------
# Loading (cached, so the table is read only once)
# ---------------------------------------------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_trials():
    """Load the test trials (start 2017-2019), read once for the whole server.

    Input: none.

    Output: a DataFrame with one row per test trial, or None if the data file is missing.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # A missing data file means no game; the page then shows a message.
    try:
        return game.load_test_trials()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Checking the model on 2,000 pairs ...")
def model_accuracy():
    """Measure how often the model picks the stopped trial, on 2,000 pairs.

    The seed is fixed (0), so every visitor sees the same number.

    Input: none.

    Output: the share of pairs the model got right (number between 0 and 1).
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Draw the pairs with a fixed seed, then let the model pick in each.
    trials = get_trials()
    pairs = game.sample_pairs(trials, game.ACCURACY_PAIRS, seed=0)
    return game.pairwise_accuracy(trials, pairs)


# ---------------------------------------------------------------------------------------------------------------
# The trial cards
# ---------------------------------------------------------------------------------------------------------------

def flag(value):
    """Check whether a 0/1 column holds a 1. A missing value counts as no.

    Input: value is one cell of a 0/1 column (number or missing).

    Output: True if the value is 1, otherwise False.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    return bool(pd.notna(value) and value == 1)


def age_range(row):
    """Write the eligible ages in words, for example '18 to 65 years' or '18 years and older'.

    Input: row is one trial (a row of the trials table) with min_age_years, max_age_years and has_max_age.

    Output: the age range as text, or "Not stated".
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    low = row["min_age_years"]
    high = row["max_age_years"]
    has_low = pd.notna(low)
    # "No maximum age" is stored as has_max_age = 0 plus a stand-in number, so the flag decides.
    has_high = flag(row["has_max_age"]) and pd.notna(high)

    if has_low and has_high:
        return f"{low:g} to {high:g} years"
    if has_low:
        return f"{low:g} years and older"
    if has_high:
        return f"Up to {high:g} years"
    return "Not stated"


def card_html(row, name, outcome=None):
    """Build the HTML of one trial card.

    The card shows only design facts. The chip with the real outcome is added when outcome is given, and the page
    gives it only after the guess.

    Input: row is one trial (a row of the trials table); name is the card name, "A" or "B" (text); outcome is
    "terminated", "completed" or None to hide the outcome.

    Output: the card content as an HTML text.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    # Missing values become "Not stated". label() does that itself when it gets None.
    conditions = "Not stated"
    if pd.notna(row["conditions"]) and row["conditions"]:
        conditions = row["conditions"]

    sponsor = "Not stated"
    if pd.notna(row["lead_sponsor_name"]):
        sponsor = row["lead_sponsor_name"]

    masking = None
    if pd.notna(row["masking"]):
        masking = row["masking"]

    allocation = None
    if pd.notna(row["allocation"]):
        allocation = row["allocation"]

    arms = "Not stated"
    if pd.notna(row["n_arms"]) and row["n_arms"]:
        arms = f"{int(row['n_arms'])}"

    if flag(row["healthy_volunteers"]):
        healthy = "Accepted"
    else:
        healthy = "Not accepted"

    # The facts a player sees: the same design information the model gets, in plain words.
    facts = [
        ("Conditions", conditions),
        ("Sponsor", sponsor),
        ("Blinding", label("masking", masking)),
        ("Allocation", label("allocation", allocation)),
        ("Number of arms", arms),
        ("Healthy volunteers", healthy),
        ("Eligible ages", age_range(row)),
    ]
    lines = ""
    for fact_name, fact_value in facts:
        lines += f'<div class="card-sub"><b>{esc(fact_name)}:</b> {esc(fact_value)}</div>'

    # The chips on top: the card name, the phase and (only after the guess) the outcome.
    chips = chip(f"Trial {name}") + chip(label("phase", row["phase_label"]))
    if outcome:
        chip_text, chip_colour = OUTCOME_CHIP[outcome]
        chips += chip(chip_text, chip_colour)

    return f'{chips}<div class="card-title">{esc(row["brief_title"])}</div>{lines}'


# ---------------------------------------------------------------------------------------------------------------
# What the buttons do. Streamlit calls these functions before it runs the page again.
# ---------------------------------------------------------------------------------------------------------------

def start_game():
    """Draw ten new rounds and set the score back to zero.

    Input: none.

    Output: nothing. It writes the new game into st.session_state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state.game_rounds = game.new_game(get_trials(), ROUNDS)
    st.session_state.game_round = 0
    # The user's pick of every finished round: "terminated" (right) or "completed" (wrong).
    st.session_state.game_picks = []
    st.session_state.game_revealed = False


def pick(outcome):
    """Save the guess of the user and show the answer.

    Input: outcome is the real outcome of the trial the user clicked, "terminated" or "completed" (text).

    Output: nothing. It stores the pick in st.session_state and switches to the answer view.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state.game_picks.append(outcome)
    st.session_state.game_revealed = True


def next_round():
    """Go on to the next round (the answer is hidden again).

    Input: none.

    Output: nothing. It raises the round counter in st.session_state.
    """
    # AI use: written with Claude Code (Anthropic), see docs/AI_USE.md
    st.session_state.game_round += 1
    st.session_state.game_revealed = False


# ---------------------------------------------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------------------------------------------

# ---- 1. Header and loading ----
page_header("Play the game",
            "Guess the stopped trial: two past trials from the same phase. One was stopped early, one was "
            "completed. Which was which?")

# Load the test trials and the model's accuracy. If a file is missing, say which one and stop.
trials = get_trials()
if trials is None:
    st.error("The trial data is missing. Run `python src/fetch_data.py` and `python src/build_dataset.py`, "
             "then reload.")
    st.stop()
try:
    accuracy = model_accuracy()
except FileNotFoundError:
    st.error("The trained model is missing (`models/gbm.joblib`). Run `python src/train.py`, then reload.")
    st.stop()

# ---- 2. Game state and score ----
# A new visitor gets a new game. The game lives in st.session_state, so it survives the reruns after each click.
if "game_rounds" not in st.session_state:
    start_game()
rounds = st.session_state.game_rounds
index = st.session_state.game_round
picks = st.session_state.game_picks

# Count the points of the rounds that are finished. Picking the terminated trial is the right answer.
you = 0
model_points = 0
for number in range(len(picks)):
    if picks[number] == "terminated":
        you += 1
    if rounds[number]["model_pick"] == "terminated":
        model_points += 1

# A line about the model's accuracy, shown in every round.
st.markdown(f"On 2,000 random pairs the model picked the stopped trial **{accuracy:.0%}** of the time. "
            "Guessing gets 50%. These trials started in 2017-2019, after the period the model learned from.")

# ---- 3. After the last round: the final score ----
if index >= len(rounds):
    # Three tiles: your score, the model's score and the model's accuracy on 2,000 pairs.
    st.markdown("### Final score")
    left, middle, right = st.columns(3)
    left.metric("You", f"{you} / {len(rounds)}")
    middle.metric("Model", f"{model_points} / {len(rounds)}")
    right.metric("Model on 2,000 pairs", f"{accuracy:.0%}")
    # Who won this game.
    if you > model_points:
        st.markdown("**You beat the model.**")
    elif you < model_points:
        st.markdown("**The model wins this time.**")
    else:
        st.markdown("**A draw.**")

    # One table row per round: the trial that was really stopped, and who got it right.
    stopped_titles = []
    your_results = []
    model_results = []
    for number in range(len(rounds)):
        stopped_titles.append(trials.at[rounds[number]["terminated"], "brief_title"])
        if picks[number] == "terminated":
            your_results.append("Right")
        else:
            your_results.append("Wrong")
        if rounds[number]["model_pick"] == "terminated":
            model_results.append("Right")
        else:
            model_results.append("Wrong")
    # Show the round table and offer a new game.
    table = pd.DataFrame({"Stopped early": stopped_titles, "You": your_results, "Model": model_results},
                         index=range(1, len(rounds) + 1))
    table.index.name = "Round"
    st.table(table)

    st.button("Play again", type="primary", on_click=start_game, icon=":material/replay:")
    st.stop()

# ---- 4. One round: the two cards ----
# The current round and whether the user already guessed.
current = rounds[index]
revealed = st.session_state.game_revealed

# game.new_game decided at random which outcome is shown on the left (card A). The other one goes right (card B).
if current["left"] == "terminated":
    order = ["terminated", "completed"]
else:
    order = ["completed", "terminated"]

# The round counter and the running score.
st.markdown(f"**Round {index + 1} of {len(rounds)}** &nbsp;·&nbsp; You {you} &nbsp;·&nbsp; Model {model_points}",
            unsafe_allow_html=True)

# Two cards side by side. Before the guess each card has a button, after the guess a link to the real record.
columns = st.columns(2)
for position in range(2):
    name = SIDES[position]
    outcome = order[position]
    row = trials.loc[current[outcome]]
    # Only show the real outcome on the card after the user has guessed.
    shown_outcome = None
    if revealed:
        shown_outcome = outcome

    with columns[position]:
        with st.container(key=f"card_game_{name}"):
            st.markdown(card_html(row, name, shown_outcome), unsafe_allow_html=True)
            if revealed:
                st.link_button("Open on ClinicalTrials.gov", f"https://clinicaltrials.gov/study/{row['nct_id']}",
                               icon=":material/open_in_new:")
            else:
                st.button(f"Trial {name} was stopped early", key=f"pick_{index}_{name}", on_click=pick,
                          args=(outcome,), type="primary", use_container_width=True)

# ---- 5. After the guess: the answer ----
# After the guess: the right answer, the user's pick and the model's pick in one sentence.
if revealed:
    truth = SIDES[order.index("terminated")]
    user_side = SIDES[order.index(picks[-1])]
    model_side = SIDES[order.index(current["model_pick"])]

    # Compare both picks with the truth.
    user_result = "wrong"
    if user_side == truth:
        user_result = "right"
    model_result = "wrong"
    if model_side == truth:
        model_result = "right"
    st.markdown(f"**Trial {truth} was stopped early.** You picked {user_side}: {user_result}. "
                f"The model picked {model_side}: {model_result}.")

    # The button to go on. After the last round it leads to the final score.
    button_text = "Next round"
    if index + 1 >= len(rounds):
        button_text = "See the final score"
    st.button(button_text, type="primary", on_click=next_round, icon=":material/arrow_forward:")

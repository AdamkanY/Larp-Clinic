# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Shared look of the Trial-Radar pages: the one colour palette, the Plotly template built from it, one stylesheet
(style.css), full-width bands, the tick scale and the page header."""
import html
import re
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

STYLE = Path(__file__).resolve().parent / "style.css"
TICKS = 100  # one tick = 1% of the past trials

# The only place colours are defined. style.css reads them as CSS variables (--ink, --low-tint, ...), every Plotly
# chart through the "trial_radar" template below. .streamlit/config.toml cannot import Python, so its [theme]
# repeats five of them; tests/test_app.py checks that it still matches.
PALETTE = {
    # Surfaces and text
    "ink": "#0A1F1A", "ink_2": "#123029", "paper": "#EDEFE6", "paper_2": "#F6F7F1", "line": "#C8CCBE",
    "field_line": "#78857D",  # 3:1 against the paper, so input borders are visible
    "muted": "#4B5A53", "muted_on_ink": "#A9B8B0", "glow": "#226852",
    # The pink accent: only "this trial" and the one main action
    "accent": "#FF3D8B", "accent_deep": "#C80F5A", "accent_text": "#9E0A4A",
    # Risk tones (chips, cards, the comparison bar): text, tint behind text, on dark bands, pale chart band
    "low": "#1F6B52", "mid": "#8A5A0B", "high": "#A3282A",
    "low_tint": "#DCEBE2", "mid_tint": "#F3E6C6", "high_tint": "#F2D9D5",
    "low_on_ink": "#6FCBA5", "mid_on_ink": "#E8B94A", "high_on_ink": "#FF8A75",
    "low_band": "#C5DDD0", "mid_band": "#EBDDAE", "high_band": "#E3B7B0",
    # Chart series: one deep green for "share stopped early", orange for "raises the risk", grey for neither.
    # Each is at least 3:1 against the paper.
    "bar": "#17463A", "up": "#B8452E", "neutral": "#6B7A72", "pale": "#D3DAD3",
}
PALETTE["down"] = PALETTE["bar"]  # "lowers the risk" uses the same green as the data bars
FONT_FAMILY = "IBM Plex Sans, sans-serif"
TEMPLATE = "trial_radar"

# The finished stylesheet is kept here after the first call, so style.css is read from disk only once per server
# (Streamlit runs a page again on every click).
LOADED_STYLESHEET = {}


def alpha(colour, opacity):
    """A palette colour like "#0A1F1A" as "rgba(10,31,26,opacity)".

    Plotly has no other way to make a colour partly transparent.
    """
    value = colour.lstrip("#")
    # A hex colour is three pairs of digits: red, green and blue, each from 00 to FF.
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{opacity})"


def plotly_template():
    """The look of all charts as a Plotly template, built from the palette.

    Transparent background so the paper shows through, thin grid lines, dark tooltips, muted axis text, the
    palette's series colours, a heatmap scale from paper to orange and reference lines in the muted colour.
    """
    # Two text styles: normal text in ink, small text (axes, annotations) in the muted colour.
    font = {"family": FONT_FAMILY, "size": 13, "color": PALETTE["ink"]}
    small = {"family": FONT_FAMILY, "size": 12, "color": PALETTE["muted"]}

    # Both axes look the same.
    axis = {
        "tickfont": small,
        "title": {"font": small},
        "gridcolor": alpha(PALETTE["ink"], 0.12),
        "linecolor": PALETTE["ink"],
        "zeroline": False,
        "automargin": True,
    }

    # Heatmaps (src/viz.py rate_heatmap) shade from the light paper through the pale yellow risk band to the
    # "raises the risk" orange, so darker always means "stopped more often". Plotly picks dark or light cell text
    # for contrast on its own.
    risk_scale = [[0.0, PALETTE["paper_2"]], [0.5, PALETTE["mid_band"]], [1.0, PALETTE["up"]]]

    layout = {
        "font": font,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": [PALETTE["bar"], PALETTE["up"], PALETTE["neutral"], PALETTE["accent_deep"]],
        "hoverlabel": {
            "bgcolor": PALETTE["ink"],
            "bordercolor": PALETTE["ink"],
            "align": "left",
            "font": {"family": FONT_FAMILY, "color": PALETTE["paper"], "size": 13},
        },
        "xaxis": axis,
        "yaxis": axis,
        "legend": {"font": font, "title": {"text": ""}},
        "title": {"font": {"family": FONT_FAMILY, "size": 15, "color": PALETTE["ink"]}},
        "shapedefaults": {"line": {"color": PALETTE["muted"]}},
        "annotationdefaults": {"font": small},
    }
    return go.layout.Template(data={"heatmap": [go.Heatmap(colorscale=risk_scale)]}, layout=layout)


# Registered on import, so every figure the pages (and src/viz.py) build afterwards starts from this look.
pio.templates[TEMPLATE] = plotly_template()
pio.templates.default = TEMPLATE


def chart(fig, key=None):
    """Show a Plotly figure in the app look, full width. Every page uses this instead of st.plotly_chart.

    theme=None stops Streamlit from laying its own chart theme over our template.
    """
    # Streamlit's frontend still fills the font and both backgrounds from its page theme, unless the figure's own
    # layout sets them (values that only stand in the template do not count). So we copy them from the template
    # onto the layout here. What a chart set itself is kept.
    base = pio.templates[TEMPLATE].layout
    font = {}
    font.update(base.font.to_plotly_json())
    font.update(fig.layout.font.to_plotly_json())
    paper_colour = fig.layout.paper_bgcolor
    if not paper_colour:
        paper_colour = base.paper_bgcolor
    plot_colour = fig.layout.plot_bgcolor
    if not plot_colour:
        plot_colour = base.plot_bgcolor
    fig.update_layout(template=TEMPLATE, font=font, paper_bgcolor=paper_colour, plot_bgcolor=plot_colour)

    # The margins grow to fit the axis titles.
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, theme=None, width="stretch", key=key)


def css_variables():
    """The palette as CSS variables for the top of style.css, for example "ink_2" becomes "--ink-2"."""
    variables = []
    for name, value in PALETTE.items():
        css_name = name.replace("_", "-")
        variables.append(f"--{css_name}: {value};")
    return ":root {" + " ".join(variables) + "}"


def stylesheet():
    """The palette's CSS variables followed by the text of style.css, built once per server."""
    if "css" not in LOADED_STYLESHEET:
        LOADED_STYLESHEET["css"] = css_variables() + "\n" + STYLE.read_text(encoding="utf-8")
    return LOADED_STYLESHEET["css"]


def plain(title):
    """Drop a leading emoji or symbol from a page title. The look is plain type, no pictograms."""
    # "^[^\w]+" means: everything at the start that is not a letter, a digit or an underscore.
    return re.sub(r"^[^\w]+", "", str(title))


def esc(text):
    """Make a text safe to put inside HTML that goes through st.markdown.

    Characters like < and & are escaped. The dollar sign too, because st.markdown would start a formula with it.
    """
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def chip(text, tone="gray"):
    """A small pill label, for example a phase or a status. Tones: gray, green, yellow, red."""
    return f'<span class="chip {tone}">{esc(text)}</span>'


def require_router(page):
    """Send a page that was opened directly (browser refresh, bookmark) through Home.py first.

    Opened on its own, a page would miss the top navigation and the wide layout that Home.py sets up. Home.py
    sets "via_router" and, if a target was stored here, switches straight back to that page.
    """
    if not st.session_state.get("via_router"):
        st.session_state["target_page"] = page
        st.switch_page("Home.py")


def inject_css():
    """Apply the shared stylesheet to the current page."""
    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


def tick_scale(marked, label=""):
    """The scale that runs through the app: 100 thin lines, the first `marked` of them tall and pink.

    It is decoration (hidden from screen readers). `label` is the visible sentence that says what the lines mean.
    """
    # One <i> per tick. The CSS class "on" makes a tick tall and pink, "--i" is its position for the animation.
    ticks = ""
    for i in range(TICKS):
        if i < marked:
            css_class = "on"
        else:
            css_class = ""
        ticks += f'<i class="{css_class}" style="--i:{i}"></i>'

    caption = ""
    if label:
        caption = f'<p class="ticks-caption">{esc(label)}</p>'
    return f'<div class="ticks" aria-hidden="true">{ticks}</div>{caption}'


def hero(title, subtitle, number=None, lead=None, scale=None):
    """The first band of the landing page: title, subtitle, a giant number and a tick scale.

    `number` is the giant figure (as text), `lead` the sentence under it and `scale` the HTML from tick_scale().
    All three can be left out.
    """
    inject_css()

    # The optional parts stay empty when they were not given.
    figure = ""
    if number:
        figure = f'<p class="hero-number reveal" style="--d:1">{esc(number)}</p>'
    lead_html = ""
    if lead:
        lead_html = f'<p class="hero-lead reveal" style="--d:2">{esc(lead)}</p>'
    if not scale:
        scale = ""

    # "reveal" fades a part in when the page loads, "--d" is its place in the order.
    st.markdown(
        f'<div class="hero-copy"><h1 class="hero-title reveal" style="--d:0">{esc(plain(title))}</h1>'
        f'<p class="hero-sub reveal" style="--d:0">{esc(subtitle)}</p>{figure}{lead_html}'
        f'<div class="reveal" style="--d:3">{scale}</div></div>',
        unsafe_allow_html=True,
    )


def band_copy(headline, text, number=None, number_label=None):
    """The HTML for one band of the landing page: headline, one sentence and an optional big number.

    The link to the page is not part of it. The page adds it as a widget below.
    """
    figure = ""
    if number:
        figure = (f'<div class="band-figure"><p class="band-number">{esc(number)}</p>'
                  f'<p class="band-number-label">{esc(number_label)}</p></div>')
    return (f'<div class="band-copy"><div class="band-text"><h2 class="band-title">{esc(headline)}</h2>'
            f'<p class="band-sentence">{esc(text)}</p></div>{figure}</div>')


def page_header(title, subtitle=None):
    """The dark band that opens every page except Home: a very large title and an optional sentence below it."""
    inject_css()
    sub = ""
    if subtitle:
        sub = f'<p class="page-sub reveal" style="--d:1">{esc(subtitle)}</p>'
    st.markdown(f'<div class="page-head"><h1 class="reveal" style="--d:0">{esc(plain(title))}</h1>{sub}</div>',
                unsafe_allow_html=True)

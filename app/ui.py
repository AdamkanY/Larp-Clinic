"""AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.

Shared look of the Trial-Radar pages: the one colour palette, the Plotly template built from it, one stylesheet
(style.css), full-width bands, the tick scale and the page header.
"""
import html  # escapes text for safe use inside HTML
import re  # regular expressions, to clean page titles
from pathlib import Path  # builds file paths that work on every computer

import plotly.graph_objects as go  # figure and template classes of Plotly
import plotly.io as pio  # registers the chart template
import streamlit as st  # the web app framework

# The stylesheet file and the number of ticks of the tick scale.
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
# Font and template name shared by the charts.
FONT_FAMILY = "IBM Plex Sans, sans-serif"
TEMPLATE = "trial_radar"

# The finished stylesheet is kept here after the first call, so style.css is read from disk only once per server
# (Streamlit runs a page again on every click).
LOADED_STYLESHEET = {}


def alpha(colour, opacity):
    """Turn a palette colour like "#0A1F1A" into "rgba(10,31,26,opacity)".

    Plotly has no other way to make a colour partly transparent.

    Input: colour is a hex colour as text, for example "#0A1F1A"; opacity is a number from 0 (clear) to 1 (solid).

    Output: the colour as an rgba text.
    """
    # Remove the hash sign first.
    value = colour.lstrip("#")
    # A hex colour is three pairs of digits: red, green and blue, each from 00 to FF.
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{opacity})"


def plotly_template():
    """Build the look of all charts as a Plotly template, from the palette.

    Transparent background so the paper shows through, thin grid lines, dark tooltips, muted axis text, the
    palette's series colours, a heatmap scale from paper to orange and reference lines in the muted colour.

    Input: none.

    Output: a Plotly template object.
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

    # The whole layout: fonts, transparent backgrounds, series colours, tooltips, axes, legend and title.
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
    # The heatmap colour scale is part of the template data.
    return go.layout.Template(data={"heatmap": [go.Heatmap(colorscale=risk_scale)]}, layout=layout)


# Registered on import, so every figure the pages (and src/viz.py) build afterwards starts from this look.
pio.templates[TEMPLATE] = plotly_template()
pio.templates.default = TEMPLATE


def chart(fig, key=None):
    """Show a Plotly figure in the app look, full width. Every page uses this instead of st.plotly_chart.

    theme=None stops Streamlit from laying its own chart theme over our template.

    Input: fig is the Plotly figure to show; key is an optional Streamlit widget key (text) for the chart.

    Output: nothing. It draws the chart on the page.
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
    # Draw it full width without the Streamlit chart theme.
    st.plotly_chart(fig, theme=None, width="stretch", key=key)


def css_variables():
    """Turn the palette into CSS variables for the top of style.css, for example "ink_2" becomes "--ink-2".

    Input: none.

    Output: the CSS rule as a text.
    """
    # One CSS variable per palette entry.
    variables = []
    for name, value in PALETTE.items():
        css_name = name.replace("_", "-")
        variables.append(f"--{css_name}: {value};")
    # All variables go into one :root rule.
    return ":root {" + " ".join(variables) + "}"


def stylesheet():
    """Get the palette's CSS variables followed by the text of style.css, built once per server.

    Input: none.

    Output: the whole stylesheet as a text.
    """
    # Build it on the first call only, then reuse the stored text.
    if "css" not in LOADED_STYLESHEET:
        LOADED_STYLESHEET["css"] = css_variables() + "\n" + STYLE.read_text(encoding="utf-8")
    return LOADED_STYLESHEET["css"]


def plain(title):
    """Drop a leading emoji or symbol from a page title. The look is plain type, no pictograms.

    Input: title is the page title (text, or anything that can be turned into text).

    Output: the title as a text without leading symbols.
    """
    # "^[^\w]+" means: everything at the start that is not a letter, a digit or an underscore.
    return re.sub(r"^[^\w]+", "", str(title))


def esc(text):
    """Make a text safe to put inside HTML that goes through st.markdown.

    Characters like < and & are escaped. The dollar sign too, because st.markdown would start a formula with it.

    Input: text is the text to protect (anything that can be turned into text).

    Output: the escaped text.
    """
    # Escape the HTML characters, then the dollar sign.
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def chip(text, tone="gray"):
    """Build a small pill label, for example a phase or a status.

    Input: text is the label (text); tone is the colour, one of "gray", "green", "yellow" or "red".

    Output: the label as an HTML text.
    """
    return f'<span class="chip {tone}">{esc(text)}</span>'


def require_router(page):
    """Send a page that was opened directly (browser refresh, bookmark) through Home.py first.

    Opened on its own, a page would miss the top navigation and the wide layout that Home.py sets up. Home.py
    sets "via_router" and, if a target was stored here, switches straight back to that page.

    Input: page is the path of the calling page, for example "pages/1_Find_a_Trial.py".

    Output: nothing. If the page was opened directly it stores the page and switches to Home.py.
    """
    # Only act when Home.py has not run in this session yet.
    if not st.session_state.get("via_router"):
        st.session_state["target_page"] = page
        st.switch_page("Home.py")


def inject_css():
    """Apply the shared stylesheet to the current page.

    Input: none.

    Output: nothing. It writes the style block on the page.
    """
    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


def tick_scale(marked, label=""):
    """Build the scale that runs through the app: 100 thin lines, the first `marked` of them tall and pink.

    It is decoration (hidden from screen readers). `label` is the visible sentence that says what the lines mean.

    Input: marked is how many ticks are tall and pink (whole number, 0 to 100); label is the sentence under the
    scale (text, may be empty).

    Output: the scale as an HTML text.
    """
    # One <i> per tick. The CSS class "on" makes a tick tall and pink, "--i" is its position for the animation.
    ticks = ""
    for i in range(TICKS):
        if i < marked:
            css_class = "on"
        else:
            css_class = ""
        ticks += f'<i class="{css_class}" style="--i:{i}"></i>'

    # The caption is only added when there is a label.
    caption = ""
    if label:
        caption = f'<p class="ticks-caption">{esc(label)}</p>'
    # Ticks first, caption below.
    return f'<div class="ticks" aria-hidden="true">{ticks}</div>{caption}'


def hero(title, subtitle, number=None, lead=None, scale=None):
    """Draw the first band of the landing page: title, subtitle, a giant number and a tick scale.

    Input: title is the big question (text); subtitle is the line below it (text); number is the giant figure (text,
    optional); lead is the sentence under the figure (text, optional); scale is the HTML from tick_scale()
    (optional).

    Output: nothing. It writes the band on the page.
    """
    # Make sure the styles are on the page.
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
    """Build the HTML for one band of the landing page: headline, one sentence and an optional big number.

    The link to the page is not part of it. The page adds it as a widget below.

    Input: headline is the band title (text); text is one sentence (text); number is a big figure (text, optional);
    number_label is the words under that figure (text, needed when number is given).

    Output: the band content as an HTML text.
    """
    # The big number block only exists when a number was given.
    figure = ""
    if number:
        figure = (f'<div class="band-figure"><p class="band-number">{esc(number)}</p>'
                  f'<p class="band-number-label">{esc(number_label)}</p></div>')
    # Headline and sentence on the left, the optional figure beside them.
    return (f'<div class="band-copy"><div class="band-text"><h2 class="band-title">{esc(headline)}</h2>'
            f'<p class="band-sentence">{esc(text)}</p></div>{figure}</div>')


def page_header(title, subtitle=None):
    """Draw the dark band that opens every page except Home: a very large title and an optional sentence below it.

    Input: title is the page title (text); subtitle is the sentence below it (text, optional).

    Output: nothing. It writes the band on the page.
    """
    # Make sure the styles are on the page, then build the optional subtitle.
    inject_css()
    sub = ""
    if subtitle:
        sub = f'<p class="page-sub reveal" style="--d:1">{esc(subtitle)}</p>'
    # Title and subtitle as one HTML block.
    st.markdown(f'<div class="page-head"><h1 class="reveal" style="--d:0">{esc(plain(title))}</h1>{sub}</div>',
                unsafe_allow_html=True)

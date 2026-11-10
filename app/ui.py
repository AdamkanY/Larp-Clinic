# AI use: written with the help of Claude Code (Anthropic) as a coding assistant. What the AI was asked, the share
# of AI-written lines per file and what the team did by hand are recorded in docs/AI_USE.md.
"""Shared look of the Trial-Radar pages: the one colour palette, the Plotly template built from it, one stylesheet
(style.css), full-width bands, the tick scale and the page header."""
import functools
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


def alpha(colour, opacity):
    """A palette hex colour as rgba() with the given opacity (Plotly has no color-mix)."""
    value = colour.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{opacity})"


def plotly_template():
    """The chart look as a Plotly template: transparent so the paper shows through, hairline grid, ink tooltips,
    muted axis text, the palette's series colours, a paper-to-orange heatmap scale and dashed reference lines in
    the muted colour."""
    font = {"family": FONT_FAMILY, "size": 13, "color": PALETTE["ink"]}
    small = {"family": FONT_FAMILY, "size": 12, "color": PALETTE["muted"]}
    axis = {"tickfont": small, "title": {"font": small}, "gridcolor": alpha(PALETTE["ink"], 0.12),
            "linecolor": PALETTE["ink"], "zeroline": False, "automargin": True}
    # Heatmaps (src/viz.py rate_heatmap) shade from the light paper through the pale yellow risk band to the
    # "raises the risk" orange, so darker always means "stopped more often". Plotly picks dark or light cell text
    # for contrast on its own.
    risk_scale = [[0.0, PALETTE["paper_2"]], [0.5, PALETTE["mid_band"]], [1.0, PALETTE["up"]]]
    return go.layout.Template(data={"heatmap": [go.Heatmap(colorscale=risk_scale)]}, layout={
        "font": font, "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": [PALETTE["bar"], PALETTE["up"], PALETTE["neutral"], PALETTE["accent_deep"]],
        "hoverlabel": {"bgcolor": PALETTE["ink"], "bordercolor": PALETTE["ink"], "align": "left",
                       "font": {"family": FONT_FAMILY, "color": PALETTE["paper"], "size": 13}},
        "xaxis": axis, "yaxis": axis, "legend": {"font": font, "title": {"text": ""}},
        "title": {"font": {"family": FONT_FAMILY, "size": 15, "color": PALETTE["ink"]}},
        "shapedefaults": {"line": {"color": PALETTE["muted"]}},
        "annotationdefaults": {"font": small},
    })


# Registered on import, so every figure the pages (and src/viz.py) build afterwards starts from this look.
pio.templates[TEMPLATE] = plotly_template()
pio.templates.default = TEMPLATE


def chart(fig, key=None):
    """Show a Plotly figure in the app look, full width.

    theme=None stops Streamlit from laying its own chart theme over the template. Streamlit's frontend still fills
    the font and both backgrounds from its page theme unless the figure's own layout sets them (template values do
    not count), so they are copied from the template onto the layout here; values a chart set itself are kept.
    Axis margins grow to fit their titles.
    """
    base = pio.templates[TEMPLATE].layout
    fig.update_layout(
        template=TEMPLATE,
        font={**base.font.to_plotly_json(), **fig.layout.font.to_plotly_json()},
        paper_bgcolor=fig.layout.paper_bgcolor or base.paper_bgcolor,
        plot_bgcolor=fig.layout.plot_bgcolor or base.plot_bgcolor,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, theme=None, width="stretch", key=key)


def css_variables():
    """The palette as CSS custom properties (ink_2 -> --ink-2), for the top of style.css."""
    return ":root {" + " ".join(f"--{name.replace('_', '-')}: {value};" for name, value in PALETTE.items()) + "}"


@functools.lru_cache(maxsize=1)
def stylesheet():
    """The palette's CSS variables followed by the text of style.css, built once per server."""
    return css_variables() + "\n" + STYLE.read_text(encoding="utf-8")


def plain(title):
    """Drop a leading emoji or symbol from a page title; the look is plain type, no pictograms."""
    return re.sub(r"^[^\w]+", "", str(title))


def esc(text):
    """Escape text for use inside HTML that goes through st.markdown (also keeps '$' from turning into math)."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def chip(text, tone="gray"):
    """A small pill label, for example a phase or a status. Tones: gray, green, yellow, red."""
    return f'<span class="chip {tone}">{esc(text)}</span>'


def require_router(page):
    """Send a page that was opened directly (browser refresh, bookmark) through Home.py first.

    Opened on its own, a page would miss the top navigation and the wide layout that Home.py sets up. Home.py
    sets `via_router` and, if a target was stored here, switches straight back to that page.
    """
    if not st.session_state.get("via_router"):
        st.session_state["target_page"] = page
        st.switch_page("Home.py")


def inject_css():
    """Apply the shared stylesheet to the current page."""
    st.markdown(f"<style>{stylesheet()}</style>", unsafe_allow_html=True)


def tick_scale(marked, label=""):
    """The signature scale: 100 hairlines, the first `marked` of them tall and in the accent colour.

    Decorative (hidden from screen readers); `label` is the visible sentence that says what the ticks mean.
    """
    ticks = "".join(f'<i class="{"on" if i < marked else ""}" style="--i:{i}"></i>' for i in range(TICKS))
    caption = f'<p class="ticks-caption">{esc(label)}</p>' if label else ""
    return f'<div class="ticks" aria-hidden="true">{ticks}</div>{caption}'


def hero(title, subtitle, number=None, lead=None, scale=None):
    """The landing page's first band. `number` is the giant figure (text), `lead` the sentence under it.

    `scale` is the HTML of a tick scale. Without a number the title is simply set very large.
    """
    inject_css()
    figure = f'<p class="hero-number reveal" style="--d:1">{esc(number)}</p>' if number else ""
    lead_html = f'<p class="hero-lead reveal" style="--d:2">{esc(lead)}</p>' if lead else ""
    st.markdown(
        f'<div class="hero-copy"><h1 class="hero-title reveal" style="--d:0">{esc(plain(title))}</h1>'
        f'<p class="hero-sub reveal" style="--d:0">{esc(subtitle)}</p>{figure}{lead_html}'
        f'<div class="reveal" style="--d:3">{scale or ""}</div></div>',
        unsafe_allow_html=True,
    )


def band_copy(headline, text, number=None, number_label=None):
    """Headline, one sentence and an optional big number for one landing-page band (the action is a widget below)."""
    figure = (f'<div class="band-figure"><p class="band-number">{esc(number)}</p>'
              f'<p class="band-number-label">{esc(number_label)}</p></div>') if number else ""
    return (f'<div class="band-copy"><div class="band-text"><h2 class="band-title">{esc(headline)}</h2>'
            f'<p class="band-sentence">{esc(text)}</p></div>{figure}</div>')


def page_header(title, subtitle=None):
    """Dark band that opens a page: the title set very large, an optional one-sentence description, a ruler edge."""
    inject_css()
    sub = f'<p class="page-sub reveal" style="--d:1">{esc(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<div class="page-head"><h1 class="reveal" style="--d:0">{esc(plain(title))}</h1>{sub}</div>',
                unsafe_allow_html=True)

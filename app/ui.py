"""Shared look of the Trial-Radar pages: one stylesheet (style.css), full-width bands, the tick scale and the page header."""
import functools
import html
import re
from pathlib import Path

import streamlit as st

STYLE = Path(__file__).resolve().parent / "style.css"
TICKS = 100  # one tick = 1% of the past trials


@functools.lru_cache(maxsize=1)
def stylesheet():
    """Text of style.css, read once per server."""
    return STYLE.read_text(encoding="utf-8")


def plain(title):
    """Drop a leading emoji or symbol from a page title; the look is plain type, no pictograms."""
    return re.sub(r"^[^\w]+", "", str(title))


def esc(text):
    """Escape text for use inside HTML that goes through st.markdown (also keeps '$' from turning into math)."""
    return html.escape(str(text), quote=True).replace("$", "&#36;")


def chip(text, tone="gray"):
    """A small pill label, for example a phase or a status. Tones: gray, green, yellow, red."""
    return f'<span class="chip {tone}">{esc(text)}</span>'


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

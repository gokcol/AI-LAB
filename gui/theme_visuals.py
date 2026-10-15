"""Theme-aware lesson illustrations and plots.

Inline diagrams use CSS colors, so Browser default can change without a Python
rerun. Matplotlib data (images, heatmaps, contours and series) keeps its original
colors; only the figure furniture is recolored. Native Vega charts retain their
hover/zoom behavior and use SVG so the same CSS theme reaches their labels.
"""
from __future__ import annotations

import colorsys
import functools
import io
import re
import threading
import uuid

import streamlit as st

_SVG = re.compile(r"<svg\b[\s\S]*?</svg\s*>", re.I)
_PAINT = re.compile(r'((?:fill|stroke|stop-color)=[\"\'])(#[0-9a-fA-F]{3,6})([\"\'])')
_LOCK = threading.RLock()


def dark_color(value: str) -> str:
    """Keep diagram color families, with dark surfaces and readable light ink."""
    if value.lower() in {"#fafaf7", "#ffffff", "#fff", "#f4f7fb"}:
        return "#1b2839"
    if value.lower() in {"#e2e2da", "#e3e6ea", "#efeee8"}:
        return "#40536d"
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    r, g, b = (int(value[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, light, sat = colorsys.rgb_to_hls(r, g, b)
    if light > .78:  # pale teaching panels become tinted dark panels
        light = .16 + (1 - light) * .30
        sat = min(sat, .38)
    else:  # labels, arrows and nodes retain their hue, with enough contrast
        light = max(.72, light)
        sat = min(sat, .78)
    return "#" + "".join(f"{round(c * 255):02x}" for c in colorsys.hls_to_rgb(h, light, sat))


def diagram_html(body: str, selected: str) -> str:
    """Adapt only inline SVG paint attributes, never prose, images or plot data."""
    if "<svg" not in body or selected not in {"dark", "browser"}:
        return body

    def adapt(match):
        svg = match.group()
        if 'class="ailab-plot"' in svg:
            return svg
        colors = set()
        def paint(m):
            color = m[2].lower()
            colors.add(color)
            return f'{m[1]}var(--diagram-{color[1:]},{color}){m[3]}'
        svg = _PAINT.sub(paint, svg)
        rules = ";".join(f"--diagram-{c[1:]}:{dark_color(c)}" for c in sorted(colors))
        # Each diagram has its own palette; never let a diagram style page text.
        scope = "diagram-" + uuid.uuid4().hex
        svg = svg.replace("<svg ", f'<svg class="{scope}" ', 1)
        css = f".{scope}{{{rules}}}"
        if selected == "browser":
            css = f"@media (prefers-color-scheme: dark){{{css}}}"
        return f"<style>{css}</style>{svg}"
    return _SVG.sub(adapt, body)


def figure_svg(fig, **save_options) -> str:
    """Export a sharp plot with CSS-driven furniture and unchanged data colors.

    No global rcParams or per-visitor Matplotlib style is installed. Restore the
    supplied figure after export, including annotations and legend backgrounds.
    """
    from matplotlib.text import Text
    from matplotlib.colors import to_hex
    from matplotlib import rc_context

    saved = []
    tokens = {"#010203": "var(--ailab-panel)", "#020304": "var(--ailab-ink)",
              "#030405": "var(--ailab-muted)", "#040506": "var(--ailab-border)"}
    def set_prop(artist, prop, value):
        saved.append((artist, prop, getattr(artist, "get_" + prop)()))
        getattr(artist, "set_" + prop)(value)

    with _LOCK:
        try:
            set_prop(fig.patch, "facecolor", "#010203")
            for ax in fig.axes:
                set_prop(ax.patch, "facecolor", "#010203")
                for spine in ax.spines.values():
                    set_prop(spine, "edgecolor", "#040506")
                for line in ax.get_xgridlines() + ax.get_ygridlines():
                    set_prop(line, "color", "#040506")
                for axis in (ax.xaxis, ax.yaxis):
                    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                        for line in (tick.tick1line, tick.tick2line):
                            set_prop(line, "color", "#030405")
                            set_prop(line, "markeredgecolor", "#030405")
                            set_prop(line, "markerfacecolor", "#030405")
                # Labels over a colormap are data annotations: preserve their
                # deliberate black/white contrast against the original cells.
                data_labels = set(ax.texts) if ax.images or any(
                    type(c).__name__ in {"QuadMesh", "QuadContourSet", "TriContourSet"}
                    for c in ax.collections) else set()
                for text in ax.findobj(Text):
                    if text in data_labels and text.get_bbox_patch() is None:
                        continue
                    set_prop(text, "color", "#020304")
                    if text.get_bbox_patch() is not None:
                        set_prop(text.get_bbox_patch(), "facecolor", "#010203")
                        set_prop(text.get_bbox_patch(), "edgecolor", "#040506")
                legend = ax.get_legend()
                if legend:
                    set_prop(legend.get_frame(), "facecolor", "#010203")
                    set_prop(legend.get_frame(), "edgecolor", "#040506")
                # Neutral reference lines (not colored data series).
                for line in ax.lines:
                    if str(line.get_color()).lower() == "none":
                        continue
                    color = to_hex(line.get_color())
                    if color in {"#000000", "#333333", "#33312e", "#595959", "#808080", "#cccccc", "#d9d9d9"}:
                        set_prop(line, "color", "#030405")
            for text in fig.texts:
                set_prop(text, "color", "#020304")
            buf = io.StringIO()
            options = {"bbox_inches": "tight", **save_options, "format": "svg"}
            options.pop("dpi", None)
            with rc_context({"svg.fonttype": "none"}):
                fig.savefig(buf, **options)
        finally:
            for artist, prop, value in reversed(saved):
                getattr(artist, "set_" + prop)(value)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    # Namespace IDs: independent figures often have identical clip/marker IDs.
    prefix = "plot" + uuid.uuid4().hex + "_"
    svg = re.sub(r'(?<=id=")([^"\s]+)', lambda m: prefix + m[1], svg)
    svg = re.sub(r'(?<=url\(#)([^)]+)', lambda m: prefix + m[1], svg)
    svg = re.sub(r'(?<=href="#)([^"]+)', lambda m: prefix + m[1], svg)
    for source, target in tokens.items():
        svg = svg.replace(source, target)
    svg = svg.replace("<svg ", '<svg class="ailab-plot" role="img" aria-label="Lesson plot" ', 1)
    svg = re.sub(r'\s(?:width|height)="[^\"]+pt"', '', svg, count=2)
    return svg.replace("<svg ", '<svg style="display:block;width:100%;height:auto" ', 1)


def install() -> None:
    """Install once, before localization captures Streamlit's bound methods."""
    if getattr(st, "_ailab_visuals_installed", False):
        return
    from streamlit.delta_generator import DeltaGenerator
    from streamlit.elements.vega_charts import VegaChartsMixin

    def markdown_wrapper(original):
        @functools.wraps(original)
        def render(self, body, *args, **kwargs):
            if kwargs.get("unsafe_allow_html") and isinstance(body, str):
                body = diagram_html(body, st.session_state.get("theme", "browser"))
            return original(self, body, *args, **kwargs)
        return render

    DeltaGenerator.markdown = markdown_wrapper(DeltaGenerator.markdown)
    st.markdown = st._main.markdown

    def pyplot(self, fig=None, clear_figure=None, *, width="stretch", use_container_width=None, **kwargs):
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()
            clear_figure = True if clear_figure is None else clear_figure
        rendered = self.markdown(figure_svg(fig, **kwargs), unsafe_allow_html=True,
                                 width=width if isinstance(width, int) else "stretch")
        if clear_figure:
            fig.clear()
        return rendered
    DeltaGenerator.pyplot = pyplot
    st.pyplot = st._main.pyplot

    original_chart = VegaChartsMixin._altair_chart
    @functools.wraps(original_chart)
    def svg_chart(self, altair_chart, *args, **kwargs):
        chart = altair_chart.copy(deep=True)
        meta = chart.usermeta if isinstance(chart.usermeta, dict) else {}
        chart.usermeta = {**meta, "embedOptions": {**meta.get("embedOptions", {}), "renderer": "svg"}}
        return original_chart(self, chart, *args, **kwargs)
    VegaChartsMixin._altair_chart = svg_chart
    st._ailab_visuals_installed = True

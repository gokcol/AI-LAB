"""Theme-aware lesson illustrations and plots.

Inline diagrams use CSS colors, so Browser default can change without a Python
rerun. Matplotlib lines, markers and arrows get a brighter dark palette while
images, heatmaps and scalar color scales keep their original colors. Native Vega charts retain their
hover/zoom behavior and use SVG so the same CSS theme reaches their labels.
"""
from __future__ import annotations

import colorsys
import copy
import functools
import html
import io
import math
import re
import threading
import textwrap
import uuid
import xml.etree.ElementTree as ET

import streamlit as st

_SVG = re.compile(r"<svg\b[\s\S]*?</svg\s*>", re.I)
_PAINT = re.compile(r'((?:fill|stroke|stop-color)=[\"\'])(#[0-9a-fA-F]{3,6})([\"\'])')
_LOCK = threading.RLock()


def plot_color(value: str) -> str:
    """Lift series lightness, preserving hue and at least 5:1 panel contrast."""
    from matplotlib.colors import to_rgb, to_hex

    def luminance(rgb):
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb]
        return sum(c * w for c, w in zip(linear, (.2126, .7152, .0722)))

    h, light, sat = colorsys.rgb_to_hls(*to_rgb(value))
    light = max(light, .72 if sat > .1 else .76)
    background = luminance(to_rgb("#1b2839"))
    while light < 1:
        rgb = colorsys.hls_to_rgb(h, light, sat)
        if (luminance(rgb) + .05) / (background + .05) >= 5:
            break
        light = min(1, light + .01)
    return to_hex(colorsys.hls_to_rgb(h, light, sat))


def native_line_palette_css() -> str:
    """Dark overrides for Streamlit's categorical line charts and their keys.

    Restrict this to line charts: continuous color scales and heatmaps must not
    inherit a categorical palette. The caller supplies the theme/media scope.
    """
    colors = ("#0068c9", "#83c9ff", "#ff2b2b", "#ffabab", "#29b09d",
              "#7defa1", "#ff8700", "#ffd16a", "#6d3fc0", "#d5dae5")
    chart = '[data-testid="stVegaLiteChart"]:has(.mark-line.role-mark)'
    marks = ':is(.mark-line.role-mark,.mark-symbol.role-mark,.role-legend-symbol)'
    return "\n".join(
        f'{chart} {marks} [{prop}="{color}" i] {{{prop}:{plot_color(color)} !important;}}'
        for color in colors for prop in ("stroke", "fill")
    )


def _series_palette(svg: str, signal_ids: set[str], overlay_ids: set[str], selected: str) -> str:
    """Theme only exported data artists, including their marker definitions.

    Scalar-mapped collections never enter signal_ids. CSS fallbacks retain the
    original light palette, and the media query follows OS changes immediately.
    """
    if selected not in {"dark", "browser"} or not signal_ids:
        return svg
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg)
    palette = {}
    for group in root.iter():
        if group.get("id") not in signal_ids:
            continue
        for element in group.iter():
            style = element.get("style", "")
            def paint(match):
                color = match[2].lower()
                key = "--series-" + color[1:]
                if key not in palette:
                    palette[key] = plot_color(color)
                return f"{match[1]}: var({key},{color})"
            style = re.sub(r"(fill|stroke):\s*(#[0-9a-fA-F]{6})\b", paint, style)
            if style:
                element.set("style", style)
    scope = "series-" + uuid.uuid4().hex
    root.set("class", scope)
    rules = ";".join(f"{key}:{value}" for key, value in sorted(palette.items()))
    css = f".{scope}{{{rules}}}"
    if overlay_ids:
        # A bright arrow also needs an edge where it crosses a pale heatmap
        # cell. Outline only the overlays; never filter the underlying pixels.
        namespace = "{http://www.w3.org/2000/svg}"
        outline_id = scope + "-outline"
        outline = ET.SubElement(root, namespace + "filter", {
            "id": outline_id, "filterUnits": "userSpaceOnUse",
            "x": "-10%", "y": "-10%", "width": "120%", "height": "120%",
        })
        alpha = ET.SubElement(outline, namespace + "feComponentTransfer", {"in": "SourceAlpha"})
        ET.SubElement(alpha, namespace + "feFuncA", {"type": "linear", "slope": "3"})
        ET.SubElement(outline, namespace + "feMorphology", {"operator": "dilate", "radius": ".6", "result": "edge"})
        ET.SubElement(outline, namespace + "feFlood", {"flood-color": "#101926"})
        ET.SubElement(outline, namespace + "feComposite", {"in2": "edge", "operator": "in"})
        merge = ET.SubElement(outline, namespace + "feMerge")
        ET.SubElement(merge, namespace + "feMergeNode")
        ET.SubElement(merge, namespace + "feMergeNode", {"in": "SourceGraphic"})
        css += ",".join(f".{scope} [id='{gid}']" for gid in sorted(overlay_ids))
        css += f"{{filter:url(#{outline_id})}}"
    if selected == "browser":
        css = f"@media (prefers-color-scheme: dark){{{css}}}"
    style = ET.Element("{http://www.w3.org/2000/svg}style")
    style.text = css
    root.insert(0, style)
    return ET.tostring(root, encoding="unicode")


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
        if re.match(r'<svg\b[^>]*\bclass=[\"\'][^\"\']*\bailab-plot\b', svg):
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


def responsive_diagrams(body: str) -> str:
    """Pair untouched desktop SVGs with vertically arranged phone diagrams."""
    from gui.mobile_diagrams import mobile_diagram

    def adapt(match):
        svg = match.group()
        try:
            root = ET.fromstring(svg)
            if "ailab-plot" in root.get("class", "").split():
                return svg
            bounds = [float(v) for v in root.get("viewBox", "").replace(",", " ").split()]
            if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds) or min(bounds[2:]) <= 0:
                return svg
        except (ET.ParseError, ValueError):
            return svg
        if not any(e.tag.rsplit("}", 1)[-1] == "text" for e in root.iter()):
            return svg
        return ('<div class="ailab-diagram"><div class="ailab-diagram-desktop">'
                f'{svg}</div>{mobile_diagram(root)}</div>')

    return _SVG.sub(adapt, body)


def _legend_rows(fig, legends, temporary):
    """Lay legends below all plot ink, wrapping into rows within figure width.

    The source legends are hidden by the caller. Separate figure-level rows
    avoid covering bars, reference lines, labels, or another subplot. They are
    included by bbox_inches='tight' without shrinking any data axes.
    """
    if not legends:
        return
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    available = fig.bbox.width * .94
    bottom = fig.get_tightbbox(renderer).y0 * fig.dpi - 10 * fig.dpi / 72
    for index, (ax, original) in enumerate(legends):
        labels = [text.get_text() for text in original.get_texts()]
        font = original.get_texts()[0].get_fontproperties().copy()
        # Wrap prose without breaking mathematical expressions or long words.
        width = max(16, int(available / (font.get_size_in_points() * fig.dpi / 72 * .62)) - 7)
        labels = [textwrap.fill(label, width, break_long_words=False, break_on_hyphens=False)
                  if '$' not in label else label for label in labels]
        title = original.get_title().get_text()
        if len(legends) > 1 and ax is not None:
            title = ax.get_title() or title or f"Plot {index + 1}"
        if '$' not in title:
            title = textwrap.fill(title, width, break_long_words=False, break_on_hyphens=False)
        for columns in range(min(4, len(labels)), 0, -1):
            row = fig.legend(original.legend_handles, labels, title=title or None,
                             prop=font, title_fontsize=font.get_size_in_points(),
                             loc="upper center", bbox_to_anchor=(.5, bottom / fig.bbox.height),
                             bbox_transform=fig.transFigure, ncols=columns,
                             borderaxespad=0, frameon=False,
                             handlelength=original.handlelength,
                             handler_map=original.get_legend_handler_map())
            temporary.append(row)
            bounds = row.get_window_extent(renderer)
            if bounds.width <= available or columns == 1:
                bottom = bounds.y0 - 10 * fig.dpi / 72
                break
            row.remove()
            temporary.pop()


def figure_svg(fig, *, selected="light", **save_options) -> str:
    """Export a sharp plot with themed furniture and readable series colors.

    No global rcParams or per-visitor Matplotlib style is installed. Restore the
    supplied figure after export, including annotations and legend backgrounds.
    """
    from matplotlib.text import Text
    from matplotlib.colors import to_hex
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    saved = []
    temporary_legends = []
    signal_ids = set()
    overlay_ids = set()
    tokens = {"#010203": "var(--ailab-panel)", "#020304": "var(--ailab-ink)",
              "#030405": "var(--ailab-muted)", "#040506": "var(--ailab-border)"}
    def set_prop(artist, prop, value):
        saved.append((artist, prop, getattr(artist, "get_" + prop)()))
        getattr(artist, "set_" + prop)(value)

    with _LOCK:
        canvas = fig.canvas
        try:
            # Detached/copied figures can have a base canvas with no renderer.
            # Legend measurements need Agg; never leave the caller's canvas changed.
            if not hasattr(canvas, "get_renderer"):
                FigureCanvasAgg(fig)
            legends = [(ax, ax.get_legend()) for ax in fig.axes if ax.get_legend() is not None]
            legends += [(None, legend) for legend in fig.legends]
            legends = [(ax, legend) for ax, legend in legends
                       if legend.get_visible() and legend.get_texts()]
            for _, legend in legends:
                set_prop(legend, "visible", False)
            _legend_rows(fig, legends, temporary_legends)
            set_prop(fig.patch, "facecolor", "#010203")
            for ax in fig.axes:
                # Mark series and annotation arrows independently of plot
                # furniture. Never remap colorbars or scalar-mapped collections.
                if not hasattr(ax, "_colorbar"):
                    signals = list(ax.lines) + list(ax.patches)
                    signals += [c for c in ax.collections if c.get_array() is None
                                and type(c).__name__ not in {"QuadMesh", "QuadContourSet", "TriContourSet"}]
                    signals += [t.arrow_patch for t in ax.texts if getattr(t, "arrow_patch", None) is not None]
                    legend = ax.get_legend()
                    if legend:
                        signals += list(legend.legend_handles)
                    for artist in signals:
                        if artist is None:
                            continue
                        gid = "signal-" + uuid.uuid4().hex
                        signal_ids.add(gid)
                        if ax.images or any(type(c).__name__ in {"QuadMesh", "QuadContourSet", "TriContourSet"}
                                            for c in ax.collections):
                            overlay_ids.add(gid)
                        set_prop(artist, "gid", gid)
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
            for legend in temporary_legends:
                for text in legend.findobj(Text):
                    set_prop(text, "color", "#020304")
                for handle in legend.legend_handles:
                    if handle is not None:
                        gid = "signal-" + uuid.uuid4().hex
                        signal_ids.add(gid)
                        set_prop(handle, "gid", gid)
            buf = io.StringIO()
            options = {"bbox_inches": "tight", **save_options, "format": "svg"}
            options.pop("dpi", None)
            with rc_context({"svg.fonttype": "none"}):
                fig.savefig(buf, **options)
        finally:
            for legend in temporary_legends:
                legend.remove()
            for artist, prop, value in reversed(saved):
                getattr(artist, "set_" + prop)(value)
            fig.set_canvas(canvas)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    # Namespace IDs: independent figures often have identical clip/marker IDs.
    prefix = "plot" + uuid.uuid4().hex + "_"
    svg = re.sub(r'(?<=id=")([^"\s]+)', lambda m: prefix + m[1], svg)
    svg = re.sub(r'(?<=url\(#)([^)]+)', lambda m: prefix + m[1], svg)
    svg = re.sub(r'(?<=href="#)([^"]+)', lambda m: prefix + m[1], svg)
    for source, target in tokens.items():
        svg = svg.replace(source, target)
    svg = _series_palette(svg, {prefix + gid for gid in signal_ids},
                          {prefix + gid for gid in overlay_ids}, selected)
    if 'class="series-' in svg:
        svg = svg.replace('class="series-', 'class="ailab-plot series-', 1)
    else:
        svg = svg.replace("<svg ", '<svg class="ailab-plot" ', 1)
    svg = svg.replace("<svg ", '<svg role="img" aria-label="Lesson plot" ', 1)
    svg = re.sub(r'\s(?:width|height)="[^\"]+pt"', '', svg, count=2)
    return svg.replace("<svg ", '<svg style="display:block;width:100%;height:auto" ', 1)


def _compact_layout(compact):
    """Size and label an owned, single-panel figure for a phone."""
    from matplotlib.text import Text
    from matplotlib.ticker import AutoLocator, FixedLocator, MaxNLocator, ScalarFormatter

    compact.set_size_inches(3.25, max(2.8, min(3.5, compact.get_figheight())))
    ax = compact.axes[0]
    numeric_ticks = isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
    for text in compact.findobj(Text):
        text.set_fontsize(max(12, text.get_fontsize()))
    ax.title.set_fontsize(max(14, ax.title.get_fontsize()))
    for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
        if "$" not in text.get_text():
            text.set_text(textwrap.fill(text.get_text(), 26, break_long_words=False,
                                        break_on_hyphens=False))
    if type(ax.xaxis.get_major_locator()) is AutoLocator:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    elif numeric_ticks and isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks(), nbins=4))
    legends = [ax.get_legend()] if ax.get_legend() is not None else []
    legends += compact.legends
    visibility = [legend.get_visible() for legend in legends]
    for legend in legends:
        legend.set_visible(False)
    compact.tight_layout(pad=.9)
    compact.set_layout_engine(None)
    for legend, visible in zip(legends, visibility):
        legend.set_visible(visible)


def _compact_figure(fig):
    """Reflow simple numeric plots on a copy, preserving the source figure."""
    from matplotlib.ticker import AutoLocator, ScalarFormatter

    if len(fig.axes) != 1:
        return None
    ax = fig.axes[0]
    numeric_ticks = isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter)
    if (ax.name != "rectilinear" or not ax.axison or ax.images or ax.texts
            or ax.get_subplotspec() is None
            or (not isinstance(ax.xaxis.get_major_locator(), AutoLocator)
                and not numeric_ticks and len(ax.get_xticks()) > 4)):
        return None
    compact = copy.deepcopy(fig)
    _compact_layout(compact)
    return compact


def _stacked_panels(fig, *, selected="light", **save_options):
    """Export independent subplots in reading order with local legend rows.

    A colorbar or a twin axis is part of its plot, not another teaching panel.
    Keep those composite figures together. All current multi-panel lessons use
    independent rectilinear subplots, including shared axes and image panels.
    """
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    axes = [ax for ax in fig.axes if ax.get_visible()]
    if (len(axes) < 2 or any(hasattr(ax, "_colorbar") or ax.name != "rectilinear"
                             or ax.get_subplotspec() is None for ax in axes)):
        return None
    specs = [ax.get_subplotspec() for ax in axes]
    if len(set(specs)) != len(specs):  # overlaid axes must stay overlaid
        return None
    # Use grid cells, not aspect-adjusted image boxes, to preserve row order.
    axes.sort(key=lambda ax: (-round(ax.get_subplotspec().get_position(fig).y1, 6),
                              ax.get_subplotspec().get_position(fig).x0))
    panels = []
    for index, source in enumerate(axes):
        panel = copy.deepcopy(fig)
        try:
            ax = panel.axes[fig.axes.index(source)]
            limits = source.get_xlim(), source.get_ylim()
            for other in list(panel.axes):
                if other is not ax:
                    panel.delaxes(other)
            ax.set_subplotspec(GridSpec(1, 1, figure=panel)[0])
            # Shared axes normally suppress inner tick labels. Each stacked
            # panel needs its own scale and the original shared limits.
            ax.tick_params(labelleft=True, labelbottom=True)
            for dimension in ("x", "y"):
                if not getattr(source, f"get_{dimension}label")():
                    siblings = getattr(source, f"get_shared_{dimension}_axes")().get_siblings(source)
                    label = next((getattr(other, f"get_{dimension}label")()
                                  for other in fig.axes if other in siblings
                                  and getattr(other, f"get_{dimension}label")()), "")
                    getattr(ax, f"set_{dimension}label")(label)
            ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
            if index:
                # Figure-wide headings/keys appear once, with the first panel.
                for text in panel.texts:
                    text.set_text("")
                for legend in list(panel.legends):
                    legend.remove()
            _compact_layout(panel)
            svg = figure_svg(panel, selected=selected, **save_options)
            label = html.escape(f"Panel {index + 1}: {source.get_title() or 'Lesson plot'}", quote=True)
            svg = svg.replace('aria-label="Lesson plot"', f'aria-label="{label}"', 1)
            panels.append(f'<div class="ailab-plot-panel">{svg}</div>')
        finally:
            plt.close(panel)
    return "".join(panels)


def responsive_figure_html(fig, *, selected="light", **save_options) -> str:
    """Keep mobile text at least 14 CSS px instead of shrinking a whole chart.

    Simple charts get a narrow layout with larger text. Independent subplots
    stack vertically; composite diagrams retain their geometry and can scroll.
    CSS selects the variant without rerunning simulations on viewport changes.
    """
    from matplotlib.text import Text
    import matplotlib.pyplot as plt

    with _LOCK:
        desktop = figure_svg(fig, selected=selected, **save_options)
        stacked = _stacked_panels(fig, selected=selected, **save_options)
        if stacked is not None:
            return (f'<div class="ailab-responsive-plot"><div class="ailab-plot-desktop">{desktop}</div>'
                    '<div class="ailab-plot-mobile ailab-plot-stack" role="group" '
                    f'aria-label="Chart panels in reading order">{stacked}</div></div>')
        compact = _compact_figure(fig)
        try:
            mobile_fig = compact if compact is not None else fig
            mobile = figure_svg(compact, selected=selected, **save_options) if compact is not None else desktop
            fonts = [t.get_fontsize() for t in mobile_fig.findobj(Text)
                     if t.get_visible() and t.get_text().strip()]
            width = float(re.search(r'viewBox="[^"]*?([\d.]+) [\d.]+"', mobile)[1])
            minimum = math.ceil(width * 14 / min(fonts, default=12))
        finally:
            if compact is not None:
                plt.close(compact)
    # Keep IDs unique even when both variants reuse the same exported geometry.
    if compact is None:
        mobile = re.sub(r'plot([a-f0-9]+)_', r'mobileplot\1_', mobile)
        mobile = mobile.replace("series-", "mobile-series-")
    return (f'<div class="ailab-responsive-plot"><div class="ailab-plot-desktop">{desktop}</div>'
            f'<div class="ailab-plot-mobile" style="--ailab-plot-width:{minimum}px">'
            '<div class="ailab-plot-scroll" role="region" tabindex="0" '
            'aria-label="Lesson plot; use left and right arrow keys to explore wide plots">'
            f'{mobile}</div><p class="ailab-plot-hint">↔ Swipe across wider plots to see every label.</p>'
            '</div></div>')


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
                body = responsive_diagrams(body)
            return original(self, body, *args, **kwargs)
        return render

    DeltaGenerator.markdown = markdown_wrapper(DeltaGenerator.markdown)
    st.markdown = st._main.markdown

    def pyplot(self, fig=None, clear_figure=None, *, width="stretch", use_container_width=None, **kwargs):
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()
            clear_figure = True if clear_figure is None else clear_figure
        rendered = self.markdown(responsive_figure_html(fig, selected=st.session_state.get("theme", "browser"), **kwargs), unsafe_allow_html=True,
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
        chart = chart.configure_axis(labelFontSize=14, titleFontSize=14).configure_legend(
            labelFontSize=14, titleFontSize=14)
        meta = chart.usermeta if isinstance(chart.usermeta, dict) else {}
        chart.usermeta = {**meta, "embedOptions": {**meta.get("embedOptions", {}), "renderer": "svg"}}
        return original_chart(self, chart, *args, **kwargs)
    VegaChartsMixin._altair_chart = svg_chart
    st._ailab_visuals_installed = True

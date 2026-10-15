"""Guard scientific color semantics and the shared theme rendering boundary."""
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gui.theme_visuals import diagram_html, figure_svg, plot_color


@pytest.mark.parametrize('color', ['#185fa5', '#1d6fb8', '#c0507a', '#1d9e75', '#a32d2d', '#000000', '#0000ff'])
def test_series_palette_has_clear_contrast_against_dark_panels(color):
    from matplotlib.colors import to_rgb
    def luminance(value):
        channels = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in to_rgb(value)]
        return sum(v * w for v, w in zip(channels, (.2126, .7152, .0722)))
    assert (luminance(plot_color(color)) + .05) / (luminance('#1b2839') + .05) >= 4.95


def test_dark_series_arrows_markers_and_legend_share_a_palette_and_restore_artists():
    fig, ax = plt.subplots()
    line, = ax.plot([0, 1], [0, 1], 'o-', color='#185fa5', label='blue series', gid='original-line')
    arrow = ax.annotate('', xy=(1, 0), xytext=(0, 0),
                        arrowprops={'arrowstyle': '-|>', 'color': '#c0507a'})
    points = ax.scatter([.5], [.25], color='#1d9e75')
    ax.legend()
    before = (line.get_gid(), arrow.arrow_patch.get_gid(), points.get_gid())
    light = figure_svg(fig, selected='light')
    dark = figure_svg(fig, selected='dark')
    browser = figure_svg(fig, selected='browser')
    ET.fromstring(dark)
    assert '--series-' not in light
    assert '@media (prefers-color-scheme: dark)' in browser
    assert '@media' not in dark
    # Both the plotted series/marker and its legend key use the same variable.
    assert dark.count('var(--series-185fa5,#185fa5)') >= 4
    assert 'var(--series-c0507a,#c0507a)' in dark
    assert 'var(--series-1d9e75,#1d9e75)' in dark
    assert before == (line.get_gid(), arrow.arrow_patch.get_gid(), points.get_gid())
    assert line.get_color() == '#185fa5'
    assert diagram_html(dark, 'dark') == dark
    plt.close(fig)


def test_dark_palette_preserves_scalar_collections_and_image_pixels():
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [1, 0]], cmap='Blues')
    mapped = ax.scatter([0, 1], [0, 1], c=[0., 1.], cmap='Blues', gid='scalar-data')
    fig.colorbar(mapped)
    ax.plot([0, 1], [1, 0], color='#185fa5')
    light = ET.fromstring(figure_svg(fig, selected='light'))
    dark = ET.fromstring(figure_svg(fig, selected='dark'))
    def scalar_paints(root):
        group = next(e for e in root.iter() if e.get('id', '').endswith('scalar-data'))
        return [e.get('style') for e in group.iter()]
    def image_pixels(root):
        return [e.get('{http://www.w3.org/1999/xlink}href') for e in root.iter()
                if e.tag.endswith('}image')]
    assert scalar_paints(light) == scalar_paints(dark)
    assert image_pixels(light) == image_pixels(dark)
    css = next(e.text for e in dark.iter() if e.tag.endswith('}style'))
    assert 'filter:url(' in css and 'scalar-data' not in css
    assert not any(e.tag.endswith('}filter') for e in light.iter())
    plt.close(fig)


def test_native_line_palette_only_applies_in_dark_themes():
    from gui.ui import _theme_overrides
    assert 'stroke="#0068c9"' in _theme_overrides('dark')
    assert 'stroke="#0068c9"' not in _theme_overrides('light')
    assert 'stroke="#0068c9"' not in _theme_overrides('accessible')
    assert '@media (prefers-color-scheme: dark)' in _theme_overrides('browser')
    assert ':has(.mark-line.role-mark)' in _theme_overrides('dark')


def test_diagrams_follow_browser_and_explicit_choices_without_changing_prose():
    body = 'The code #33312E stays literal. <svg viewBox="0 0 10 10"><rect fill="#FAFAF7"/><text fill="#33312E">x</text></svg>'
    assert diagram_html(body, 'light') == body
    assert diagram_html(body, 'accessible') == body
    dark = diagram_html(body, 'dark')
    assert 'The code #33312E stays literal.' in dark
    assert 'fill="var(--diagram-fafaf7,#fafaf7)"' in dark
    assert 'prefers-color-scheme' not in dark
    assert '@media (prefers-color-scheme: dark)' in diagram_html(body, 'browser')


def test_plot_preserves_data_colors_and_restores_the_supplied_figure():
    fig, axes = plt.subplots(1, 2)
    line, = axes[0].plot([1, 2], [2, 1], color='#185FA5', label='series')
    axes[0].set_title('A label')
    axes[0].legend()
    im = axes[1].imshow(np.array([[0., 1.], [1., 0.]]), cmap='Blues', vmin=0, vmax=1)
    cell = axes[1].text(0, 0, '0.00', color='#333333')
    before = (fig.get_facecolor(), axes[0].get_facecolor(), axes[0].title.get_color(), cell.get_color())
    svg = figure_svg(fig)
    ET.fromstring(svg)
    assert 'var(--ailab-panel)' in svg and 'var(--ailab-ink)' in svg
    assert '#185fa5' in svg  # colored series is not inverted or remapped
    assert '#333333' in svg  # label over a light heatmap cell stays dark
    assert 'data:image/png;base64,' in svg  # colormap pixels are preserved
    assert before == (fig.get_facecolor(), axes[0].get_facecolor(), axes[0].title.get_color(), cell.get_color())
    assert line.get_color() == '#185FA5' and im.get_clim() == (0, 1)
    assert diagram_html(svg, 'dark') == svg
    # Two figures must not share clip-path/marker IDs on the same lesson page.
    other = figure_svg(fig)
    ids = lambda text: {e.attrib['id'] for e in ET.fromstring(text).iter() if 'id' in e.attrib}
    assert not ids(svg) & ids(other)
    plt.close(fig)


def test_native_chart_stays_interactive_and_uses_svg(tmp_path):
    from streamlit.testing.v1 import AppTest
    from gui import theme_visuals
    theme_visuals.install()
    app = AppTest.from_string('''
import streamlit as st
import pandas as pd
st.line_chart(pd.DataFrame({'a':[1,2], 'b':[2,1]}))
st.dataframe(pd.DataFrame({'x':[1,2]}))
''').run()
    assert not app.exception
    import json
    spec = json.loads(app.get('vega_lite_chart')[0].proto.spec)
    assert spec['usermeta']['embedOptions']['renderer'] == 'svg'
    assert spec.get('params') or any(layer.get('params') for layer in spec.get('layer', []))  # native pan/zoom selection survives
    assert app.dataframe  # keep native sorting, resizing and download controls

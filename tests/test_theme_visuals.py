"""Guard scientific color semantics and the shared theme rendering boundary."""
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from gui.theme_visuals import diagram_html, figure_svg


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

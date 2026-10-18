"""Guard scientific color semantics and the shared theme rendering boundary."""
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gui.theme_visuals import diagram_html, figure_svg, plot_color, responsive_diagrams, responsive_figure_html


@pytest.mark.parametrize('selected', ['light', 'dark', 'browser'])
def test_mobile_diagram_preserves_geometry_and_scales_inherited_labels(selected):
    svg = '''<svg viewBox="0 0 600 200" width="100%" role="img" aria-label="Chain rule">
    <defs><marker id="arrow"><path d="M0,0 L5,3 L0,6"/></marker></defs>
    <g font-size="12" fill="#172033"><text x="20" y="30">Input</text>
    <g style="font-size:75%"><text x="150" y="80">Gradient<tspan font-size="6">1</tspan></text></g></g>
    <path d="M50,50 L150,50" stroke="#172033" marker-end="url(#arrow)"/></svg>'''
    themed = diagram_html(svg, selected)
    result = responsive_diagrams('Before' + themed + 'After')
    # Preserve the SVG verbatim: connected arrows, subscripts, colors, accessible
    # title and desktop geometry must not be reconstructed or reflowed.
    assert themed in result or themed.split('</style>')[-1] in result
    assert result.startswith('Before') and result.endswith('After')
    root = ET.fromstring('<div>' + result + '</div>')
    wrapper = root.find('div[@class="ailab-diagram"]')
    minimum = int(wrapper.get('style').split(':')[1].removesuffix('px'))
    assert minimum * 9 / 600 >= 14
    assert minimum * 9 / 600 < 14.02
    region = wrapper.find('div[@role="region"]')
    assert region.get('tabindex') == '0' and 'arrow keys' in region.get('aria-label')
    assert region.find('svg').get('aria-label') == 'Chain rule'


@pytest.mark.parametrize('font', ['10px', '7.5pt', '.625em', '62.5%'])
def test_mobile_diagram_font_units_produce_same_readable_scale(font):
    svg = f'<svg viewBox="0 0 400 100"><text style="font-size:{font}">Weight</text></svg>'
    root = ET.fromstring(responsive_diagrams(svg))
    assert root.get('style') == '--ailab-diagram-width:560px'


def test_diagram_adapter_leaves_plots_icons_and_unparseable_content_alone():
    fragments = [
        '<svg class="ailab-plot" viewBox="0 0 400 100"><text>Axis</text></svg>',
        '<svg viewBox="0 0 20 20"><path d="M0,0 L20,20"/></svg>',
        '<svg><text>No view box</text></svg>',
        '<svg viewBox="0 0 400 100"><text>Invalid & XML</text></svg>',
        '<p>Plain lesson prose</p>',
    ]
    for fragment in fragments:
        assert responsive_diagrams(fragment) == fragment


@pytest.mark.parametrize('selected', ['light', 'dark'])
def test_mobile_plot_has_readable_text_without_changing_source_data_or_layout(selected):
    fig, ax = plt.subplots(figsize=(4.7, 4))
    line, = ax.plot([0, 1, 2], [1, 3, 2], label='Measured probability')
    ax.set(title='Probability distribution', xlabel='Outcome', ylabel='Probability')
    ax.legend(fontsize=8)
    fig.tight_layout()
    before = (tuple(fig.get_size_inches()), ax.get_position().bounds,
              ax.xaxis.get_major_locator(), ax.get_xlim(), ax.get_ylim(),
              ax.title.get_fontsize(), ax.get_legend().get_texts()[0].get_fontsize())
    root = ET.fromstring(responsive_figure_html(fig, selected=selected))
    variants = root.findall('.//{http://www.w3.org/2000/svg}svg')
    assert len(variants) == 2
    mobile = variants[1]
    import re
    minimum = float(re.search(r'--ailab-plot-width:(\d+)px', ET.tostring(root, encoding='unicode'))[1])
    width = float(mobile.get('viewBox').split()[2])
    text = mobile.findall('.//{http://www.w3.org/2000/svg}text')
    fonts = [float(re.search(r'font-size: ([\d.]+)px', t.get('style', ''))[1]) for t in text]
    # Even a narrow 320px phone keeps labels and legends >=14px, and this
    # ordinary single-panel chart fits without needing horizontal panning.
    assert min(fonts) * max(292, minimum) / width >= 14
    assert minimum <= 292
    assert tuple(fig.get_size_inches()) == before[0]
    assert (ax.get_position().bounds, ax.xaxis.get_major_locator(), ax.get_xlim(), ax.get_ylim(),
            ax.title.get_fontsize(), ax.get_legend().get_texts()[0].get_fontsize()) == before[1:]
    assert list(line.get_xdata()) == [0, 1, 2] and list(line.get_ydata()) == [1, 3, 2]
    assert not fig.legends and ax.get_legend().get_visible()
    plt.close(fig)


def test_colorbar_stays_with_its_plot_and_uses_independent_svg_ids():
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow([[0, 1], [1, 0]])
    ax.plot([0, 1], [1, 0], label='Direction')
    ax.legend(fontsize=8)
    fig.colorbar(im, ax=ax)
    root = ET.fromstring(responsive_figure_html(fig, selected='dark'))
    svgs = root.findall('.//{http://www.w3.org/2000/svg}svg')
    assert svgs[0].get('viewBox') == svgs[1].get('viewBox')
    paths = lambda svg: [p.get('d') for p in svg.findall('.//{http://www.w3.org/2000/svg}path')]
    assert paths(svgs[0]) == paths(svgs[1])
    ids = [element.get('id') for element in root.iter() if element.get('id')]
    assert len(ids) == len(set(ids))  # clip paths, markers AND overlay filters
    region = root.find('.//div[@role="region"]')
    assert region.get('tabindex') == '0'
    assert 'arrow keys' in region.get('aria-label')
    plt.close(fig)


def test_detached_multi_panel_figure_exports_without_a_canvas_renderer():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 3))
    for ax in fig.subplots(1, 2):
        ax.plot([0, 1], [1, 0], label='Descent')
        ax.legend()
    canvas = fig.canvas
    assert not hasattr(canvas, 'get_renderer')
    root = ET.fromstring(responsive_figure_html(fig, selected='dark'))
    assert len(root.findall('.//{http://www.w3.org/2000/svg}svg')) == 3
    assert fig.canvas is canvas
    assert all(ax.get_legend().get_visible() for ax in fig.axes)


@pytest.mark.parametrize('selected', ['light', 'dark'])
@pytest.mark.parametrize('shape,images', [((1, 2), False), ((1, 3), False), ((2, 2), True)])
def test_mobile_panels_stack_in_reading_order_preserving_data_scales_and_desktop(shape, images, selected, monkeypatch):
    from gui import theme_visuals
    fig, axes = plt.subplots(*shape, figsize=(9.2, 3.4), sharey=True)
    axes = np.asarray(axes).ravel()
    for i, ax in enumerate(axes):
        ax.set(title=f'Panel {i + 1}', xlabel='Input')
        ax.plot([0, 1], [i, i + 1], label=f'Series {i + 1}')
        ax.legend(fontsize=7.5)
        if images and i % 2 == 0:
            ax.imshow([[0, 1], [2, 3]], cmap='Blues', vmin=0, vmax=3)
    axes[0].set_ylabel('Shared output')
    fig.tight_layout()
    fig.canvas.draw()
    source = [(ax.get_position().bounds, ax.get_xlim(), ax.get_ylim()) for ax in axes]
    before = ET.fromstring(figure_svg(fig, selected=selected))
    original_export = theme_visuals.figure_svg
    exported = []
    def inspect_panel(figure, **kwargs):
        if figure is not fig:
            assert len(figure.axes) == 1
            ax = figure.axes[0]
            i = len(exported)
            assert ax.get_xlim() == source[i][1] and ax.get_ylim() == source[i][2]
            assert ax.get_ylabel() == 'Shared output'
            assert any(t.get_visible() for t in ax.get_yticklabels())
            assert list(ax.lines[0].get_ydata()) == [i, i + 1]
            if images and i % 2 == 0:
                assert np.array_equal(ax.images[0].get_array(), [[0, 1], [2, 3]])
                assert ax.images[0].get_clim() == (0, 3)
            exported.append(ax.get_title())
        return original_export(figure, **kwargs)
    monkeypatch.setattr(theme_visuals, 'figure_svg', inspect_panel)
    root = ET.fromstring(responsive_figure_html(fig, selected=selected))
    svgs = root.findall('.//{http://www.w3.org/2000/svg}svg')
    assert len(svgs) == 1 + len(axes)
    assert exported == [f'Panel {i + 1}' for i in range(len(axes))]
    paths = lambda svg: [p.get('d') for p in svg.findall('.//{http://www.w3.org/2000/svg}path')]
    assert paths(svgs[0]) == paths(before) and svgs[0].get('viewBox') == before.get('viewBox')
    for i, svg in enumerate(svgs[1:]):
        labels = ''.join(svg.itertext())
        assert f'Series {i + 1}' in labels
        assert all(f'Series {j + 1}' not in labels for j in range(len(axes)) if j != i)
        assert float(svg.get('viewBox').split()[2]) < 260  # readable at 320px, without panning
    assert not root.findall('.//div[@class="ailab-plot-scroll"]')
    ids = [e.get('id') for e in root.iter() if e.get('id')]
    assert len(ids) == len(set(ids))
    assert [(ax.get_position().bounds, ax.get_xlim(), ax.get_ylim()) for ax in axes] == source
    assert len(axes[0].get_shared_y_axes().get_siblings(axes[0])) == len(axes)
    plt.close(fig)


@pytest.mark.parametrize('example', ['bernoulli', 'distances', 'subplots'])
def test_legend_rows_do_not_cover_data_or_axis_labels_and_restore_the_figure(example, monkeypatch):
    fig, axes = plt.subplots(1, 2 if example == 'subplots' else 1,
                             figsize=(7.2 if example == 'subplots' else 4.7, 3.2))
    axes = np.atleast_1d(axes)
    for index, ax in enumerate(axes):
        if example == 'bernoulli':
            ax.bar([0, 1], [.5, .5], color='#185fa5')
            ax.axvline(.5, color='#c0507a', label='mean = 0.50')
            ax.axvspan(0, 1, color='#c0507a', alpha=.08, label='±1 std')
        else:
            ax.hist([1, 1, 2, 2, 2, 3], color='#185fa5')
            ax.axvline(.9, color='#c0507a', label='min 0.90')
            ax.axvline(9.63, color='#1d9e75', label='max 9.63')
        ax.set(xlabel='pairwise distance' if example != 'bernoulli' else 'outcome',
               ylabel='count', title=f'Distribution {index + 1}')
        ax.legend(frameon=example != 'distances', fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    source = [(ax.get_legend(), ax.get_legend().get_bbox_to_anchor().bounds,
               ax.get_position().bounds) for ax in axes]
    save = fig.savefig
    observed = []
    def checked_save(*args, **kwargs):
        fig.draw_without_rendering()
        renderer = fig.canvas.get_renderer()
        data_bottom = min(ax.get_tightbbox(renderer).y0 for ax in axes)
        rows = [legend.get_window_extent(renderer) for legend in fig.legends]
        assert len(rows) == len(axes)
        assert all(row.y1 < data_bottom for row in rows)
        assert all(row.x0 >= 0 and row.x1 <= fig.bbox.width for row in rows)
        assert all(not a.overlaps(b) for i, a in enumerate(rows) for b in rows[i + 1:])
        assert all(not legend.get_visible() for legend, _, _ in source)
        observed.append(rows)
        return save(*args, **kwargs)
    monkeypatch.setattr(fig, 'savefig', checked_save)
    for selected in ('light', 'dark'):
        svg = figure_svg(fig, selected=selected)
        ET.fromstring(svg)
        assert not fig.legends
        for ax, (legend, anchor, position) in zip(axes, source):
            assert ax.get_legend() is legend and legend.get_visible()
            assert legend.get_bbox_to_anchor().bounds == anchor
            assert ax.get_position().bounds == position
    assert len(observed) == 2
    plt.close(fig)


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
''', default_timeout=15).run()
    assert not app.exception
    import json
    spec = json.loads(app.get('vega_lite_chart')[0].proto.spec)
    assert spec['usermeta']['embedOptions']['renderer'] == 'svg'
    assert spec['config']['axis']['labelFontSize'] == 14
    assert spec['config']['legend']['labelFontSize'] == 14
    assert spec.get('params') or any(layer.get('params') for layer in spec.get('layer', []))  # native pan/zoom selection survives
    assert app.dataframe  # keep native sorting, resizing and download controls

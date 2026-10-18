"""Phone layouts for lesson flows, comparisons and spatial illustrations.

Layout metadata groups the source SVG's labels by their teaching meaning. Live
values still come from the SVG on every render; no numeric results are copied
into the metadata. Desktop SVGs never pass through this renderer.
"""
from __future__ import annotations

import copy
import colorsys
import functools
import html
import json
from pathlib import Path
import re
import uuid
import xml.etree.ElementTree as ET


@functools.lru_cache(maxsize=1)
def layouts():
    return {entry['label']: entry for entry in json.loads(
        (Path(__file__).parent / 'content' / 'mobile_diagrams.json').read_text(encoding='utf-8'))}


def _texts(root):
    return [e for e in root.iter() if e.tag.rsplit('}', 1)[-1] == 'text']


def _label_markup(element):
    """Keep color-coded tspan keys (for example blue/red class symbols)."""
    result = html.escape(element.text or '')
    for child in element:
        content = _label_markup(child)
        color = child.get('fill')
        if color:
            content = f'<span style="color:{html.escape(color, quote=True)}">{content}</span>'
        result += content + html.escape(child.tail or '')
    return result


def _graphic(root, bounds, selected, title):
    """Keep spatial data intact; replace small annotations with readable keys.

    Numbered callouts refer to the original positions. The full source labels
    appear below the image at normal text size. Independent comparison panels
    use authored view boxes, never inferred strips through connected diagrams.
    """
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    graphic = copy.deepcopy(root)
    x, y, width, height = bounds
    graphic.set('viewBox', ' '.join(str(v) for v in bounds))
    graphic.set('style', 'width:100%;max-width:100%;height:auto;display:block')
    graphic.set('width', '100%')
    graphic.attrib.pop('height', None)
    graphic.set('aria-label', title or root.get('aria-label', 'Lesson illustration'))
    # Pale strokes were used as ink in several old SVGs. Their dark palette
    # treated them like pale panel fills, making bowl curves almost disappear.
    # Only mobile stroke ink changes here; preserve fills and desktop colors.
    for element in graphic.iter():
        paint = re.search(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b', element.get('stroke', ''))
        if paint:
            value = paint[1]
            if len(value) == 3:
                value = ''.join(c * 2 for c in value)
            hue, light, saturation = colorsys.rgb_to_hls(*(int(value[i:i+2], 16) / 255 for i in (0, 2, 4)))
            if light > .78:
                role = ('muted' if saturation < .15 else
                        'coral' if hue < .08 or hue > .92 else
                        'amber' if hue < .18 else 'green' if hue < .47 else 'blue')
                element.set('stroke', f'var(--ailab-{role})')
    namespace = graphic.tag.removesuffix('svg')
    source_texts = _texts(graphic)
    parents = {child: parent for parent in graphic.iter() for child in parent}
    radius = width * 10 / 220
    placed = []
    for index, text in enumerate(source_texts):
        parent = parents[text]
        parent.remove(text)
        if index not in selected:
            continue
        # All lesson annotations use numeric x/y. Rotation on text is omitted
        # for upright keys; the underlying geometry keeps its transformations.
        px = float(text.get('x', '0').split()[0])
        py = float(text.get('y', '0').split()[0])
        if not (x <= px <= x + width and y <= py <= y + height):
            continue  # a shared caption belongs below the panel
        px = min(x + width - radius * 1.3, max(x + radius * 1.3, px))
        py = min(y + height - radius * 1.3, max(y + radius * 1.3, py))
        original = px, py
        # Keep nearby callouts separate, with a leader back to their anchor.
        for _ in range(len(source_texts) + 1):
            if all((px - a) ** 2 + (py - b) ** 2 >= (radius * 2.3) ** 2 for a, b in placed):
                break
            py += radius * 2.4
            if py > y + height - radius * 1.3:
                py = y + radius * 1.3
                px = min(x + width - radius * 1.3, px + radius * 2.4)
        placed.append((px, py))
        if original != (px, py):
            ET.SubElement(graphic, namespace + 'line', {
                'x1': str(original[0]), 'y1': str(original[1]), 'x2': str(px), 'y2': str(py),
                'stroke': 'var(--ailab-muted)', 'stroke-width': str(radius / 8),
            })
        ET.SubElement(graphic, namespace + 'circle', {
            'cx': str(px), 'cy': str(py), 'r': str(radius),
            'fill': 'var(--ailab-blue)', 'stroke': 'var(--ailab-panel)',
            'stroke-width': str(radius / 7),
        })
        marker = ET.SubElement(graphic, namespace + 'text', {
            'x': str(px), 'y': str(py), 'text-anchor': 'middle', 'dominant-baseline': 'central',
            'style': f'font:600 {width * 14 / 220}px sans-serif;fill:var(--ailab-panel,#fff)',
        })
        marker.text = str(selected.index(index) + 1)
    # Every panel has its own marker/gradient/clip IDs, including when desktop
    # and mobile coexist in the DOM. Keep the source theme class and variables.
    prefix = 'diagram-mobile-' + uuid.uuid4().hex + '-'
    ids = {e.get('id'): prefix + e.get('id') for e in graphic.iter() if e.get('id')}
    for element in graphic.iter():
        for key, value in list(element.attrib.items()):
            if key == 'id':
                value = ids[value]
            else:
                value = re.sub(r'url\(#([^)]*)\)', lambda m: 'url(#' + ids.get(m[1], m[1]) + ')', value)
                if value.startswith('#') and value[1:] in ids:
                    value = '#' + ids[value[1:]]
            element.set(key, value)
    return ET.tostring(graphic, encoding='unicode')


def mobile_diagram(root):
    source_labels = _texts(root)
    labels = [''.join(e.itertext()).strip() for e in source_labels]
    markup = [_label_markup(e) for e in source_labels]
    label = root.get('aria-label', 'Lesson illustration')
    layout = layouts().get(label)
    # New spatial illustrations still fit on phones. Known flows/comparisons
    # have explicit teaching order rather than an unreliable geometry heuristic.
    if layout is None or layout['text_count'] != len(labels):
        layout = {'kind': 'figure', 'sections': [{'title': '', 'labels': list(range(len(labels))),
                    'bounds': [float(v) for v in root.get('viewBox').replace(',', ' ').split()]}]}
    escape = html.escape
    sections = []
    used = set(layout.get('lead_labels', []))
    for section in layout['sections']:
        indices = section['labels']
        used.update(indices)
        title = section['title']
        parts = [f'<strong class="ailab-diagram-title">{escape(title)}</strong>'] if title else []
        if 'bounds' in section:
            parts.append(_graphic(root, section['bounds'], indices, title))
            parts.append('<ol class="ailab-diagram-key">' + ''.join(
                f'<li>{markup[i]}</li>' for i in indices) + '</ol>')
        elif layout.get('compact'):
            parts = ['<p>' + ' · '.join(markup[i] for i in indices) + '</p>']
        else:
            parts.append(''.join(f'<p>{markup[i]}</p>' for i in indices))
        sections.append('<li class="ailab-diagram-card">' + ''.join(parts) + '</li>')
    notes = [markup[i] for i in range(len(labels)) if i not in used]
    summary = layout.get('summary', '')
    scope = escape(root.get('class', ''), quote=True)
    return (f'<div class="ailab-diagram-mobile ailab-diagram-{layout["kind"]} {scope}" '
            f'role="group" aria-label="{escape(label, quote=True)}">'
            + ''.join(f'<p>{markup[i]}</p>' for i in layout.get('lead_labels', []))
            + (f'<p class="ailab-diagram-summary">{escape(summary)}</p>' if summary else '')
            + '<ol class="ailab-diagram-sections">' + ''.join(sections) + '</ol>'
            + (('<div class="ailab-diagram-notes">' + ''.join(f'<p>{n}</p>' for n in notes) + '</div>') if notes else '')
            + '</div>')

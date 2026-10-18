"""Responsive lesson tables without changing desktop data or formatting."""
from __future__ import annotations

import functools
import html
import math
import re


def _cells(line):
    """Split GFM rows, keeping escaped pipes inside their original cells."""
    result, start, slashes = [], 0, 0
    for i, character in enumerate(line):
        if character == '|' and slashes % 2 == 0:
            result.append(line[start:i].strip())
            start = i + 1
        slashes = slashes + 1 if character == '\\' else 0
    result.append(line[start:].strip())
    if line.lstrip().startswith('|'):
        result.pop(0)
    if not result[-1]:
        result.pop()
    return result


_LITERAL = re.compile(
    r'^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}(?P=fence)[^\n]*$'
    r'|`+[^`\n]*`+|\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+\$'
    r'|<(?:https?://|mailto:)[^>]+>|<[\w.+-]+@[\w.-]+>', re.M)
_HTML = re.compile(r'<!--[\s\S]*?-->|</?[A-Za-z][^>]*>|<![A-Z][^>]*>')
_ESCAPING = re.compile(f'(?P<literal>{_LITERAL.pattern})|(?P<tag>{_HTML.pattern})', re.M)


def _escape_raw_html(body):
    # Enabling our inline labels must not enable HTML that a caller originally
    # asked Streamlit to display as text. Leave code, math and autolinks alone.
    result, end = [], 0
    for match in _ESCAPING.finditer(body):
        result.append(body[end:match.start()])
        result.append(html.escape(match[0]) if match.group('tag') else match[0])
        end = match.end()
    result.append(body[end:])
    return ''.join(result)


def responsive_markdown_tables(body, *, unsafe_allow_html=False):
    """Add hidden inline column labels; CSS makes each row a phone card.

    Keep Markdown/KaTeX in the normal Streamlit renderer, including formulas,
    links and emphasis. Desktop headers, alignment and values are unchanged.
    """
    if '|' not in body or body.startswith('<div class="ailab-data-cards"'):
        return body, False
    from streamlit.string_util import clean_text
    lines = clean_text(body).splitlines()
    tables, i, fence = [], 0, None
    while i < len(lines) - 1:
        line = lines[i]
        marker = re.match(r'^ {0,3}(`{3,}|~{3,})', line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                fence = None
            i += 1
            continue
        if fence or line.startswith(('    ', '\t')) or '|' not in line:
            i += 1
            continue
        headers, divider = _cells(line), _cells(lines[i + 1])
        if len(headers) < 2 or len(headers) != len(divider) or not all(
                re.fullmatch(r':?-+:?', cell) for cell in divider):
            i += 1
            continue
        end = i + 2
        while end < len(lines) and lines[end].strip() and '|' in lines[end] and not lines[end].startswith(('    ', '\t')):
            end += 1
        if end > i + 2:
            tables.append((i, end, headers))
        i = end
    if not tables:
        return body, False
    if not unsafe_allow_html:
        lines = _escape_raw_html('\n'.join(lines)).splitlines()
    for start, end, _ in tables:
        headers = _cells(lines[start])
        for row in range(start + 2, end):
            cells = _cells(lines[row])
            # GFM pads short rows and ignores surplus cells. Retain the same
            # visible values and column associations as the desktop table.
            cells = (cells + [''] * len(headers))[:len(headers)]
            labelled = [f'<span class="ailab-table-label" aria-hidden="true">{header}</span>{cell}'
                        for header, cell in zip(headers, cells)]
            lines[row] = '| ' + ' | '.join(labelled) + ' |'
    return '\n'.join(lines), True


def _display(value):
    import pandas as pd
    if value is None or value is pd.NA or value is pd.NaT:
        return '—'
    if isinstance(value, float):
        return format(value, '.6g') if math.isfinite(value) else ('—' if math.isnan(value) else str(value))
    return str(value)


def dataframe_cards(frame, *, hide_index=None):
    """Read-only snapshots of the same rows and columns as the native grid."""
    escape = html.escape
    headers = [' / '.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in frame.columns]
    groups = []
    for start in range(0, len(frame), 10):
        cards = []
        for position in range(start, min(start + 10, len(frame))):
            fields = list(zip(headers, frame.iloc[position].tolist()))
            if hide_index is not True:
                index = frame.index[position]
                if isinstance(index, tuple):
                    fields = [(str(name or f'Index {i+1}'), value) for i, (name, value)
                              in enumerate(zip(frame.index.names, index))] + fields
                else:
                    fields.insert(0, (str(frame.index.name or 'Index'), index))
            cards.append(f'<article class="ailab-data-card" aria-label="Row {position + 1}">'
                         f'<strong class="ailab-data-card-title">Row {position + 1}</strong><dl>'
                         + ''.join(f'<div><dt>{escape(header)}</dt><dd>{escape(_display(value))}</dd></div>'
                                   for header, value in fields) + '</dl></article>')
        content = ''.join(cards)
        if len(frame) > 10:
            opened = ' open="open"' if start == 0 else ''
            content = (f'<details class="ailab-table-page"{opened}>'
                       f'<summary>Rows {start + 1}–{min(start + 10, len(frame))} of {len(frame)}</summary>'
                       f'{content}</details>')
        groups.append(content)
    if not groups:
        groups = ['<p>No rows to display.</p><p>Columns: ' + escape(', '.join(headers)) + '</p>']
    return '<div class="ailab-data-cards" role="group" aria-label="Table rows">' + ''.join(groups) + '</div>'


def install_dataframe_cards(DeltaGenerator):
    original = DeltaGenerator.dataframe

    @functools.wraps(original)
    def dataframe(self, data=None, *args, **kwargs):
        # Editable/selection-driven grids keep their native interaction model.
        if kwargs.get('on_select', 'ignore') != 'ignore' or kwargs.get('column_config'):
            return original(self, data, *args, **kwargs)
        from streamlit import dataframe_util
        if data is None or dataframe_util.is_pandas_styler(data):
            return original(self, data, *args, **kwargs)
        frame = dataframe_util.convert_anything_to_pandas_df(data, ensure_copy=True)
        if kwargs.get('column_order') is not None:
            frame = frame.loc[:, list(kwargs['column_order'])]
        # Delta paths are stable across reruns, including tables in columns/tabs.
        import hashlib
        key = hashlib.sha256(self._get_delta_path_str().encode()).hexdigest()[:16]
        outer = self.container(key='ailab-responsive-table-' + key)
        desktop = outer.container(key='ailab-table-desktop-' + key)
        result = original(desktop, data, *args, **kwargs)
        mobile = outer.container(key='ailab-table-mobile-' + key)
        mobile.markdown(dataframe_cards(frame, hide_index=kwargs.get('hide_index')), unsafe_allow_html=True)
        return result

    DeltaGenerator.dataframe = dataframe

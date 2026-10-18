"""Keep table values, mathematical Markdown and read-only grid behavior intact."""
import xml.etree.ElementTree as ET

import pandas as pd
from streamlit.testing.v1 import AppTest

from gui.mobile_tables import dataframe_cards, responsive_markdown_tables


def test_markdown_column_labels_preserve_formulas_links_alignment_and_escaped_pipes():
    source = r'''Before the table.

| **model** | $\varphi'(z)$ | explanation |
|:---|---:|---|
| ReLU | $1$ if $z>0$ | [reference](https://example.org) and `a\|b` |
| sigmoid | $p(1-p)$ | |

After the table.'''
    result, changed = responsive_markdown_tables(source)
    assert changed
    assert '|:---|---:|---|' in result
    assert result.count('aria-hidden="true"') == 6
    assert result.count('>$\\varphi\'(z)$</span>') == 2
    assert r'[reference](https://example.org) and `a\|b`' in result
    assert r'$p(1-p)$' in result
    assert 'Before the table.' in result and result.endswith('After the table.')


def test_labels_do_not_enable_original_html_or_change_code_math_and_autolinks():
    source = '''<img src="$x$" title="`code`" onerror="bad()">

| header | value |
|---|---|
| <b>literal</b> | `$x < y$` and $x < y$ |

<https://example.org>
'''
    result, changed = responsive_markdown_tables(source)
    assert changed
    assert '<img' not in result and '&lt;img' in result
    assert '<b>literal</b>' not in result and '&lt;b&gt;literal&lt;/b&gt;' in result
    assert '`$x < y$` and $x < y$' in result
    assert '<https://example.org>' in result
    trusted, _ = responsive_markdown_tables(source, unsafe_allow_html=True)
    assert '<b>literal</b>' in trusted


def test_fenced_examples_and_non_tables_are_unchanged():
    for body in ('No table | here', '```markdown\n| a | b |\n|---|---|\n| c | d |\n```',
                 '~~~\n| a | b |\n|---|---|\n| c | d |\n~~~'):
        assert responsive_markdown_tables(body) == (body, False)


def test_native_cards_preserve_row_column_associations_missing_values_and_all_groups():
    frame = pd.DataFrame({'name': ['<b>unsafe</b>'] + [f'item {i}' for i in range(1, 12)],
                          'score': [None] + [i / 10 for i in range(1, 12)]})
    frame.index.name = 'sample'
    before = frame.copy(deep=True)
    root = ET.fromstring(dataframe_cards(frame))
    cards = root.findall('.//article')
    assert len(cards) == 12
    fields = lambda card: [(e.find('dt').text, e.find('dd').text) for e in card.findall('./dl/div')]
    assert fields(cards[0]) == [('sample', '0'), ('name', '<b>unsafe</b>'), ('score', '—')]
    assert fields(cards[11]) == [('sample', '11'), ('name', 'item 11'), ('score', '1.1')]
    assert not root.findall('.//b')
    assert len(root.findall('details')) == 2
    assert root.find('details').get('open') is not None
    assert root.findall('details')[1].get('open') is None
    assert 'sample' not in dataframe_cards(frame, hide_index=True)
    pd.testing.assert_frame_equal(frame, before)


def test_native_dataframe_in_a_column_retains_values_and_returns_its_native_result():
    at = AppTest.from_string('''
import streamlit as st
from gui import theme_visuals
theme_visuals.install()
left, right = st.columns(2)
result = left.dataframe({'x': [1, 2], 'y': [3, 4]}, hide_index=True)
right.write(type(result).__name__)
''', default_timeout=60).run()
    assert not at.exception
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.to_dict('list') == {'x': [1, 2], 'y': [3, 4]}
    assert any('DeltaGenerator' in e.value for e in at.markdown)
    assert any('ailab-data-card' in e.value for e in at.markdown)

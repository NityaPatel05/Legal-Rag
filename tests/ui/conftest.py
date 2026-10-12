"""UI tests import ui/ modules as top-level (api, render, state), the way
`streamlit run streamlit_app.py` does -- so ui/ goes on sys.path.

AppTest's ButtonGroup (streamlit 1.45) was written for st.feedback, whose
value is a list: for a single-select st.pills / st.segmented_control it
iterates the selected *string* and fails on None. Patched here so the pages
can be driven as users drive them; the app itself is unaffected."""

import sys
from pathlib import Path

import pytest

UI_DIR = Path(__file__).parents[2] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

pytest.importorskip("streamlit")


@pytest.fixture(autouse=True)
def _single_select_button_groups(monkeypatch):
    from streamlit.testing.v1 import element_tree

    def indices(self):
        value = self.value
        values = [] if value is None else value if isinstance(value, list) else [value]
        return [self.options.index(self.format_func(v)) for v in values]

    monkeypatch.setattr(element_tree.ButtonGroup, "indices", property(indices))

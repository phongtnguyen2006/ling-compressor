"""Streamlit smoke checks for the local experimental dashboard."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "semantic_ui" / "app.py"


@pytest.mark.parametrize("arm", ["B0", "B1", "B2"])
def test_dashboard_proposes_sample_with_token_chart(arm):
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception

    app.selectbox[0].set_value("phrase · authored-phrase")
    app.selectbox[1].set_value(arm)
    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert [metric.label for metric in app.metric] == [
        "Original tokens", "Proposed tokens", "Tokens saved", "Estimated input cost saved",
    ]
    assert int(app.metric[2].value) >= 0
    assert app.get("plotly_chart")
    assert any(tab.label.startswith("Selected edits") for tab in app.tabs)


def test_dashboard_explains_missing_vectors():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.selectbox[0].set_value("phrase · authored-phrase")
    app.selectbox[1].set_value("B2")
    app.text_input[0].set_value("/nonexistent/semantic-vectors.txt")
    app.run()
    app.button[0].click().run()

    assert not app.exception
    assert any("local vector" in error.value.lower() for error in app.error)
    assert not app.metric

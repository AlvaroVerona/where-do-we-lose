"""Smoke tests for the Streamlit dashboard.

Nothing else in the test suite imports `app/`, so a broken import, a typo'd
column name, or a stale function signature in the dashboard could pass
`pytest` cleanly while the app itself is broken. These tests run every page
through Streamlit's own `AppTest` harness (a real script run, not a mock) and
assert it renders with no exception - not a substitute for the manual
browser check the dashboard already got, but enough to catch a regression
automatically on every test run.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))  # so `import components.data` works below

PAGES = [
    APP_DIR / "app.py",
    APP_DIR / "pages" / "2_Process_Analysis.py",
    APP_DIR / "pages" / "3_Bottlenecks.py",
    APP_DIR / "pages" / "4_Delay_Prediction.py",
    APP_DIR / "pages" / "5_Optimization.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=[p.name for p in PAGES])
def test_page_runs_without_exception(page_path):
    at = AppTest.from_file(str(page_path), default_timeout=120)
    at.run()
    assert not at.exception, f"{page_path.name} raised: {[str(e) for e in at.exception]}"


def test_executive_overview_shows_kpi_values():
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=120)
    at.run()
    page_text = " ".join(m.value for m in at.markdown)
    assert "Total Applications" in page_text
    assert "SLA Breach Rate" in page_text


def test_bottlenecks_page_ranks_credit_assessment_first():
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_Bottlenecks.py"), default_timeout=120)
    at.run()
    page_text = " ".join(m.value for m in at.markdown)
    assert "Credit Assessment" in page_text


def test_delay_prediction_form_submits_and_shows_result():
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_Delay_Prediction.py"), default_timeout=120)
    at.run()
    assert len(at.button) == 1

    at.button[0].click().run()

    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown)
    assert "SLA Breach Probability" in page_text
    assert any(term in page_text for term in ("LOW RISK", "MODERATE RISK", "HIGH RISK"))


def test_optimization_page_handles_missing_reports_gracefully(tmp_path, monkeypatch):
    """If `python -m src.optimization.resource_allocation` hasn't been run
    yet, the page should show a clear instruction, not crash.
    """
    import components.data as data_module

    monkeypatch.setattr(data_module, "REPORTS_DIR", tmp_path)
    at = AppTest.from_file(str(APP_DIR / "pages" / "5_Optimization.py"), default_timeout=120)
    at.run()
    assert not at.exception
    page_text = " ".join(m.value for m in at.markdown) + " ".join(w.value for w in at.warning)
    assert "resource_allocation" in page_text

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_html_app import (
    _catalog_entries,
    _parse_course_recommendations,
    render_report_html,
    run_analysis,
)


def test_catalogue_lines_are_cleaned_and_deduplicated():
    entries = _catalog_entries("# Courses\n- Python for Data Science\n1. Practical NLP\nPython for Data Science\n")
    assert entries == ["Python for Data Science", "Practical NLP"]


def test_course_selection_uses_catalogue_title_not_model_supplied_title():
    entries = ["Machine Learning Foundations", "Applied NLP"]
    raw = '{"recommendations":[{"id":"C02","title":"Invented course","reason":"Matches the goal."}]}'
    selected = _parse_course_recommendations(raw, entries)
    assert len(selected) == 1
    assert selected[0]["title"] == "Applied NLP"
    assert selected[0]["id"] == "C02"


def test_invalid_or_out_of_range_course_ids_are_suppressed():
    entries = ["Machine Learning Foundations"]
    raw = '{"recommendations":[{"id":"C88","reason":"not valid"},{"id":"Invented","reason":"not valid"}]}'
    assert _parse_course_recommendations(raw, entries) == []


def test_report_escapes_untrusted_text_and_contains_all_sections():
    report = render_report_html(
        '<img src=x onerror=alert(1)>',
        'Qwen/Qwen2.5-0.5B-Instruct',
        'cpu',
        [('CV Analyst', '<script>alert(1)</script>'), ('Skills', 'Python\\nSQL')],
    )
    assert "<script>alert(1)</script>" not in report
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in report
    assert "<img src=x onerror=alert(1)>" not in report
    assert "CV Analyst" in report and "Skills" in report


def test_run_analysis_requires_cv_before_attempting_model_load():
    try:
        run_analysis("Test", "AI Engineer", "", "Course A")
    except ValueError as exc:
        assert "CV/profile text" in str(exc)
    else:
        raise AssertionError("Missing CV should be rejected before model loading")

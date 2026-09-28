from gd_playground import app as streamlit_app
from tests.fake_streamlit import FakeStreamlit


def test_main_renders_default_layout(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert fake_st.page_config["page_title"] == "Gradient Descent Playground"
    assert fake_st.session_state["data"] is not None
    assert len(fake_st.figures) == 3


def test_main_run_button_populates_history(monkeypatch):
    fake_st = FakeStreamlit(pressed={"Run"})
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert fake_st.session_state["history"] is not None
    assert fake_st.session_state["animation_history"] is None


def test_status_lines_agree_with_their_numbers(monkeypatch):
    """«Использует 3 точки», «за 1 итерацию»: the noun follows the number."""
    fake_st = FakeStreamlit(
        pressed={"Run"},
        overrides={"Gradient descent type": "Mini-batch SGD", "Mini-batch size": 3, "Iterations (Run)": 1},
    )
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert "Использует 3 точки для каждого обновления параметров." in fake_st.captions
    assert fake_st.warnings == ["Сходимость не достигнута за 1 итерацию."]

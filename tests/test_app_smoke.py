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

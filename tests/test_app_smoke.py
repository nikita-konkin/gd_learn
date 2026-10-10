from gd_playground import app as streamlit_app
from tests.fake_streamlit import FakeStreamlit


def test_main_renders_default_layout(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert fake_st.page_config["page_title"] == "Тренажёр: градиентный спуск"
    assert fake_st.session_state["data"] is not None
    assert len(fake_st.figures) == 3


def test_main_run_button_populates_history(monkeypatch):
    fake_st = FakeStreamlit(pressed={"Запустить"})
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert fake_st.session_state["history"] is not None
    assert fake_st.session_state["animation_history"] is None


def test_status_lines_agree_with_their_numbers(monkeypatch):
    """«3 случайные точки», «за 1 итерацию»: the noun follows the number."""
    fake_st = FakeStreamlit(
        pressed={"Запустить"},
        overrides={
            "Сколько точек смотреть перед одним шагом": "Mini-batch SGD",
            "Размер пакета": 3,
            "Шагов по кнопке «Запустить»": 1,
        },
    )
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert (
        "Перед каждым шагом усредняется 3 случайные точки." in fake_st.captions
    )
    assert fake_st.warnings == [
        "За 1 итерацию ошибка всё ещё заметно падает — "
        "спуск не дошёл до дна. Дайте больше шагов или увеличьте их размер."
    ]


def test_a_step_that_overshoots_is_not_called_convergence(monkeypatch):
    """Step 1 on the default data: the loss repeats while the weights swing across the minimum."""
    fake_st = FakeStreamlit(pressed={"Запустить"}, overrides={"Размер шага (learning rate)": 1.0})
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert not fake_st.successes
    assert fake_st.warnings[0].startswith("Ошибка не падает: шаг так велик")


def test_a_crawling_descent_is_told_to_take_bigger_steps(monkeypatch):
    fake_st = FakeStreamlit(pressed={"Запустить"}, overrides={"Размер шага (learning rate)": 1e-5})
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app.main()

    assert "шаг так мал" in fake_st.warnings[0]

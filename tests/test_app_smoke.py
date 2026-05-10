from gd_playground import app as streamlit_app


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error

    def __setattr__(self, key, value):
        self[key] = value


class FakeContext:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self.root

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeColumn:
    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self.root

    def __exit__(self, exc_type, exc, tb):
        return False

    def button(self, *args, **kwargs):
        return self.root.button(*args, **kwargs)

    def selectbox(self, *args, **kwargs):
        return self.root.selectbox(*args, **kwargs)

    def slider(self, *args, **kwargs):
        return self.root.slider(*args, **kwargs)


class FakeStreamlit:
    def __init__(self, pressed=None, overrides=None):
        self.session_state = SessionState()
        self.pressed = set(pressed or [])
        self.overrides = overrides or {}
        self.page_config = {}
        self.sidebar = FakeContext(self)
        self.figures = []
        self.rerun_called = False

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        value = self.overrides.get(label, options[index])
        if key is not None:
            self.session_state[key] = value
        return value

    def slider(self, label, min_value, max_value, value=None, step=None, key=None, **kwargs):
        resolved = self.overrides.get(label, value)
        if key is not None:
            self.session_state[key] = resolved
        return resolved

    def number_input(self, label, value=None, **kwargs):
        return self.overrides.get(label, value)

    def button(self, label, **kwargs):
        return label in self.pressed

    def checkbox(self, label, value=False, **kwargs):
        return self.overrides.get(label, value)

    def select_slider(self, label, options, value=None, key=None, **kwargs):
        resolved = self.overrides.get(label, value if value is not None else options[0])
        if key is not None:
            self.session_state[key] = resolved
        return resolved

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [FakeColumn(self) for _ in range(count)]

    def tabs(self, labels):
        return [FakeContext(self) for _ in labels]

    def plotly_chart(self, figure, **kwargs):
        self.figures.append(figure)

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def rerun(self):
        self.rerun_called = True


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

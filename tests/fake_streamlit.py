"""A stand-in for the ``streamlit`` module, for interface smoke tests.

Every playground needs the same set of stubs, so they live in one place:
otherwise a widget added to one interface breaks the other apps' tests.
"""

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

    def __getattr__(self, name):
        # A column can do everything the root object can: button, metric,
        # selectbox and so on. Listing them one by one means an AttributeError
        # every time a widget is added to an interface.
        return getattr(self.root, name)


class FakeStreamlit:
    def __init__(self, pressed=None, overrides=None):
        self.session_state = SessionState()
        self.pressed = set(pressed or [])
        self.overrides = overrides or {}
        self.page_config = {}
        self.sidebar = FakeContext(self)
        self.figures = []
        self.expanders = []
        self.metrics = []
        self.warnings = []
        self.infos = []
        self.errors = []
        self.successes = []
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

    def divider(self, *args, **kwargs):
        return None

    def container(self, *args, **kwargs):
        return FakeContext(self)

    def expander(self, label, expanded=False, **kwargs):
        self.expanders.append(label)
        return FakeContext(self)

    def toggle(self, label, value=False, **kwargs):
        return self.overrides.get(label, value)

    def multiselect(self, label, options, default=None, **kwargs):
        return self.overrides.get(label, list(default) if default is not None else [])

    def text_area(self, label, value=None, key=None, **kwargs):
        return self.overrides.get(label, value if value is not None else "")

    def text_input(self, label, value="", **kwargs):
        return self.overrides.get(label, value)

    def radio(self, label, options, index=0, **kwargs):
        return self.overrides.get(label, options[index])

    def markdown(self, *args, **kwargs):
        return None

    def plotly_chart(self, figure, **kwargs):
        self.figures.append(figure)

    def info(self, body="", *args, **kwargs):
        self.infos.append(str(body))

    def warning(self, body="", *args, **kwargs):
        self.warnings.append(str(body))

    def error(self, body="", *args, **kwargs):
        self.errors.append(str(body))

    def success(self, body="", *args, **kwargs):
        self.successes.append(str(body))

    def write(self, *args, **kwargs):
        return None

    def metric(self, label="", value=None, *args, **kwargs):
        self.metrics.append((str(label), value))
        return None

    def code(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def rerun(self):
        self.rerun_called = True

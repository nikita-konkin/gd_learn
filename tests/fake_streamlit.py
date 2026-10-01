"""A stand-in for the ``streamlit`` module, for interface smoke tests.

Every playground needs the same set of stubs, so they live in one place:
otherwise a widget added to one interface breaks the other apps' tests.

What a test reads afterwards is the *visible* output — ``metrics``, ``captions``,
``infos``, ``warnings`` — because that is what pins a playground's numbers to
the lecture's. A stub that silently swallowed them would make the tests pass
while the page showed anything at all.
"""


class SessionState(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:
            raise AttributeError(item) from error

    def __setattr__(self, key, value):
        self[key] = value


class FakeProxy:
    """Anything that behaves like the root object: a column, a tab, the sidebar.

    Delegates every attribute, so a widget added to an interface does not need a
    matching stub added here. Listing them one by one is what caused an
    ``AttributeError`` every time an interface grew a control.
    """

    def __init__(self, root):
        self.root = root

    def __enter__(self):
        return self.root

    def __exit__(self, exc_type, exc, tb):
        return False

    def __getattr__(self, name):
        return getattr(self.root, name)


class FakeStreamlit:
    def __init__(self, pressed=None, overrides=None):
        self.session_state = SessionState()
        self.pressed = set(pressed or [])
        self.overrides = overrides or {}
        self.page_config = {}
        self.sidebar = FakeProxy(self)
        self.figures = []
        self.expanders = []
        self.metrics = []
        self.warnings = []
        self.infos = []
        self.errors = []
        self.successes = []
        self.captions = []
        self.markdowns = []
        self.sliders = []
        self.dataframes = []
        self.rerun_called = False

    # ---------------------------------------------------------------- layout
    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def title(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [FakeProxy(self) for _ in range(count)]

    def tabs(self, labels):
        return [FakeProxy(self) for _ in labels]

    def container(self, *args, **kwargs):
        return FakeProxy(self)

    def expander(self, label, expanded=False, **kwargs):
        self.expanders.append(label)
        return FakeProxy(self)

    def divider(self, *args, **kwargs):
        return None

    # --------------------------------------------------------------- widgets
    def selectbox(self, label, options, index=0, key=None, **kwargs):
        value = self.overrides.get(label, options[index])
        if key is not None:
            self.session_state[key] = value
        return value

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, key=None, **kwargs):
        # The default is recorded as well as the resolved value: a playground
        # whose sliders do not open on the lecture's settings shows numbers that
        # match nothing, and that is worth a test of its own.
        self.sliders.append((label, value, min_value, max_value))
        resolved = self.overrides.get(label, value)
        if key is not None:
            self.session_state[key] = resolved
        return resolved

    def select_slider(self, label, options, value=None, key=None, **kwargs):
        default = value if value is not None else options[0]
        self.sliders.append((label, default, options[0], options[-1]))
        resolved = self.overrides.get(label, default)
        if key is not None:
            self.session_state[key] = resolved
        return resolved

    def number_input(self, label, value=None, **kwargs):
        return self.overrides.get(label, value)

    def button(self, label, **kwargs):
        return label in self.pressed

    def checkbox(self, label, value=False, **kwargs):
        return self.overrides.get(label, value)

    def toggle(self, label, value=False, **kwargs):
        return self.overrides.get(label, value)

    def radio(self, label, options, index=0, **kwargs):
        return self.overrides.get(label, options[index])

    def multiselect(self, label, options, default=None, **kwargs):
        return self.overrides.get(label, list(default) if default is not None else [])

    def text_input(self, label, value="", **kwargs):
        return self.overrides.get(label, value)

    def text_area(self, label, value=None, key=None, **kwargs):
        return self.overrides.get(label, value if value is not None else "")

    # ---------------------------------------------------------------- output
    def caption(self, body="", *args, **kwargs):
        self.captions.append(str(body))

    def markdown(self, body="", *args, **kwargs):
        self.markdowns.append(str(body))

    def write(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def metric(self, label="", value=None, *args, **kwargs):
        self.metrics.append((str(label), value))
        return None

    def dataframe(self, data=None, *args, **kwargs):
        self.dataframes.append(data)
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

    def rerun(self):
        self.rerun_called = True

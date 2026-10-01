"""Code every playground shares, shipped with each app by ``scripts/build_site.py``.

Each app runs as its own stlite page and gets only the files the site builder
lists for it. Until this package existed, a helper two apps needed was copied
into both, and a test compared the copies — four copies of the pyarrow shim,
three of the number formatting, seven hand-written lists of links to the other
pages. Now the builder adds this package to every app, so there is one copy.

Keep it light: standard library and numpy only, since every app downloads it.
``palette`` holds the ИСТ-51 lecture slides' colours; the other modules serve
every page on the site.
"""

=====================
Installing using uv
=====================

Quick Start
-----------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package and project manager.
It can also download and manage Python itself, so no separate Python or
Anaconda installation is needed.

1. Install uv (see the `uv installation guide
   <https://docs.astral.sh/uv/getting-started/installation/>`_).  For example:

.. code-block:: console

    # Windows (PowerShell)
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

    # macOS / Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

2. Get the TMIP-EMAT source code and set up the environment:

.. code-block:: console

    git clone https://github.com/CTPSSTAFF/tmip-emat.git
    cd tmip-emat
    uv sync
    uv run jupyter lab

``uv sync`` creates a ``.venv`` folder in the project, installs the Python
version named in ``.python-version`` if needed, and installs TMIP-EMAT and its
dependencies exactly as recorded in ``uv.lock``.

Using TMIP-EMAT in Another Project
----------------------------------

To add TMIP-EMAT to an existing uv project, or install it into a
virtual environment with pip:

.. code-block:: console

    uv add "emat @ git+https://github.com/CTPSSTAFF/tmip-emat.git"

    # or, within an activated virtual environment
    uv pip install "emat @ git+https://github.com/CTPSSTAFF/tmip-emat.git"

Running Jupyter
---------------

Use ``uv run`` to run commands inside the project environment without
activating it:

.. code-block:: console

    uv run jupyter lab
    uv run jupyter notebook myfilename.ipynb

To make the environment available as a kernel to an existing Jupyter
installation:

.. code-block:: console

    uv run python -m ipykernel install --user --name EMAT --display-name "EMAT"

Development
-----------

.. code-block:: console

    uv sync --all-extras --dev
    uv run pytest

Troubleshooting
---------------

- The ``pydot`` package renders diagrams using the Graphviz program, which is
  not installed by pip or uv.  Install it separately from
  `graphviz.org <https://graphviz.org/download/>`_ if you need those diagrams.
- Dependencies are pinned to versions compatible with the current code
  (see ``pyproject.toml``); Python 3.10 is required.
- If the interface seems sluggish in the Jupyter notebook, try a different
  browser; Google Chrome has been found to be much more performant for the
  interactive visualizations.

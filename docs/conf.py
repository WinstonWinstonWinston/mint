# -- Path setup --------------------------------------------------------------
import sys, pathlib, importlib.util
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # repo root

# Optional: avoid heavy deps blocking import
autodoc_mock_imports = ["torch", "torch_geometric", "e3nn", "wandb"]

# Safe sanity check (optional)
assert importlib.util.find_spec("mint"), "Sphinx can't import 'mint' from repo root"

# -- Project information -----------------------------------------------------
project = "mint"
author = "Winston"
from mint import __version__ as release  # noqa: E402

# -- Extensions --------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    # "sphinx.ext.intersphinx",  # removed: avoid external links clutter
    "sphinx.ext.linkcode",
    "autoapi.extension",        # AutoAPI
]

# ----------------------- AutoAPI (emit like e3nn: api/mint) -----------------------
import pathlib as _p
autoapi_type = "python"
autoapi_dirs = [str(_p.Path(__file__).resolve().parents[1] / "mint")]
autoapi_root = "api"                 # generates docs under docs/api/*
autoapi_add_toctree_entry = True     # creates api/index and inserts toctree
autoapi_python_class_content = "class"   # only this class' doc (no base __init__ doc)
autoapi_member_order = "bysource"        # stable, source-order listing
autoapi_options = [
    "members",
    "undoc-members",
    # no "inherited-members" (hide torch.nn.Module etc.)
    # no "show-inheritance"  (hide base class boxes)
    # no "special-members", "__init__"
]
autoapi_keep_files = False
autoapi_ignore = ["*/tests/*", "*/_*/**", "*/__main__.py"]

# ----------------------- Typing / docstring rendering -----------------------
autodoc_inherit_docstrings = False        # do not pull parent docstrings
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# -- HTML --------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
myst_update_mathjax = False
html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
default_role = "any"

# -- linkcode (points to YOUR repo) -----------------------------------------
GITHUB_USER = "winstonwinstonwinston"
GITHUB_REPO = "mint"
GITHUB_BRANCH = release if isinstance(release, str) else "main"

def linkcode_resolve(domain, info):
    if domain != "py" or not info.get("module"):
        return None
    try:
        import inspect, os, importlib
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        fn = inspect.getsourcefile(obj)
        if not fn:
            return None
        source, lineno = inspect.getsourcelines(obj)
        rel = os.path.relpath(fn, start=str(_p.Path(__file__).resolve().parents[1]))
        return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{rel}#L{lineno}-L{lineno+len(source)-1}"
    except Exception:
        return None

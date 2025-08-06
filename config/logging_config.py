import logging
import sys

def setup_logging():
    """Configure root logger to write INFO+ to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # avoid adding multiple handlers if called twice
    if not root.handlers:
        root.addHandler(handler)


import logging
import sys


def configure_logging(level=logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return root
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root.addHandler(handler)
    return root


logger = configure_logging()

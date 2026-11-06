"""Route GIST warnings and errors into the portal's step messages.

The portal shows each entry of StepState[i].Messages in results.json on the
step's Results tab (info = blue, warning = yellow, error = red). Anything only
written to stdout/stderr is invisible to normal users.

show_message_to_portal() adds one explicit message. portal_step() wraps a
whole step so that:
  * every UserWarning raised via warnings.warn (e.g. from gistMC) becomes a
    deduplicated warning message,
  * a ValueError raised anywhere in the step becomes an error message, the
    step is marked unsuccessful and results.json is still written,
  * any other exception propagates without writing results.json, so the portal
    marks the run FAILED and sends the admin crash email as before.
"""

import re
import sys
import traceback
import warnings
from contextlib import contextmanager

from TexNetWebToolGPWrappers.Constants import (
    MESSAGE_LEVEL_ERROR,
    MESSAGE_LEVEL_INFO,
    MESSAGE_LEVEL_WARNING,
)

MESSAGE_LEVELS = {
    "info": MESSAGE_LEVEL_INFO,
    "warning": MESSAGE_LEVEL_WARNING,
    "error": MESSAGE_LEVEL_ERROR,
}

# Internal prefixes such as "gistMC.initPP WARNING: " or "gistMC.addWells ERROR: ".
_INTERNAL_PREFIX = re.compile(r"^\s*[\w.]*:?\s*(?:WARNING|ERROR)S?:\s*")


def show_message_to_portal(helper, step_index, message, level="info"):
    """Add a message ("info", "warning" or "error") to the step's portal Messages."""
    helper.addMessageWithStepIndex(step_index, message, MESSAGE_LEVELS[level])


def clean_message(text):
    """Strip internal function prefixes so the message reads well in the UI."""
    cleaned = _INTERNAL_PREFIX.sub("", str(text)).strip()
    return cleaned[:1].upper() + cleaned[1:]


@contextmanager
def portal_step(helper, step_index):
    """Run a portal step, forwarding warnings/ValueErrors as messages and writing results.json."""
    caught = []
    error = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            yield
    except ValueError as e:
        traceback.print_exc()
        error = e
    finally:
        # Recorded warnings are not printed, so re-emit them to keep them in stderr.
        for w in caught:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno)

    if error is not None:
        show_message_to_portal(helper, step_index, clean_message(error), "error")

    seen = set()
    for w in caught:
        if not issubclass(w.category, UserWarning):
            continue
        text = clean_message(w.message)
        if text and text not in seen:
            seen.add(text)
            show_message_to_portal(helper, step_index, text, "warning")

    if error is not None:
        helper.setSuccessForStepIndex(step_index, False)

    helper.writeResultsFile()

    if error is not None:
        sys.exit(1)

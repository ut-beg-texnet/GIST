"""Live progress reporting for GIST step scripts.

The TexNet portal captures step stdout line-by-line and surfaces lines
prefixed with "Info:" as the run's latest status message while the step is
still running. flush=True guarantees the line reaches the portal immediately
(a redirected stdout is block-buffered otherwise).
"""


def report_progress(message: str) -> None:
    print(f"Info: {message}", flush=True)

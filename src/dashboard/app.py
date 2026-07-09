"""
app.py
======

Main entry point for the Enterprise Network Intrusion Intelligence System
Streamlit dashboard.

This module is responsible ONLY for orchestration:
    - Configuring the Streamlit page (title, icon, layout).
    - Initializing application-wide logging.
    - Rendering a sidebar navigation menu.
    - Routing to the appropriate page module (home, prediction,
      analytics, about) based on the user's selection.
    - Providing a top-level exception boundary so that an unhandled
      error in any single page does not crash the entire application.

Run with:
    streamlit run src/dashboard/app.py

Author: Member C
Python: 3.11
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict

import streamlit as st

# --------------------------------------------------------------------------
# Ensure the project root is importable when Streamlit executes this file
# directly (Streamlit runs scripts as __main__, which breaks package-style
# imports unless the project root is explicitly added to sys.path).
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------
# Shared utility imports (owned by other team members / shared module).
# These are wrapped defensively so this file can still be inspected/loaded
# even if src/utils/config.py or src/utils/logger.py have not been
# generated yet. At runtime in the final project, these should resolve
# normally.
# --------------------------------------------------------------------------
try:
    from src.utils.logger import get_logger
except ImportError:  # pragma: no cover - fallback until utils/logger.py exists
    def get_logger(name: str) -> logging.Logger:
        """
        Fallback logger factory used only if src.utils.logger is
        unavailable. Mirrors the expected shared logging interface.

        Args:
            name: Name of the logger, typically __name__ of the caller.

        Returns:
            A configured logging.Logger instance.
        """
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

try:
    from src.utils.config import APP_TITLE, APP_ICON, LAYOUT
except ImportError:  # pragma: no cover - fallback until utils/config.py exists
    APP_TITLE = "Enterprise Network Intrusion Intelligence System"
    APP_ICON = "🛡️"
    LAYOUT = "wide"


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Page module imports (Member C's own modules).
# Wrapped in a try/except so app.py remains runnable in isolation for
# review purposes, even before sibling files are generated. Each import
# failure is logged clearly to aid debugging once the missing file is
# created.
# --------------------------------------------------------------------------
def _safe_import_page(module_name: str, attr_name: str) -> Callable[[], None]:
    """
    Safely import a page-rendering function from a dashboard submodule.

    If the target module or attribute does not yet exist (e.g. it has
    not been generated yet by the developer), a placeholder render
    function is returned instead of raising an ImportError. This keeps
    app.py independently runnable and provides a clear on-screen message
    rather than crashing the whole application.

    Args:
        module_name: Dotted path to the submodule, e.g. "src.dashboard.home".
        attr_name: Name of the callable to import from that module.

    Returns:
        A zero-argument callable that renders the requested page, or a
        placeholder callable if the import failed.
    """
    try:
        module = __import__(module_name, fromlist=[attr_name])
        page_function = getattr(module, attr_name)
        logger.debug("Successfully imported %s from %s", attr_name, module_name)
        return page_function
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Could not import '%s' from '%s': %s. A placeholder page will "
            "be shown instead.",
            attr_name,
            module_name,
            exc,
        )

        def _placeholder() -> None:
            st.warning(
                f"The page module `{module_name}` is not available yet. "
                f"Expected a callable named `{attr_name}`."
            )

        return _placeholder


render_home = _safe_import_page("src.dashboard.home", "render")
render_prediction = _safe_import_page("src.dashboard.prediction", "render")
render_analytics = _safe_import_page("src.dashboard.analytics", "render")
render_about = _safe_import_page("src.dashboard.about", "render")


# --------------------------------------------------------------------------
# Navigation configuration
# --------------------------------------------------------------------------
PAGES: Dict[str, Callable[[], None]] = {
    "🏠 Home": render_home,
    "🔍 Prediction": render_prediction,
    "📊 Analytics": render_analytics,
    "ℹ️ About": render_about,
}


def configure_page() -> None:
    """
    Configure global Streamlit page settings.

    This must be the first Streamlit command executed in the script,
    per Streamlit's API requirements. Wrapped in a try/except because
    `st.set_page_config` raises a StreamlitAPIException if called more
    than once within a single script run (e.g. during hot-reload).
    """
    try:
        st.set_page_config(
            page_title=APP_TITLE,
            page_icon=APP_ICON,
            layout=LAYOUT,
            initial_sidebar_state="expanded",
        )
        logger.info("Streamlit page configuration applied successfully.")
    except st.errors.StreamlitAPIException as exc:
        logger.debug("Page config already set for this session: %s", exc)


def render_sidebar() -> str:
    """
    Render the sidebar navigation menu and system branding.

    Returns:
        The label of the page selected by the user.
    """
    with st.sidebar:
        st.title(f"{APP_ICON} {APP_TITLE}")
        st.markdown("---")
        selection = st.radio(
            label="Navigate",
            options=list(PAGES.keys()),
            key="nav_selection",
        )
        st.markdown("---")
        st.caption("Academic Project · CICIDS2017 · Network Intrusion Detection")
    return selection


def render_footer() -> None:
    """Render a persistent footer with basic system status information."""
    st.markdown("---")
    st.caption(
        "Enterprise Network Intrusion Intelligence System | "
        "Powered by Streamlit, scikit-learn & CICIDS2017"
    )


def main() -> None:
    """
    Application entry point.

    Configures the page, renders the sidebar, routes to the selected
    page module, and provides a top-level exception boundary so that
    unexpected errors are logged and surfaced to the user gracefully
    instead of producing an unhandled stack trace in the UI.
    """
    configure_page()
    logger.info("Dashboard application started.")

    try:
        selection = render_sidebar()
        logger.info("User navigated to page: %s", selection)

        page_function = PAGES.get(selection)
        if page_function is None:
            logger.error("No page function registered for selection: %s", selection)
            st.error("Requested page could not be found.")
        else:
            page_function()

        render_footer()

    except Exception as exc:  # noqa: BLE001 - top-level UI safety net
        logger.exception("Unhandled exception while rendering the dashboard.")
        st.error(
            "An unexpected error occurred while loading this page. "
            "Please check the application logs for details."
        )
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(exc)))


if __name__ == "__main__":
    main()
"""
The session state an AppTest run holds, across Streamlit versions.

Streamlit 1.67 wraps AppTest's session state in a proxy that exposes
to_dict(); earlier versions returned the state object itself, which
exposes filtered_state. Both give user state plus keyed widget values.
"""

from __future__ import annotations

from typing import Any


def held_state(app) -> dict[str, Any]:
    state = app.session_state
    if hasattr(type(state), "to_dict"):
        return state.to_dict()
    return state.filtered_state

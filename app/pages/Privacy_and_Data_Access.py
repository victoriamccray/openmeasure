"""
Privacy & Data Access - the full disclosure, in one place.

Not a workflow: no catalog entry, no module_key, no lifecycle stage -
matches pages/Explore_Real_Data.py and pages/Method_Selection.py's
precedent for reference pages declared directly in Home.py rather than
through shared/catalog.py.

Every page's short "Data handling on this page" expander (rendered via
shared.data_handling.render_data_handling_summary) links back here. This
page adds the full picture: what the five categories mean, every page's
disclosure side by side from the same shared.data_handling.DISCLOSURES
registry those expanders draw from, and the known limitations stated
plainly rather than smoothed over.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from shared.data_handling import DISCLOSURES

st.set_page_config(
    page_title="OpenMeasure · Privacy & Data Access",
    page_icon=":material/privacy_tip:",
    layout="centered",
)

st.title("Privacy & Data Access")
st.caption(
    "OpenMeasure does not currently persist uploaded research datasets "
    "to a database. Most workflows process data in memory; specific "
    "exceptions and session-scoped handling are disclosed below."
)

st.divider()

st.subheader("What the Categories Mean")
st.markdown(
    """
- **Data access** - how data gets into the page: a **user upload**, a **local** path read directly on the machine running the app, a **public remote** source fetched live at runtime, or a **bundled example** shipped with OpenMeasure itself.
- **Processing** - whether data is handled **in memory** only, or written to a **temporary file** at any point.
- **Persistent storage** - whether anything is written to a database or other storage that outlives the running process. Every page today reports **None**.
- **Data accessed** - the specific fields or files a page actually reads, stated plainly rather than left to be inferred.
- **Redistribution** - whether OpenMeasure bundles a copy of a third-party dataset (**Bundled**), deliberately does not (**Not bundled**), or there is no third-party dataset involved at all (**Not applicable** - true wherever only data the user brings is processed).
"""
)

st.divider()

st.subheader("Every Page, Side By Side")

table_rows = [
    {
        "Page": item.page,
        "Data access": ", ".join(item.data_access),
        "Processing": ", ".join(item.processing),
        "Persistent storage": item.persistent_storage,
        "Data accessed": item.data_accessed,
        "Redistribution": item.redistribution,
    }
    for item in DISCLOSURES
]
st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

st.caption("Notes for each page, in full:")
for item in DISCLOSURES:
    if item.notes:
        st.write(f"**{item.page}**")
        st.caption(item.notes)

st.divider()

st.subheader("OpenMeasure's Code vs. Its Hosting Platform")
st.write(
    "Everything above describes what OpenMeasure's own code does. It is "
    "not a claim about the platform that runs the app underneath it."
)
st.markdown(
    """
- **OpenMeasure application telemetry** - none. No page in this app sends usage events, click tracking, or any other application-level analytics anywhere.
- **Hosting/platform processing** - this app runs on Streamlit Community Cloud. An uploaded file is copied from your browser to that platform's backend and held in an in-memory buffer there (RAM, not disk) - server-side processing, not processing that happens solely on your own device. Per Streamlit's own documentation, an uploaded file is removed from that memory when it is replaced, the uploader is cleared, or the browser tab is closed. Streamlit Community Cloud also runs its own separate viewer analytics for a hosted app (a total and a recent-unique viewer count) independent of anything OpenMeasure's own code does; for a public app, Streamlit's documentation states that viewers outside the app owner's workspace are shown to the owner as anonymous pseudonyms.
- **Research-data persistence by OpenMeasure** - none, beyond the exceptions already disclosed below under Known Limitations.
"""
)
st.caption(
    "The file-uploader and app-analytics behavior described above is "
    "Streamlit's, not OpenMeasure's, and is summarized from Streamlit's "
    "own current documentation (docs.streamlit.io) rather than "
    "guaranteed to stay accurate as that platform changes; consult it "
    "directly for what applies today."
)

st.divider()

st.subheader("Known Limitations")
st.write(
    "Stated as plainly as the rest of the app states its statistical "
    "caveats, not smoothed over:"
)
st.warning(
    "HealthRing writes an uploaded archive to a temporary file in the "
    "app environment, which is not actively deleted by OpenMeasure in "
    "the current implementation - it can outlive the browser session "
    "until the host or container reclaims it."
)
st.warning(
    "shared/handoff.py, used by the Cross-Analysis Implications page, "
    "records the uploaded filename verbatim in session state alongside "
    "a hash of the data itself. A filename that happened to include "
    "identifying text would be retained as-is for the session."
)
st.caption(
    "This page describes current implementation behavior. It is not a "
    "legal or compliance certification."
)

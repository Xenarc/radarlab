"""Streamlit front-end: edit RadarSystem / Scenario as JSON, see matplotlib output live.

Run with:  streamlit run radar_app.py
"""

import contextlib
import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must be set before pyplot is imported

from main import radar_diagnostics
from radar import RadarSystem
from scenario import Scenario
from scenario_1 import build_radar_1, build_radar_2, build_scenario_1, build_scenario_2, build_scenario_3
from visualise import visualise

import matplotlib.pyplot as plt
import streamlit as st
from pydantic import ValidationError

RADAR_PRESETS = {"Radar 1": build_radar_1, "Radar 2 (APG-79 class)": build_radar_2}
SCENARIO_PRESETS = {"Scenario 1": build_scenario_1,"Scenario 2": build_scenario_2, "Scenario 3": build_scenario_3}

st.set_page_config(page_title="Radar sandbox", layout="wide")


def to_json(model) -> str:
    return model.model_dump_json(indent=2)


def load_radar_preset() -> None:
    st.session_state["radar_json"] = to_json(RADAR_PRESETS[st.session_state["radar_preset"]]())


def load_scenario_preset() -> None:
    st.session_state["scenario_json"] = to_json(
        SCENARIO_PRESETS[st.session_state["scenario_preset"]]()
    )


# Initialise editors on first run
if "radar_json" not in st.session_state:
    st.session_state["radar_json"] = to_json(build_radar_1())
if "scenario_json" not in st.session_state:
    st.session_state["scenario_json"] = to_json(build_scenario_3())


st.title("Radar sandbox")
st.caption("Edit the JSON, then press Ctrl+Enter (or click outside the box) to re-run.")

left, right = st.columns([1, 1.3], gap="large")

# ------------------------------ Editors -----------------------------------
with left:
    tab_radar, tab_scenario = st.tabs(["Radar", "Scenario"])

    with tab_radar:
        st.selectbox(
            "Load preset",
            list(RADAR_PRESETS),
            key="radar_preset",
            on_change=load_radar_preset,
        )
        st.text_area("RadarSystem JSON", key="radar_json", height=600, label_visibility="collapsed")

    with tab_scenario:
        st.selectbox(
            "Load preset",
            list(SCENARIO_PRESETS),
            key="scenario_preset",
            on_change=load_scenario_preset,
        )
        st.text_area(
            "Scenario JSON", key="scenario_json", height=600, label_visibility="collapsed"
        )

# ------------------------------ Validate ----------------------------------
radar = scenario = None
errors = []

try:
    radar = RadarSystem.model_validate_json(st.session_state["radar_json"])
except ValidationError as e:
    errors.append(("Radar", e))

try:
    scenario = Scenario.model_validate_json(st.session_state["scenario_json"])
except ValidationError as e:
    errors.append(("Scenario", e))

# ------------------------------ Results -----------------------------------
with right:
    if errors:
        for name, err in errors:
            st.error(f"{name} JSON is invalid")
            for issue in err.errors():
                loc = ".".join(str(p) for p in issue["loc"]) or "(root)"
                st.code(f"{loc}: {issue['msg']}", language=None)
    else:
        plt.close("all")
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "radar_run.png"
            try:
                with contextlib.redirect_stdout(stdout):
                    if radar is not None and scenario is not None:
                        radar_diagnostics(radar, scenario)
                        visualise(scenario, radar, save_path=str(save_path))
            except Exception as exc:  # surface runtime errors in the UI
                st.exception(exc)
            else:
                if stdout.getvalue():
                    with st.expander("Diagnostics", expanded=True):
                        st.code(stdout.getvalue(), language=None)

                for num in plt.get_fignums():
                    st.pyplot(plt.figure(num), clear_figure=False)

                if save_path.exists():
                    st.download_button(
                        "Download PNG",
                        data=save_path.read_bytes(),
                        file_name="radar_run.png",
                        mime="image/png",
                    )
        plt.close("all")
if __name__ == "__main__":
    import sys

    from streamlit.runtime import exists
    from streamlit.web import cli

    # Streamlit also executes the script with __name__ == "__main__",
    # so only relaunch when we're running as plain `python lab.py`.
    if not exists():
        sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
        sys.exit(cli.main())
import math

import numpy as np

from radarlab.clutter import SeaClutter, SeaState
from radarlab.radar import RadarSystem
from radarlab.scenario import optimise_for_max_detection_range
from radarlab.kinematics import GeodeticPosition
from radarlab.kinematics import line_of_sight_in_body_frame
from radarlab.visualise import plot_sweep
from radarlab.example_scenarios import build_radar_2, build_scenario_3


def run(radar: RadarSystem=build_radar_2()) -> None:
    sea_states: list[SeaState] = [1, 2, 3, 4, 5, 6]
    ranges_m = []

    for sea_state in sea_states:
        scenario = build_scenario_3(radar_altitude=200, target_altitude=20)
        assert isinstance(scenario.clutter, SeaClutter)
        scenario.clutter.sea_state = sea_state

        try:
            max_scenario = optimise_for_max_detection_range(scenario, radar)
        except ValueError:
            ranges_m.append(math.inf)
            continue

        los = line_of_sight_in_body_frame(max_scenario.detection_geometry.radar_pose, max_scenario.detection_geometry.target_position)
        max_range = los.range_m
        ranges_m.append(max_range)

    plot_sweep(
        sea_states,
        ranges_m,
        title="Maximum detection range vs sea state",
        x_label="Sea state",
        y_label="Maximum detection range (m)",
    )


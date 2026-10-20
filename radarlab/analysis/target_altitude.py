import math

import numpy as np

from radarlab.kinematics import target_from_ned
from radarlab.visualise import plot_sweep, visualise
from radarlab.radar import RadarSystem
from radarlab.scenario import optimise_for_max_detection_range
from radarlab.kinematics import line_of_sight_in_body_frame
from radarlab.example_scenarios import build_radar_1, build_scenario_3


def run(radar: RadarSystem=build_radar_1()) -> None:
    radar_alt = 3000
    altitudes = np.linspace(0.0, radar_alt, 100)
    ranges_m = []

    for altitude in altitudes:
        scenario = build_scenario_3(radar_alt, altitude)
        try:
            max_scenario = optimise_for_max_detection_range(scenario, radar)
        except ValueError:
            ranges_m.append(math.inf)
            continue
        los = line_of_sight_in_body_frame(max_scenario.detection_geometry.radar_pose, max_scenario.detection_geometry.target_position)
        max_range = los.range_m
        ranges_m.append(max_range)
        scenario.detection_geometry.target_position = target_from_ned(
            scenario.detection_geometry.radar_pose.position,
            max_range,
            0,
            radar_alt - altitude,
        )

    plot_sweep(
        altitudes,
        ranges_m,
        title="Maximum detection range vs target altitude",
        x_label="Target Altitide (m)",
        y_label="Maximum detection range (m)",
    )


import numpy as np

from radarlab.radar import RadarSystem
from radarlab.scenario import optimise_for_max_detection_range
from radarlab.kinematics import line_of_sight_in_body_frame
from radarlab.visualise import plot_sweep
from radarlab.scenario import Scenario
from radarlab.example_scenarios import build_radar_1, build_scenario_2
import radarlab.visualise as visualise


def run(radar: RadarSystem=build_radar_1()) -> None:
    easts = np.linspace(0.0, 90_000.0, 10)
    ranges_m = []

    scenario: Scenario
    for east_m in easts:
        scenario = build_scenario_2(east=east_m)
        max_scenario = optimise_for_max_detection_range(scenario, radar)
        los = line_of_sight_in_body_frame(max_scenario.detection_geometry.radar_pose, max_scenario.detection_geometry.target_position)
        max_range = los.range_m
        ranges_m.append(max_range)

    plot_sweep(
        easts,
        ranges_m,
        title="Maximum detection range vs East displacement",
        x_label="East Displacement (m)",
        y_label="Maximum detection range (m)",
    )

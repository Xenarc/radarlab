import numpy as np

from radarlab.radar import RadarSystem
from radarlab.scenario import optimise_for_max_detection_range
from radarlab.kinematics import line_of_sight_in_body_frame
from radarlab.visualise import plot_sweep
from radarlab.example_scenarios import build_radar_1, build_scenario_2


def run(radar: RadarSystem=build_radar_1()) -> None:
    frequencies_hz = np.linspace(1e9, 12e9, 25)
    ranges_m = []

    for frequency_hz in frequencies_hz:
        radar_with_freq = radar.model_copy(
            update={
                "mode": radar.mode.model_copy(
                    update={
                        "rf": radar.mode.rf.model_copy(
                            update={"carrier_frequency_hz": float(frequency_hz)}
                        )
                    }
                )
            }
        )
        scenario = build_scenario_2()

        max_scenario = optimise_for_max_detection_range(scenario, radar_with_freq)
        los = line_of_sight_in_body_frame(max_scenario.detection_geometry.radar_pose, max_scenario.detection_geometry.target_position)
        max_range = los.range_m
        ranges_m.append(max_range)

    plot_sweep(
        frequencies_hz / 1e9,
        ranges_m,
        title="Maximum detection range vs carrier frequency",
        x_label="Carrier frequency (GHz)",
        y_label="Maximum detection range (m)",
    )

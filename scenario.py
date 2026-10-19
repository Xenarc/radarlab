import math

import numpy as np
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation
from clutter import (
    GroundClutter,
    SeaClutter,
    clutter_patch_area_m2,
    clutter_sigma0,
    grazing_angle_rad,
)
from kinematics import (
    DetectionGeometry,
    LineOfSight,
    line_of_sight_in_body_frame,
    target_from_ned,
)
from radar import (
    RadarSystem,
    echo_power_w,
    max_detection_range,
    noise_power,
    pulse_compression_gain,
    required_single_pulse_snr_db,
    snr_on_target,
)

class Scenario(BaseModel):
    """Static scenario: one radar platform and one target at a fixed position."""

    detection_geometry: DetectionGeometry
    clutter: GroundClutter | SeaClutter = Field(default_factory=GroundClutter)


def target_in_scan_volume(radar: RadarSystem, line_of_sight: LineOfSight) -> bool:
    """True if the target lies inside the scan extent, centred on the radar boresight."""
    scan = radar.mode.scan
    return (
        abs(line_of_sight.azimuth_deg) <= scan.scan_extent_az_deg / 2.0
        and abs(line_of_sight.elevation_deg) <= scan.scan_extent_el_deg / 2.0
    )


def scenario_snr_db(scenario: Scenario, radar: RadarSystem) -> float | None:
    """Single-pulse SNR (dB) at the target, or None if the target is outside the scan volume."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    if not target_in_scan_volume(radar, line_of_sight):
        print(f"target out of scan volume! line_of_sight:\n{line_of_sight}")
        return None
    snr_linear = snr_on_target(radar, detection_geometry.target, line_of_sight.range_m)
    return 10.0 * math.log10(snr_linear)


def optimise_for_max_detection_range(
    scenario: Scenario,
    radar: RadarSystem,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float = 1e-6,
) -> Scenario:
    """Return a scenario whose target sits at the maximum detectability range along the current boresight."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    if line_of_sight.range_m <= 0.0:
        raise ValueError("Target range must be positive.")

    max_range_m = max_detection_range(
        radar,
        detection_geometry.target,
        detection_probability_threshold,
        false_alarm_probability_threshold,
    )

    azimuth_rad = math.radians(line_of_sight.azimuth_deg)
    elevation_rad = math.radians(line_of_sight.elevation_deg)
    body_direction = np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            -math.sin(elevation_rad),
        ],
        dtype=float,
    )

    attitude = detection_geometry.radar_pose.attitude
    body_to_local = Rotation.from_euler(
        "ZYX",
        [attitude.yaw_deg, attitude.pitch_deg, attitude.roll_deg],
        degrees=True,
    )
    local_direction = body_to_local.apply(body_direction)
    north_m, east_m, down_m = max_range_m * local_direction

    new_target_position = target_from_ned(
        detection_geometry.radar_pose.position,
        float(north_m),
        float(east_m),
        float(down_m),
    )

    return scenario.model_copy(
        update={
            "detection_geometry": detection_geometry.model_copy(
                update={"target_position": new_target_position}
            )
        }
    )


def scenario_sinr_db(scenario: Scenario, radar: RadarSystem) -> float | None:
    """Single-pulse SINR (dB) at the target including clutter, or None if out of scan volume."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    if not target_in_scan_volume(radar, line_of_sight):
        print(f"target out of scan volume! line_of_sight:\n{line_of_sight}")
        return None
    sinr_linear = sinr_on_target(radar, scenario)
    return 10.0 * math.log10(sinr_linear)


def scenario_detection_margin_db(
    scenario: Scenario,
    radar: RadarSystem,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float = 1e-6,
) -> float | None:
    """Available SNR minus required SNR (dB). Positive means detectable. None if out of scan volume."""
    available_snr_db = scenario_snr_db(scenario, radar)
    if available_snr_db is None:
        return None
    required_db = required_single_pulse_snr_db(
        radar.receiver,
        scenario.detection_geometry.target,
        detection_probability_threshold,
        false_alarm_probability_threshold,
    )
    return available_snr_db - required_db


def scenario_sinr_margin_db(
    scenario: Scenario,
    radar: RadarSystem,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float = 1e-6,
) -> float | None:
    """Available SINR minus required SNR (dB), including clutter. Positive means detectable."""
    available_sinr_db = scenario_sinr_db(scenario, radar)
    if available_sinr_db is None:
        return None
    required_db = required_single_pulse_snr_db(
        radar.receiver,
        scenario.detection_geometry.target,
        detection_probability_threshold,
        false_alarm_probability_threshold,
    )
    return available_sinr_db - required_db


def clutter_rcs_m2(scenario: Scenario, radar: RadarSystem) -> float:
    """Clutter radar cross section (m^2) = sigma0 * illuminated area."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )

    psi = grazing_angle_rad(
        detection_geometry.radar_pose.position.altitude_m,
        line_of_sight.range_m,
        detection_geometry.target_position.altitude_m,
    )
    if psi is None:
        return 0
    return clutter_sigma0(scenario.clutter, psi, radar.mode) * clutter_patch_area_m2(
        radar, scenario.detection_geometry
    )


def clutter_power_w(
    radar: RadarSystem,
    scenario: Scenario,
) -> float:
    """Clutter power (W) at the receiver input from one clutter source (pre-compression)."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    return echo_power_w(radar, clutter_rcs_m2(scenario, radar), line_of_sight.range_m)


def sinr_on_target(radar: RadarSystem, scenario: Scenario) -> float:
    """Signal-to-(noise + clutter) ratio (linear) after pulse compression, before integration."""

    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    pcg = pulse_compression_gain(radar.mode)
    signal = (
        echo_power_w(
            radar,
            detection_geometry.target.radar_cross_section_m2,
            line_of_sight.range_m,
        )
        * pcg
    )
    clutter = clutter_power_w(radar, scenario) * pcg
    return signal / (noise_power(radar) + clutter)

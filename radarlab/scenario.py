import math

from matplotlib import pyplot as plt
import numpy as np
import pymap3d
from pydantic import BaseModel, Field
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from radarlab.kinematics import (
    Attitude,
    DetectionGeometry,
    GeodeticPosition,
    Pose,
    target_from_ned,
    boresight_attitude,
)
from radarlab.clutter import (
    GroundClutter,
    SeaClutter,
    clutter_patch_area_m2,
    clutter_sigma0,
    grazing_angle_rad,
)
from radarlab.kinematics import (
    DetectionGeometry,
    LineOfSight,
    line_of_sight_in_body_frame,
    target_from_ned,
)
from radarlab.radar import (
    RadarSystem,
    echo_power_w,
    noise_power,
    pulse_compression_gain,
    required_single_pulse_snr_db,
    snr_on_target,
    to_db,
)
from radarlab.atmospheric_attenuation import atmospheric_loss

DETECTION_PROBABILITY = 0.9
FALSE_ALARM_PROBABILITY = 1e-6

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


def scenario_with_max_range(scenario: Scenario, max_range_m: float) -> Scenario:
    """Move the target to the specified maximum range while keeping its altitude fixed."""
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    if line_of_sight.range_m <= 0.0:
        raise ValueError("Target range must be positive.")

    current_north_m, current_east_m, current_down_m = pymap3d.geodetic2ned(
        detection_geometry.target_position.latitude_deg,
        detection_geometry.target_position.longitude_deg,
        detection_geometry.target_position.altitude_m,
        detection_geometry.radar_pose.position.latitude_deg,
        detection_geometry.radar_pose.position.longitude_deg,
        detection_geometry.radar_pose.position.altitude_m,
    )
    horizontal_range_m = math.hypot(current_north_m, current_east_m)
    if horizontal_range_m <= 0.0:
        raise ValueError(
            "Target must not be directly above the radar to place it at max range."
        )
    if max_range_m < abs(current_down_m):
        raise ValueError(
            "Requested range is shorter than the target's altitude offset; keeping altitude would be impossible."
        )

    # Keep the target at the same altitude as the current geometry while moving it out
    # along the existing bearing in the horizontal plane.
    desired_horizontal_range_m = math.sqrt(max_range_m**2 - current_down_m**2)
    north_m = desired_horizontal_range_m * (current_north_m / horizontal_range_m)
    east_m = desired_horizontal_range_m * (current_east_m / horizontal_range_m)
    down_m = current_down_m

    new_target_position = target_from_ned(
        detection_geometry.radar_pose.position,
        float(north_m),
        float(east_m),
        float(down_m),
    )
    return Scenario(
        detection_geometry=DetectionGeometry(
            radar_pose=Pose(
                position=scenario.detection_geometry.radar_pose.position,
                attitude=boresight_attitude(north_m, east_m, down_m),
            ),
            target_position=new_target_position,
            target=scenario.detection_geometry.target,
        ),
        clutter=scenario.clutter,
    )


def optimise_for_max_detection_range(
    scenario: Scenario,
    radar: RadarSystem,
    detection_probability_threshold: float = DETECTION_PROBABILITY,
    false_alarm_probability_threshold: float = FALSE_ALARM_PROBABILITY,
    min_range_m: float = 1.0,
    max_range_m: float = 250_000.0,
    n_grid: int = 200,
) -> Scenario:
    """Return a scenario with the target at the farthest range where SINR margin >= 0."""

    def margin_db(range_m: float) -> float:
        try:
            candidate = scenario_with_max_range(scenario, float(range_m))
        except ValueError:
            return float("nan")
        margin = scenario_sinr_margin_db(
            candidate,
            radar,
            detection_probability_threshold,
            false_alarm_probability_threshold,
        )
        return float("nan") if margin is None else margin

    ranges = np.linspace(min_range_m, max_range_m, n_grid)
    margins = np.array([margin_db(r) for r in ranges])

    valid = np.isfinite(margins)
    if not valid.any():
        raise ValueError("No valid SINR margin found in the search range.")

    detectable = valid & (margins >= 0.0)
    if not detectable.any():
        raise ValueError("Target is undetectable at every range in the search range.")

    last = int(np.flatnonzero(detectable)[-1])

    # Still detectable at the far limit: nothing to refine.
    if last == n_grid - 1:
        return scenario_with_max_range(scenario, float(ranges[last]))

    # Bracket the farthest crossing: [last (>= 0), next valid point (< 0)].
    nxt = last + 1
    while nxt < n_grid and not valid[nxt]:
        nxt += 1
    if nxt == n_grid:
        return scenario_with_max_range(scenario, float(ranges[last]))

    root = brentq(
        margin_db, ranges[last], ranges[nxt], xtol=1e-3, rtol=1e-6, maxiter=100
    )
    return scenario_with_max_range(scenario, float(root))


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
    return to_db(sinr_linear)


def scenario_snr_margin_db(
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
    atmos = atmospheric_loss(
        radar,
        line_of_sight.range_m,
        detection_geometry.radar_pose.position.altitude_m,
        detection_geometry.target_position.altitude_m,
    )
    signal = (
        echo_power_w(
            radar,
            detection_geometry.target.radar_cross_section_m2,
            line_of_sight.range_m,
        )
        * pcg
        * atmos
    )
    return signal / (noise_power(radar) + (clutter_power_w(radar, scenario)))

import math

from pydantic import BaseModel, Field
import numpy as np
import pymap3d
from scipy.spatial.transform import Rotation
from radar import RadarSystem, Receiver, Target, Transmitter, required_single_pulse_snr_db, snr_on_target


class GeodeticPosition(BaseModel):
    """A point on Earth in WGS84 coordinates."""

    latitude_deg: float = Field(ge=-90.0, le=90.0)
    longitude_deg: float = Field(ge=-180.0, le=180.0)
    altitude_m: float = Field(default=0.0, description="Height above the WGS84 ellipsoid in metres")


class Attitude(BaseModel):
    """Aerospace yaw-pitch-roll of a platform relative to local North-East-Down."""

    yaw_deg: float = Field(default=0.0, description="Heading, clockwise from north")
    pitch_deg: float = Field(default=0.0, description="Nose up is positive")
    roll_deg: float = Field(default=0.0, description="Right wing down is positive")


class Pose(BaseModel):
    """Where a platform is and which way it points."""

    position: GeodeticPosition
    attitude: Attitude = Field(default_factory=Attitude)


class Scenario(BaseModel):
    """Static scenario: one radar platform and one target at a fixed position."""

    radar_pose: Pose
    target_position: GeodeticPosition
    target: Target = Field(default_factory=Target)


class LineOfSight(BaseModel):
    """Target direction as seen from the radar, in the radar's body frame."""

    range_m: float
    azimuth_deg: float = Field(description="Positive to the right of the nose")
    elevation_deg: float = Field(description="Positive above the nose")


def line_of_sight_in_body_frame(radar_pose: Pose, target_position: GeodeticPosition) -> LineOfSight:
    """Range, azimuth and elevation of the target in the radar platform's body frame.

    Body frame: x forward, y right, z down. The radar boresight is the body x-axis.
    """
    radar = radar_pose.position
    north_m, east_m, down_m = pymap3d.geodetic2ned(
        target_position.latitude_deg, target_position.longitude_deg, target_position.altitude_m,
        radar.latitude_deg, radar.longitude_deg, radar.altitude_m,
    )
    vector_in_local_frame = np.array([north_m, east_m, down_m])

    attitude = radar_pose.attitude
    body_to_local = Rotation.from_euler(
        "ZYX", [attitude.yaw_deg, attitude.pitch_deg, attitude.roll_deg], degrees=True
    )
    forward_m, right_m, down_in_body_m = body_to_local.inv().apply(vector_in_local_frame)

    return LineOfSight(
        range_m=float(np.linalg.norm(vector_in_local_frame)),
        azimuth_deg=float(np.degrees(np.arctan2(right_m, forward_m))),
        elevation_deg=float(np.degrees(np.arctan2(-down_in_body_m, np.hypot(forward_m, right_m)))),
    )


def target_in_scan_volume(radar: RadarSystem, line_of_sight: LineOfSight) -> bool:
    """True if the target lies inside the scan extent, centred on the radar boresight."""
    scan = radar.mode.scan
    return (
        abs(line_of_sight.azimuth_deg) <= scan.scan_extent_az_deg / 2.0
        and abs(line_of_sight.elevation_deg) <= scan.scan_extent_el_deg / 2.0
    )


def scenario_snr_db(scenario: Scenario, radar: RadarSystem) -> float | None:
    """Single-pulse SNR (dB) at the target, or None if the target is outside the scan volume."""
    line_of_sight = line_of_sight_in_body_frame(scenario.radar_pose, scenario.target_position)
    if not target_in_scan_volume(radar, line_of_sight):
        print(f"target out of scan volume! line_of_sight:\n{line_of_sight}")  
        return None
    snr_linear = snr_on_target(radar, scenario.target, line_of_sight.range_m)
    return 10.0 * math.log10(snr_linear)


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
        radar.receiver, scenario.target, detection_probability_threshold, false_alarm_probability_threshold
    )
    return available_snr_db - required_db

import math

from pydantic import BaseModel, Field
import numpy as np
import pymap3d
from scipy.spatial.transform import Rotation
from radarlab.radar import (
    Target,
)

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


class DetectionGeometry(BaseModel):
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


def target_from_ned(
    radar_position: GeodeticPosition,
    north_m: float,
    east_m: float,
    down_m: float,
) -> GeodeticPosition:
    """Target position given a NED offset from the radar."""
    lat, lon, alt = pymap3d.ned2geodetic(
        north_m,
        east_m,
        down_m,
        radar_position.latitude_deg,
        radar_position.longitude_deg,
        radar_position.altitude_m,
    )
    return GeodeticPosition(latitude_deg=lat, longitude_deg=lon, altitude_m=alt)


def boresight_attitude(north_m: float, east_m: float, down_m: float) -> Attitude:
    """Attitude that points the radar boresight at a NED offset (zero roll)."""
    yaw_deg = math.degrees(math.atan2(east_m, north_m))
    ground_range_m = math.hypot(north_m, east_m)
    pitch_deg = math.degrees(math.atan2(-down_m, ground_range_m))  # negative = look down
    return Attitude(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=0.0)

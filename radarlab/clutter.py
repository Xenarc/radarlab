import math
from typing import Annotated, Final, Literal, Protocol, TypeAlias, Union

import numpy as np
from pydantic import BaseModel, Field

from radarlab.kinematics import line_of_sight_in_body_frame
from radarlab.radar import (
    Mode,
    RadarSystem,
    db_to_lin,
    range_resolution,
)
from radarlab.kinematics import DetectionGeometry

TerrainType: TypeAlias = Literal["Desert", "Farmland", "Grassland", "Woods", "Urban"]
SeaState: TypeAlias = Literal[1, 2, 3, 4, 5, 6]


class ClutterDoppler(BaseModel):
    """Gaussian clutter Doppler spectrum, in velocity terms (positive = closing)."""

    mean_velocity_m_s: float = 0.0
    sigma_velocity_m_s: float = Field(default=0.1, ge=0.0)


class GroundClutter(BaseModel):
    """Land clutter parameters."""

    type: Literal["Ground"] = "Ground"
    gamma_db: float = Field(
        default=-20,
        description="Override of the terrain's gamma (dB). Desert=-25.0, Farmland=-20.0, Grassland=-18.0, Woods=-15.0, Urban=-10.0",
    )


class SeaClutter(BaseModel):
    """Sea clutter parameters."""

    sea_state: SeaState = 3
    grazing_exponent: float = Field(default=1.5, gt=0.0)
    vv_minus_hh_db: float = 6.0


# Clutter physics
def ground_sigma0(clutter: GroundClutter, psi: float) -> float:
    """Land normalised reflectivity (linear) = gamma * sin(grazing angle)."""
    return db_to_lin(clutter.gamma_db) * math.sin(psi)


def sea_sigma0(clutter: SeaClutter, psi: float, mode: Mode) -> float:
    """Sea normalised reflectivity (linear).

    sigma0_HH = table(sea state) * (sin(psi) / sin(1°))^exponent, scaled linearly with
    frequency relative to 10 GHz. VV is `vv_minus_hh_db` above HH; the radar's
    polarisation angle blends them (cos^2 / sin^2).
    """
    psi = min(max(psi, math.radians(0.1)), math.radians(60.0))
    # Approximate X-band, HH, 1° grazing sigma0 (dB) per sea state, and typical wind (m/s).
    _SEA_SIGMA0_HH_1DEG_DB: Final[dict[SeaState, float]] = {
        1: -58.0,
        2: -52.0,
        3: -47.0,
        4: -43.0,
        5: -40.0,
        6: -37.0,
    }
    hh_db = (
        _SEA_SIGMA0_HH_1DEG_DB[clutter.sea_state]
        + 10.0
        * clutter.grazing_exponent
        * math.log10(math.sin(psi) / math.sin(math.radians(1.0)))
        + 10.0 * math.log10(mode.rf.carrier_frequency_hz / 10e9)
    )
    hh = db_to_lin(hh_db)
    vv = hh * db_to_lin(clutter.vv_minus_hh_db)
    a = math.radians(mode.polarisation.angle_deg)
    return hh * math.cos(a) ** 2 + vv * math.sin(a) ** 2


def clutter_sigma0(
    clutter: GroundClutter | SeaClutter, psi: float, mode: Mode
) -> float:
    """Normalised reflectivity (linear, m^2/m^2) of any clutter source."""
    match clutter:
        case GroundClutter():
            return ground_sigma0(clutter, psi)
        case SeaClutter():
            return sea_sigma0(clutter, psi, mode)
        case _:
            raise ValueError(f"Invalid clutter type:\n{clutter}")


def grazing_angle_rad(radar_height_m, range_m, target_height_m) -> float | None:
    if range_m <= 0:
        raise ValueError("range_m must be positive")
    ratio = (radar_height_m - target_height_m) / range_m
    psi = math.asin(max(-1.0, min(1.0, ratio)))
    if psi <= 0:
        return None
    else:
        return psi


def clutter_patch_area_m2(
    radar: RadarSystem, detection_geometry: DetectionGeometry
) -> float:
    """Illuminated clutter area (m^2) in the range cell at `range_m`.

    Azimuth extent = R * beamwidth. Ground-range extent is the smaller of the
    pulse-limited footprint (dR / cos(psi)) and the beam-limited footprint
    (R * theta_el / sin(psi)). The narrower of the transmit/receive beams is used.
    """
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
    tx, rx = radar.transmitter.beam, radar.receiver.beam
    theta_az = math.radians(min(tx.beamwidth_az_deg, rx.beamwidth_az_deg))
    theta_el = math.radians(min(tx.beamwidth_el_deg, rx.beamwidth_el_deg))
    pulse_limited = range_resolution(radar.mode) / max(math.cos(psi), 1e-6)
    beam_limited = line_of_sight.range_m * theta_el / max(math.sin(psi), 1e-6)
    return line_of_sight.range_m * theta_az * min(pulse_limited, beam_limited)

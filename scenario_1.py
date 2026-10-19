import logging
import math

import pymap3d

from clutter import GroundClutter, SeaClutter
from radar import (
    PRI,
    AntennaBeam,
    Mode,
    Polarisation,
    Power,
    Pulse,
    RadioFrequency,
    Receiver,
    Scan,
    Target,
    Transmitter,
    BarkerModulation,
    RectangularModulation,
    LFMModulation,
    RadarSystem,
)
from kinematics import (
    Attitude,
    DetectionGeometry,
    GeodeticPosition,
    Pose,
    target_from_ned,
    boresight_attitude,
)
from scenario import Scenario

def build_radar() -> RadarSystem:
    """A simple pulsed radar."""
    mode = Mode(
        rf=RadioFrequency(
            carrier_frequency_hz=3e9,
        ),
        pulse=Pulse(
            pulse_width_3dB_us=10.0,
            modulation=BarkerModulation(code_length=5),
            if_bandwidth_hz=3e6,
        ),
        pri=PRI(period_s=2e-3),
        power=Power(peak_power_dBW=32.0),
        polarisation=Polarisation(angle_deg=0.0),
        scan=Scan(scan_extent_az_deg=60.0, scan_extent_el_deg=60.0, frame_time_s=10.0),
    )
    beam = AntennaBeam(gain_dBi=35.0, beamwidth_az_deg=3.0, beamwidth_el_deg=3.0)
    transmitter = Transmitter(beam=beam)
    receiver = Receiver(
        beam=beam,
        noise_figure=6.0,
        integration_pulses=10,
        integration_method="NonCoherent",
    )
    return RadarSystem(transmitter=transmitter, receiver=receiver, mode=mode)


def build_scenario_1() -> Scenario:
    """Radar on the ground, target 20 km north, 10 km east and 3 km above it."""
    radar_position = GeodeticPosition(
        latitude_deg=-27.47, longitude_deg=153.03, altitude_m=50.0
    )
    target_latitude, target_longitude, target_altitude = pymap3d.ned2geodetic(
        20_000.0,
        10_000.0,
        -3_000.0,
        radar_position.latitude_deg,
        radar_position.longitude_deg,
        radar_position.altitude_m,
    )
    return Scenario(
        detection_geometry=DetectionGeometry(
            radar_pose=Pose(
                position=radar_position,
                attitude=Attitude(yaw_deg=10.0, pitch_deg=15.0, roll_deg=0.0),
            ),
            target_position=GeodeticPosition(
                latitude_deg=target_latitude,
                longitude_deg=target_longitude,
                altitude_m=target_altitude,
            ),
            target=Target(radar_cross_section_m2=1.0, swerling_case=1),
        ),
        clutter=GroundClutter(gamma_db=-20.0),
    )


def build_scenario_2() -> Scenario:
    """Co-altitude: airborne radar and target both at 5 km, target 25 km north
    and 15 km east. No ground/sea clutter in the main beam (horizon-looking)."""
    radar_position = GeodeticPosition(
        latitude_deg=-27.47, longitude_deg=153.03, altitude_m=5_000.0
    )
    north_m, east_m, down_m = 25_000.0, 15_000.0, 0.0
    return Scenario(
        detection_geometry=DetectionGeometry(
            radar_pose=Pose(
                position=radar_position,
                attitude=boresight_attitude(north_m, east_m, down_m),
            ),
            target_position=target_from_ned(radar_position, north_m, east_m, down_m),
            target=Target(radar_cross_section_m2=1.0, swerling_case=1),
        ),
        clutter=GroundClutter(),
    )


def build_scenario_3() -> Scenario:
    """Look-down over sea: airborne radar at 8 km, low target at 500 m,
    20 km north and 20 km east, all over open water east of Brisbane."""
    radar_position = GeodeticPosition(
        latitude_deg=-27.47, longitude_deg=154.2, altitude_m=8_000.0
    )
    north_m, east_m = 20_000.0, 20_000.0
    down_m = 7_500.0  # target at 500 m altitude
    return Scenario(
        detection_geometry=DetectionGeometry(
            radar_pose=Pose(
                position=radar_position,
                attitude=boresight_attitude(north_m, east_m, down_m),
            ),
            target_position=target_from_ned(radar_position, north_m, east_m, down_m),
            target=Target(radar_cross_section_m2=1.0, swerling_case=1),
        ),
        clutter=SeaClutter(sea_state=3),
    )

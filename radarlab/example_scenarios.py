from radarlab.kinematics import (
    DetectionGeometry,
    target_from_ned,
)
import pymap3d

from radarlab.clutter import GroundClutter, SeaClutter
from radarlab.radar import (
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
from radarlab.kinematics import (
    Attitude,
    DetectionGeometry,
    GeodeticPosition,
    Pose,
    target_from_ned,
    boresight_attitude,
)
from radarlab.scenario import Scenario


def build_radar_1() -> RadarSystem:
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


def build_radar_2() -> RadarSystem:
    """Notional X-band radar"""

    mode = Mode(
        rf=RadioFrequency(
            carrier_frequency_hz=10e9,  # X-band
        ),
        pulse=Pulse(
            pulse_width_3dB_us=13.0,
            modulation=BarkerModulation(
                code_length=13
            ),  # stand-in for pulse compression
            if_bandwidth_hz=2e6,
        ),
        pri=PRI(period_s=100e-6),  # ~10 kHz, medium PRF pulse-Doppler
        power=Power(peak_power_dBW=39.0),  # ~1,000 modules x ~8 W = ~8 kW peak
        polarisation=Polarisation(angle_deg=0.0),
        scan=Scan(scan_extent_az_deg=120.0, scan_extent_el_deg=120.0, frame_time_s=2.0),
    )
    # ~0.7 m aperture at 10 GHz: roughly 35 dBi and 3 deg beamwidth at boresight
    beam = AntennaBeam(gain_dBi=35.0, beamwidth_az_deg=3.0, beamwidth_el_deg=3.0)
    transmitter = Transmitter(beam=beam)
    receiver = Receiver(
        beam=beam,
        noise_figure=4.0,
        integration_pulses=32,
        integration_method="Coherent",
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


def build_scenario_2(east: float = 15_000) -> Scenario:
    """Co-altitude: airborne radar and target both at 5 km, target 25 km north
    and east km east. No ground/sea clutter in the main beam (horizon-looking)."""

    radar_position = GeodeticPosition(
        latitude_deg=-27.47, longitude_deg=153.03, altitude_m=5_000.0
    )
    north_m, east_m, down_m = 25_000, east, 0
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


def build_scenario_3(
    radar_altitude: float = 3_000, target_altitude: float = 500
) -> Scenario:
    """Look-down over sea: airborne radar at radar_altitude km, target at target_altitude m,
    20 km north and 20 km east, all over open water east of Brisbane."""
    radar_position = GeodeticPosition(
        latitude_deg=-27.47, longitude_deg=154.2, altitude_m=radar_altitude
    )
    north_m, east_m = 20_000, 20_000
    down_m = radar_position.altitude_m - target_altitude
    return Scenario(
        detection_geometry=DetectionGeometry(
            radar_pose=Pose(
                position=radar_position,
                attitude=boresight_attitude(north_m, east_m, down_m),
            ),
            target_position=target_from_ned(radar_position, north_m, east_m, down_m),
            target=Target(radar_cross_section_m2=1.0, swerling_case=3),
        ),
        clutter=SeaClutter(sea_state=1),
    )

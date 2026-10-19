import math

import matplotlib.pyplot as plt
import numpy as np
import pymap3d
from matplotlib.patches import Ellipse, Rectangle, Wedge

from main import DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY, radar_diagnostics, to_db
from scenario_1 import build_radar, build_scenario
from radar import (
    RadarSystem,
    max_detection_range,
    required_single_pulse_snr_db,
    snr_on_target,
)
from scenario import (
    Scenario,
    line_of_sight_in_body_frame,
    scenario_detection_margin_db,
    target_in_scan_volume,
)


def target_offset_ned_m(scenario: Scenario) -> tuple[float, float, float]:
    """Target position relative to the radar in local North-East-Down, in metres."""
    radar = scenario.radar_pose.position
    target = scenario.target_position
    return pymap3d.geodetic2ned(
        target.latitude_deg, target.longitude_deg, target.altitude_m,
        radar.latitude_deg, radar.longitude_deg, radar.altitude_m,
    )

def beam_sector_points(centre_angle_deg: float, width_deg: float, length_m: float, steps: int = 30):
    """Outline of a beam wedge from the origin. Angles are anticlockwise from the x-axis."""
    angles_rad = np.radians(
        np.linspace(centre_angle_deg - width_deg / 2.0, centre_angle_deg + width_deg / 2.0, steps)
    )
    x_points = np.concatenate([[0.0], length_m * np.cos(angles_rad)])
    y_points = np.concatenate([[0.0], length_m * np.sin(angles_rad)])
    return x_points, y_points


def plot_plan_view(axis, scenario: Scenario, radar: RadarSystem) -> None:
    """Top-down view: scan sector, steered beam, boresight and target, north up."""
    north_m, east_m, _ = target_offset_ned_m(scenario)
    yaw_deg = scenario.radar_pose.attitude.yaw_deg
    half_scan_az_deg = radar.mode.scan.scan_extent_az_deg / 2.0
    beamwidth_az_deg = radar.transmitter.beam.beamwidth_az_deg
    ground_range_m = math.hypot(north_m, east_m)
    reach_m = 1.25 * ground_range_m

    # Matplotlib angles run anticlockwise from east. Yaw runs clockwise from north.
    boresight_angle_deg = 90.0 - yaw_deg
    axis.add_patch(Wedge(
        (0.0, 0.0), reach_m,
        boresight_angle_deg - half_scan_az_deg, boresight_angle_deg + half_scan_az_deg,
        alpha=0.15, color="tab:blue", label="azimuth scan sector",
    ))
    boresight_x = reach_m * math.cos(math.radians(boresight_angle_deg))
    boresight_y = reach_m * math.sin(math.radians(boresight_angle_deg))
    axis.plot([0.0, boresight_x], [0.0, boresight_y], "--", color="tab:blue", label="boresight")

    # Beam steered onto the target
    line_of_sight_angle_deg = math.degrees(math.atan2(north_m, east_m))
    beam_x, beam_y = beam_sector_points(line_of_sight_angle_deg, beamwidth_az_deg, 1.1 * ground_range_m)
    axis.fill(beam_x, beam_y, color="tab:red", alpha=0.35, label=f"beam ({beamwidth_az_deg:g}° az)")

    beam_width_at_target_m = math.hypot(ground_range_m, 0.0) * math.radians(beamwidth_az_deg)
    axis.annotate(
        f"beam ≈ {beam_width_at_target_m:.0f} m wide at target",
        xy=(east_m, north_m), xytext=(10, -18), textcoords="offset points", fontsize=8,
    )

    axis.plot(0.0, 0.0, "k^", markersize=10, label="radar")
    axis.plot(east_m, north_m, "r*", markersize=14, label="target")

    axis.set_xlim(-reach_m, reach_m)
    axis.set_ylim(-0.1 * reach_m, reach_m)
    axis.set_aspect("equal")
    axis.set_xlabel("East (m)")
    axis.set_ylabel("North (m)")
    axis.set_title("Plan view")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.3)


def plot_side_view(axis, scenario: Scenario, radar: RadarSystem) -> None:
    """Vertical slice through the radar and target, with the elevation beam."""
    north_m, east_m, down_m = target_offset_ned_m(scenario)
    ground_range_m = math.hypot(north_m, east_m)
    height_m = -down_m
    slant_range_m = math.hypot(ground_range_m, height_m)
    elevation_above_horizon_deg = math.degrees(math.atan2(height_m, ground_range_m))
    beamwidth_el_deg = radar.transmitter.beam.beamwidth_el_deg

    beam_x, beam_y = beam_sector_points(elevation_above_horizon_deg, beamwidth_el_deg, 1.1 * slant_range_m)
    axis.fill(beam_x, beam_y, color="tab:red", alpha=0.35, label=f"beam ({beamwidth_el_deg:g}° el)")
    axis.plot([0.0, ground_range_m], [0.0, height_m], color="tab:red", label="line of sight")
    axis.plot([0.0, ground_range_m], [0.0, 0.0], ":", color="grey", label="radar horizontal")
    axis.plot([ground_range_m, ground_range_m], [0.0, height_m], ":", color="grey")
    axis.plot(0.0, 0.0, "k^", markersize=10, label="radar")
    axis.plot(ground_range_m, height_m, "r*", markersize=14, label="target")

    beam_height_at_target_m = slant_range_m * math.radians(beamwidth_el_deg)
    axis.annotate(
        f"{elevation_above_horizon_deg:.1f}° above horizon\nbeam ≈ {beam_height_at_target_m:.0f} m deep at target",
        xy=(ground_range_m * 0.35, height_m * 0.35 + 0.05 * abs(height_m)), fontsize=8,
    )

    axis.set_aspect("equal")  # equal scale, so the beam angle is drawn true
    axis.set_xlabel("Horizontal distance (m)")
    axis.set_ylabel("Height above radar (m)")
    axis.set_title("Side view (flat-earth sketch)")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(alpha=0.3)
    
def plot_body_frame_view(axis, scenario: Scenario, radar: RadarSystem) -> None:
    """Where the target sits in the radar's own azimuth/elevation coordinates."""
    line_of_sight = line_of_sight_in_body_frame(scenario.radar_pose, scenario.target_position)
    scan = radar.mode.scan
    beam = radar.transmitter.beam
    inside = target_in_scan_volume(radar, line_of_sight)

    axis.add_patch(Rectangle(
        (-scan.scan_extent_az_deg / 2.0, -scan.scan_extent_el_deg / 2.0),
        scan.scan_extent_az_deg, scan.scan_extent_el_deg,
        fill=False, edgecolor="tab:blue", linewidth=2, label="scan volume",
    ))
    axis.add_patch(Ellipse(
        (line_of_sight.azimuth_deg, line_of_sight.elevation_deg),
        beam.beamwidth_az_deg, beam.beamwidth_el_deg,
        fill=False, edgecolor="tab:red", linestyle="--", label="beam footprint on target",
    ))
    axis.plot(0.0, 0.0, "b+", markersize=12, label="boresight")
    axis.plot(
        line_of_sight.azimuth_deg, line_of_sight.elevation_deg, "r*", markersize=14,
        label=f"target ({'inside' if inside else 'outside'} scan)",
    )

    limit_deg = max(
        scan.scan_extent_az_deg / 2.0, scan.scan_extent_el_deg / 2.0,
        abs(line_of_sight.azimuth_deg), abs(line_of_sight.elevation_deg),
    ) * 1.3
    axis.set_xlim(-limit_deg, limit_deg)
    axis.set_ylim(-limit_deg, limit_deg)
    axis.set_aspect("equal")
    axis.set_xlabel("Azimuth from nose (deg, right positive)")
    axis.set_ylabel("Elevation from nose (deg, up positive)")
    axis.set_title("Radar body frame")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.3)


def plot_snr_against_range(
    axis, scenario: Scenario, radar: RadarSystem
) -> None:
    """Available SNR against range, with the required SNR for each Swerling case."""
    line_of_sight = line_of_sight_in_body_frame(scenario.radar_pose, scenario.target_position)
    scenario_limit_m = max_detection_range(
        radar, scenario.target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
    )
    ranges_m = np.logspace(3.0, math.log10(3.0 * max(scenario_limit_m, line_of_sight.range_m)), 300)
    available_db = [
        to_db(snr_on_target(radar, scenario.target, range_m)) for range_m in ranges_m
    ]
    axis.plot(ranges_m / 1000.0, available_db, color="black", linewidth=2, label="available single-pulse SNR")

    for swerling_case in (0, 1, 2, 3, 4):
        target = scenario.target.model_copy(update={"swerling_case": swerling_case})
        try:
            required_db = required_single_pulse_snr_db(
                radar.receiver, target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
            )
        except ValueError:
            continue  # this case is invalid for the receiver's integration method
        axis.axhline(
            required_db, linestyle="--", alpha=0.8 if swerling_case == scenario.target.swerling_case else 0.4,
            label=f"required, Swerling {swerling_case}",
        )

    axis.axvline(line_of_sight.range_m / 1000.0, color="tab:red", label="target range")
    axis.axvline(scenario_limit_m / 1000.0, color="tab:green", linestyle=":", label="max detection range")

    margin_db = scenario_detection_margin_db(
        scenario, radar, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
    )
    margin_text = "out of scan volume" if margin_db is None else f"margin {margin_db:+.1f} dB"
    axis.set_title(
        f"SNR budget (Pd={DETECTION_PROBABILITY}, Pfa={FALSE_ALARM_PROBABILITY:g}): {margin_text}"
    )
    axis.set_xscale("log")
    axis.set_xlabel("Range (km)")
    axis.set_ylabel("SNR (dB)")
    axis.legend(fontsize=7, loc="upper right")
    axis.grid(alpha=0.3, which="both")


def visualise(scenario: Scenario, radar: RadarSystem, save_path: str | None = None):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_plan_view(axes[0, 0], scenario, radar)
    plot_side_view(axes[0, 1], scenario, radar)
    plot_body_frame_view(axes[1, 0], scenario, radar)
    plot_snr_against_range(axes[1, 1], scenario, radar)
    figure.tight_layout()
    if save_path:
        figure.savefig(save_path, dpi=150)
    return figure


def main() -> None:
    radar = build_radar()
    scenario = build_scenario()
    radar_diagnostics(radar)
    visualise(scenario, radar, save_path="radar_run.png")
    plt.show()


if __name__ == "__main__":
    main()
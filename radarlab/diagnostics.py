import logging
import math

from radarlab.clutter import (
    clutter_patch_area_m2,
    clutter_sigma0,
    grazing_angle_rad,
)
from radarlab.radar import (
    RadarSystem,
    angular_resolution_az,
    angular_resolution_el,
    closed_form_max_detection_range,
    max_unambiguous_range,
    minimum_discernible_signal,
    noise_power,
    echo_power_w,
    range_resolution,
    required_single_pulse_snr_db,
    snr_on_target,
    to_db,
)
from radarlab.scenario import (
    DETECTION_PROBABILITY,
    FALSE_ALARM_PROBABILITY,
    DetectionGeometry,
    Scenario,
    clutter_power_w,
    clutter_rcs_m2,
    line_of_sight_in_body_frame,
    optimise_for_max_detection_range,
    scenario_snr_margin_db,
    scenario_sinr_db,
    scenario_snr_db,
    target_in_scan_volume,
)
from radarlab.example_scenarios import (
    build_radar_1,
    build_radar_2,
    build_scenario_1,
    build_scenario_2,
    build_scenario_3,
)

logger = logging.getLogger("radar_lab")


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=== %s ===", title)


def log_radar_characteristics(radar: RadarSystem) -> None:
    log_section("Radar characteristics")
    wavelength_m = 3e8 / radar.mode.rf.carrier_frequency_hz
    logger.info("wavelength_m                 = %.6f", wavelength_m)
    logger.info(
        "max_unambiguous_range_m      = %.1f", max_unambiguous_range(radar.mode)
    )
    logger.info("range_resolution_m           = %.1f", range_resolution(radar.mode))
    logger.info(
        "angular_resolution_az_deg    = %.2f",
        angular_resolution_az(radar.receiver.beam),
    )
    logger.info(
        "angular_resolution_el_deg    = %.2f",
        angular_resolution_el(radar.receiver.beam),
    )
    logger.info("pulses_integrated            = %d", radar.receiver.integration_pulses)
    receiver_noise_w = noise_power(radar)
    logger.info(
        "noise_power_w                = %.3e (%.1f dBW)",
        receiver_noise_w,
        to_db(receiver_noise_w),
    )
    mds_w = minimum_discernible_signal(radar)
    logger.info("minimum_discernible_signal_w = %.3e (%.1f dBW)", mds_w, to_db(mds_w))


def log_geometry(detection_geometry: DetectionGeometry, radar: RadarSystem) -> None:
    log_section("Geometry")
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    range_nmi = line_of_sight.range_m / 1852.0
    logger.info("range_m                      = %.1f", line_of_sight.range_m)
    logger.info("range_nmi                    = %.1f", range_nmi)
    logger.info("azimuth_deg (body frame)     = %.3f", line_of_sight.azimuth_deg)
    logger.info("elevation_deg (body frame)   = %.3f", line_of_sight.elevation_deg)
    logger.info(
        "in_scan_volume               = %s",
        target_in_scan_volume(radar, line_of_sight),
    )


def log_clutter(scenario: Scenario, radar: RadarSystem) -> None:
    log_section("Clutter")
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    psi = grazing_angle_rad(
        detection_geometry.radar_pose.position.altitude_m,
        line_of_sight.range_m,
        detection_geometry.target_position.altitude_m,
    )
    area_m2 = clutter_patch_area_m2(radar, scenario.detection_geometry)
    rcs_m2 = clutter_rcs_m2(scenario, radar)
    power_w = clutter_power_w(radar, scenario)
    if psi is not None:
        sigma0 = clutter_sigma0(scenario.clutter, psi, radar.mode)
        logger.info("grazing_angle_deg            = %.3f", math.degrees(psi))
        logger.info("sigma0                       = %.3e", sigma0)
        logger.info(
            "clutter_power_w              = %.3e (%.1f dBW)",
            power_w,
            to_db(power_w),
        )
    else:
        logger.info("grazing_angle_deg            = None")
        logger.info("sigma0                       = N/A")

    logger.info("clutter_patch_area_m2        = %.3e", area_m2)
    logger.info("clutter_rcs_m2               = %.3e", rcs_m2)


def log_link_budget(scenario: Scenario, radar: RadarSystem) -> None:
    log_section("Link budget")
    detection_geometry = scenario.detection_geometry
    line_of_sight = line_of_sight_in_body_frame(
        detection_geometry.radar_pose, detection_geometry.target_position
    )
    echo_power = echo_power_w(
        radar, detection_geometry.target.radar_cross_section_m2, line_of_sight.range_m
    )
    rx_noise_power = noise_power(radar)
    clutter_power = clutter_power_w(radar, scenario)
    snr_linear = snr_on_target(radar, detection_geometry.target, line_of_sight.range_m)
    logger.info(
        "echo_power_w                 = %.3e (%.1f dBW)",
        echo_power,
        to_db(echo_power),
    )
    logger.info(
        "noise_power                  = %.3e (%.1f dBW)",
        rx_noise_power,
        to_db(rx_noise_power),
    )
    if abs(clutter_power) > 1e-9:
        logger.info(
            "clutter_power_w              = %.3e (%.1f dBW)",
            clutter_power,
            to_db(clutter_power),
        )
    else:
        logger.info(
            "clutter_power_w              = N/A",
        )

    logger.info("single_pulse_snr_db          = %.2f", to_db(snr_linear))
    logger.info(
        "scenario_sinr_db             = %.2f", scenario_sinr_db(scenario, radar)
    )
    logger.info(
        "scenario_snr_db              = %s",
        scenario_snr_db(scenario, radar),
    )

    required_db = required_single_pulse_snr_db(
        radar.receiver,
        scenario.detection_geometry.target,
        DETECTION_PROBABILITY,
        FALSE_ALARM_PROBABILITY
        ,
    )
    margin_db = scenario_snr_margin_db(
        scenario, radar, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
    )
    logger.info("required_single_pulse_snr_db = %.2f", required_db)
    logger.info(
        "detection_margin_db          = %s",
        None if margin_db is None else round(margin_db, 2),
    )
    logger.info(
        "detectable                   = %s", margin_db is not None and margin_db > 0.0
    )


def log_max_range(scenario: Scenario, radar: RadarSystem) -> None:
    """Logs maximum detection range and SNR."""
    log_section("Maximum detection range (closed form, no clutter)")
    range_limit_m = closed_form_max_detection_range(
        radar,
        scenario.detection_geometry.target,
        DETECTION_PROBABILITY,
        FALSE_ALARM_PROBABILITY,
    )
    range_limit_nmi = range_limit_m / 1852.0
    snr_at_limit_db = to_db(
        snr_on_target(radar, scenario.detection_geometry.target, range_limit_m)
    )
    logger.info("max_detection_range_m        = %.1f", range_limit_m)
    logger.info("max_detection_range_nmi      = %.3f", range_limit_nmi)
    logger.info("snr_at_max_range_db          = %.3f", snr_at_limit_db)


def log_swerling_sweep(scenario: Scenario, radar: RadarSystem) -> None:
    log_section("Swerling case sweep (non-coherent)")
    line_of_sight = line_of_sight_in_body_frame(
        scenario.detection_geometry.radar_pose,
        scenario.detection_geometry.target_position,
    )
    available_db = to_db(
        snr_on_target(radar, scenario.detection_geometry.target, line_of_sight.range_m)
    )
    logger.info(
        "available_single_pulse_snr_db = %.2f at range %.1f m",
        available_db,
        line_of_sight.range_m,
    )
    for swerling_case in (0, 1, 2, 3, 4):
        target = scenario.detection_geometry.target.model_copy(
            update={"swerling_case": swerling_case}
        )
        try:
            required_db = required_single_pulse_snr_db(
                radar.receiver, target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
            )
        except:
            logger.info(
                "swerling %d: N/A",
                swerling_case,
            )
            continue

        range_limit_m = closed_form_max_detection_range(
            radar,
            target,
            DETECTION_PROBABILITY,
            FALSE_ALARM_PROBABILITY,
        )
        logger.info(
            "swerling %d: required_snr_db = %6.2f | margin_db = %6.2f | max_detection_range_m = %8.1f | max_detection_range_nmi = %6.3f",
            swerling_case,
            required_db,
            available_db - required_db,
            range_limit_m,
            range_limit_m / 1852.0,
        )


def log_optimised_for_max_detection_range(
    scenario: Scenario,
    radar: RadarSystem,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float,
) -> None:
    log_section("Optimised for max detection range")
    try:
        optimized = optimise_for_max_detection_range(
            scenario,
            radar,
            detection_probability_threshold,
            false_alarm_probability_threshold,
        )
        line_of_sight = line_of_sight_in_body_frame(
            optimized.detection_geometry.radar_pose,
            optimized.detection_geometry.target_position,
        )
        logger.info("optimised_target_range_m    = %.1f", line_of_sight.range_m)
        logger.info(
            "optimised_target_range_nmi  = %.3f", line_of_sight.range_m / 1852.0
        )
        logger.info(
            "scenario_sinr_db            = %s", scenario_sinr_db(optimized, radar)
        )
    except ValueError:
        logger.info("UNDETECTABLE")


def radar_diagnostics(radar: RadarSystem, scenario: Scenario) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("radar_run.log", mode="w"),
        ],
    )

    log_section("Inputs")
    logger.info("detection_probability        = %s", DETECTION_PROBABILITY)
    logger.info("false_alarm_probability      = %s", FALSE_ALARM_PROBABILITY)
    logger.info("radar = %s", radar.model_dump_json(indent=2))
    logger.info("scenario = %s", scenario.model_dump_json(indent=2))

    log_radar_characteristics(radar)
    log_geometry(scenario.detection_geometry, radar)
    log_clutter(scenario, radar)
    log_max_range(scenario, radar)
    log_swerling_sweep(scenario, radar)
    log_optimised_for_max_detection_range(
        scenario,
        radar,
        DETECTION_PROBABILITY,
        FALSE_ALARM_PROBABILITY,
    )
    log_link_budget(scenario, radar)

def main():
    radar = build_radar_2()
    scenario = build_scenario_3(radar_altitude=1100, target_altitude=10)
    radar_diagnostics(radar, scenario)


if __name__ == "__main__":
    main()
    
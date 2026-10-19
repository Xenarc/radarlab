import logging
import math

from radar import (
    RadarSystem,
    angular_resolution_az,
    angular_resolution_el,
    max_detection_range,
    max_unambiguous_range,
    minimum_discernible_signal,
    noise_power,
    power_on_target,
    range_resolution,
    required_single_pulse_snr_db,
    snr_on_target,
)
from scenario import (
    Scenario,
    line_of_sight_in_body_frame,
    scenario_detection_margin_db,
    scenario_snr_db,
    target_in_scan_volume,
)
from scenario_1 import build_radar, build_scenario

logger = logging.getLogger("radar_lab")

DETECTION_PROBABILITY = 0.9
FALSE_ALARM_PROBABILITY = 1e-6


def to_db(linear: float) -> float:
    return 10.0 * math.log10(linear)


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
        "angular_resolution_az_deg    = %.2f", angular_resolution_az(radar.receiver.beam)
    )
    logger.info(
        "angular_resolution_el_deg    = %.2f", angular_resolution_el(radar.receiver.beam)
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


def log_geometry(scenario: Scenario, radar: RadarSystem) -> None:
    log_section("Geometry")
    line_of_sight = line_of_sight_in_body_frame(
        scenario.radar_pose, scenario.target_position
    )
    logger.info("range_m                      = %.1f", line_of_sight.range_m)
    logger.info("azimuth_deg (body frame)     = %.3f", line_of_sight.azimuth_deg)
    logger.info("elevation_deg (body frame)   = %.3f", line_of_sight.elevation_deg)
    logger.info(
        "in_scan_volume               = %s",
        target_in_scan_volume(radar, line_of_sight),
    )


def log_link_budget(
    scenario: Scenario, radar: RadarSystem
) -> None:
    log_section("Link budget")
    line_of_sight = line_of_sight_in_body_frame(
        scenario.radar_pose, scenario.target_position
    )
    echo_power_w = power_on_target(
        radar, scenario.target, line_of_sight.range_m
    )
    snr_linear = snr_on_target(
        radar, scenario.target, line_of_sight.range_m
    )
    logger.info(
        "echo_power_w                 = %.3e (%.1f dBW)",
        echo_power_w,
        to_db(echo_power_w),
    )
    logger.info("single_pulse_snr_db          = %.2f", to_db(snr_linear))
    logger.info(
        "scenario_snr_db              = %s",
        scenario_snr_db(scenario, radar),
    )

    required_db = required_single_pulse_snr_db(
        radar.receiver, scenario.target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
    )
    margin_db = scenario_detection_margin_db(
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


def log_max_range_consistency(
    scenario: Scenario, radar: RadarSystem
) -> None:
    """At the maximum detection range, available SNR must equal required SNR."""
    log_section("Maximum detection range and self-check")
    range_limit_m = max_detection_range(
        radar,
        scenario.target,
        DETECTION_PROBABILITY,
        FALSE_ALARM_PROBABILITY,
    )
    range_limit_nmi = range_limit_m / 1852.0
    snr_at_limit_db = to_db(
        snr_on_target(radar, scenario.target, range_limit_m)
    )
    required_db = required_single_pulse_snr_db(
        radar.receiver, scenario.target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
    )
    logger.info("max_detection_range_m        = %.1f", range_limit_m)
    logger.info("max_detection_range_nmi      = %.3f", range_limit_nmi)
    logger.info("snr_at_max_range_db          = %.3f", snr_at_limit_db)
    logger.info("required_snr_db              = %.3f", required_db)
    logger.info("mismatch_db (should be ~0)   = %.6f", snr_at_limit_db - required_db)


def log_swerling_sweep(
    scenario: Scenario, radar: RadarSystem
) -> None:
    log_section("Swerling case sweep (non-coherent)")
    line_of_sight = line_of_sight_in_body_frame(
        scenario.radar_pose, scenario.target_position
    )
    available_db = to_db(
        snr_on_target(radar, scenario.target, line_of_sight.range_m)
    )
    logger.info(
        "available_single_pulse_snr_db = %.2f at range %.1f m",
        available_db,
        line_of_sight.range_m,
    )
    for swerling_case in (0, 1, 2, 3, 4):
        target = scenario.target.model_copy(update={"swerling_case": swerling_case})
        required_db = required_single_pulse_snr_db(
            radar.receiver, target, DETECTION_PROBABILITY, FALSE_ALARM_PROBABILITY
        )
        range_limit_m = max_detection_range(
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


def radar_diagnostics(radar: RadarSystem) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("radar_run.log", mode="w"),
        ],
    )

    scenario = build_scenario()

    log_section("Inputs")
    logger.info("detection_probability        = %s", DETECTION_PROBABILITY)
    logger.info("false_alarm_probability      = %s", FALSE_ALARM_PROBABILITY)
    logger.info("radar = %s", radar.model_dump_json(indent=2))
    logger.info("scenario = %s", scenario.model_dump_json(indent=2))

    log_radar_characteristics(radar)
    log_geometry(scenario, radar)
    log_link_budget(scenario, radar)
    log_max_range_consistency(scenario, radar)
    log_swerling_sweep(scenario, radar)


if __name__ == "__main__":
    radar = build_radar()
    radar_diagnostics(radar)

# Approximate one-way specific attenuation at sea level, standard atmosphere
# (oxygen + ~7.5 g/m^3 water vapour). Rough ITU-R P.676 values.
# (A bit dodgy around 22 GHz or above 30 GHz)
import numpy as np

from radarlab.radar import RadarSystem


_FREQ_GHZ = np.array([1.0, 3.0, 10.0, 15.0, 22.0, 35.0])
_GAMMA_DB_PER_KM = np.array([0.0055, 0.0065, 0.013, 0.035, 0.18, 0.13])

_ATMOSPHERE_SCALE_HEIGHT_M = 5_000.0  # effective e-folding height of absorber density


def specific_attenuation_db_per_km(frequency_hz: float) -> float:
    """One-way sea-level specific attenuation (dB/km), log-frequency interpolation."""
    return float(
        np.interp(np.log10(frequency_hz / 1e9), np.log10(_FREQ_GHZ), _GAMMA_DB_PER_KM)
    )


def atmospheric_loss(
    radar: RadarSystem,
    range_m: float,
    radar_altitude_m: float,
    target_altitude_m: float,
) -> float:
    """Two-way atmospheric power transmission (linear, 0..1) along a straight path.

    Attenuation falls off exponentially with altitude, so the path integral
    of gamma0 * exp(-h/H) along the line of sight is done analytically.
    """
    gamma0 = specific_attenuation_db_per_km(radar.mode.rf.carrier_frequency_hz)
    h1, h2, H = radar_altitude_m, target_altitude_m, _ATMOSPHERE_SCALE_HEIGHT_M

    if abs(h2 - h1) < 1.0:
        effective_range_m = range_m * np.exp(-h1 / H)
    else:
        effective_range_m = (
            range_m * H * (np.exp(-h1 / H) - np.exp(-h2 / H)) / (h2 - h1)
        )

    one_way_db = gamma0 * effective_range_m / 1_000.0
    return float(10.0 ** (-2.0 * one_way_db / 10.0))

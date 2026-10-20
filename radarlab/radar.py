import math
from typing import Annotated, Final, Literal, Protocol, TypeAlias, Union

from pydantic import BaseModel, Field, model_validator

# Constants
SPEED_OF_LIGHT_M_S: Final[float] = 3e8
BOLTZMANN_J_PER_K: Final[float] = 1.38e-23
REFERENCE_TEMPERATURE_K: Final[float] = 290.0

# Type aliases
SwerlingCase: TypeAlias = Literal[0, 1, 2, 3, 4]  # 0 = non-fluctuating, 1-4 = Swerling I-IV
IntegrationMethod: TypeAlias = Literal["Coherent", "NonCoherent"]


class Target(BaseModel):
    """A radar target: how large it looks and how its echo fluctuates."""

    radar_cross_section_m2: float = Field(default=1.0, gt=0.0)
    swerling_case: SwerlingCase = Field(
        default=1,
        description="0 = non-fluctuating, 1-4 = Swerling I-IV",
    )


class RadioFrequency(BaseModel):
    """Radio frequency parameters for a radar system."""

    carrier_frequency_hz: float = Field(default=10e9, gt=0.0)


class RectangularModulation(BaseModel):
    """Unmodulated pulse. Bandwidth ~ 1/tau, so there is no compression."""

    type: Literal["Rectangular"] = "Rectangular"

    def bandwidth_hz(self, pulse_width_s: float) -> float:
        return 1.0 / pulse_width_s

    def compression_ratio(self, pulse_width_s: float) -> float:
        return 1.0


class LFMModulation(BaseModel):
    """Linear FM chirp. Bandwidth is a free design parameter."""

    type: Literal["LFM"] = "LFM"
    swept_bandwidth_hz: float = Field(
        gt=0.0, alias="bandwidth_hz", description="Swept bandwidth in Hz"
    )

    def bandwidth_hz(self, pulse_width_s: float) -> float:
        return self.swept_bandwidth_hz

    def compression_ratio(self, pulse_width_s: float) -> float:
        return self.swept_bandwidth_hz * pulse_width_s  # time-bandwidth product


class BarkerModulation(BaseModel):
    """Barker phase code. Chip width = tau / N, so bandwidth ~ N / tau."""

    type: Literal["Barker"] = "Barker"
    code_length: Literal[2, 3, 4, 5, 7, 11, 13] = Field(
        description="Number of chips in the Barker code"
    )

    def bandwidth_hz(self, pulse_width_s: float) -> float:
        return self.code_length / pulse_width_s

    def compression_ratio(self, pulse_width_s: float) -> float:
        return float(self.code_length)


class PulseModulation(Protocol):

    def bandwidth_hz(self, pulse: "Pulse"): ...
    def compression_ratio(self, pulse: "Pulse"): ...


class Pulse(BaseModel):
    """Pulse parameters for a radar system."""

    pulse_width_3dB_us: float = Field(default=1.0, gt=0.0)
    modulation: BarkerModulation | RectangularModulation | LFMModulation = Field(
        default_factory=RectangularModulation
    )
    if_bandwidth_hz: float = Field(
        default=1e6, gt=0.0, description="Intermediate frequency bandwidth in Hz"
    )

    @model_validator(mode="after")
    def _if_covers_signal(self):
        bw = self.modulation.bandwidth_hz(self.pulse_width_3dB_us * 1e-6)
        if self.if_bandwidth_hz < bw:
            raise ValueError(
                f"if_bandwidth_hz ({self.if_bandwidth_hz:.3g}) is below "
                f"the signal bandwidth ({bw:.3g})"
            )
        return self


class Power(BaseModel):
    """Power parameters for a radar system."""

    peak_power_dBW: float = Field(
        default=30.0,
        description="Peak transmit power in dBW (decibels relative to 1 W)",
    )


class PRI(BaseModel):
    """Pulse repetition interval for a radar system."""

    period_s: float = Field(default=1e-3, gt=0.0)


class Polarisation(BaseModel):
    """Polarisation parameters for a radar system."""

    angle_deg: float = Field(
        default=0.0,
        ge=0.0,
        le=90.0,
        description="Polarisation angle in degrees: 0° for horizontal, 90° for vertical, etc.",
    )


class Scan(BaseModel):
    """Scan parameters for a radar system."""

    scan_extent_az_deg: float = Field(default=30.0, gt=0.0, le=360.0)
    scan_extent_el_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    frame_time_s: float = Field(default=1.0, gt=0.0)


class Mode(BaseModel):
    rf: RadioFrequency
    pulse: Pulse
    pri: PRI
    power: Power
    polarisation: Polarisation
    scan: Scan


class AntennaBeam(BaseModel):
    """Antenna beam parameters for a radar system."""

    gain_dBi: float = Field(
        default=30.0, gt=0.0, description="Gain of the antenna beam in dBi"
    )
    beamwidth_az_deg: float = Field(
        default=3.0, gt=0.0, le=180.0, description="Beamwidth in azimuth in degrees"
    )
    beamwidth_el_deg: float = Field(
        default=3.0, gt=0.0, le=180.0, description="Beamwidth in elevation in degrees"
    )


class Transmitter(BaseModel):
    """A radar transmitter with the core physics needed for pulsed-radar synthesis."""

    beam: AntennaBeam
    losses_db: float = Field(default=6.0, gt=0.0)


class Receiver(BaseModel):
    """A radar receiver with the core physics needed for pulsed-radar synthesis."""

    beam: AntennaBeam
    noise_figure: float = Field(default=8.0, gt=0.0)
    losses_db: float = Field(default=3.0, gt=0.0)

    integration_pulses: int = Field(
        default=1, gt=0, description="Number of pulses integrated"
    )
    integration_method: IntegrationMethod = Field(
        default="Coherent", description="Integration method for the processor"
    )


class RadarSystem(BaseModel):
    """A radar system with the core physics needed for pulsed-radar synthesis."""

    transmitter: Transmitter
    receiver: Receiver
    mode: Mode


# Helpers
def db_to_lin(x_db: float) -> float:
    """Convert a decibel quantity to a linear ratio."""
    return 10 ** (x_db / 10)


def to_db(linear: float) -> float:
    return 10.0 * math.log10(linear)


def wavelength_m(mode: Mode) -> float:
    """Carrier wavelength (m)."""
    return SPEED_OF_LIGHT_M_S / mode.rf.carrier_frequency_hz


# Physics
def max_unambiguous_range(mode: Mode) -> float:
    """Maximum unambiguous range (m) = c * PRI / 2."""
    return (SPEED_OF_LIGHT_M_S * mode.pri.period_s) / 2


def signal_bandwidth_hz(pulse: Pulse) -> float:
    """Occupied bandwidth (Hz) of the transmitted waveform."""
    return pulse.modulation.bandwidth_hz(pulse.pulse_width_3dB_us * 1e-6)


def range_resolution(mode: Mode) -> float:
    """Compressed range resolution (m) = c / (2 * B_signal)."""
    return SPEED_OF_LIGHT_M_S / (2.0 * signal_bandwidth_hz(mode.pulse))


def pulse_compression_gain(mode: Mode) -> float:
    """Matched-filter pulse compression gain (linear) = time-bandwidth product."""
    pulse = mode.pulse
    return pulse.modulation.compression_ratio(pulse.pulse_width_3dB_us * 1e-6)

def angular_resolution_az(beam: AntennaBeam) -> float:
    """Azimuth angular resolution (deg), taken as the azimuth beamwidth."""
    return beam.beamwidth_az_deg


def angular_resolution_el(beam: AntennaBeam) -> float:
    """Elevation angular resolution (deg), taken as the elevation beamwidth."""
    return beam.beamwidth_el_deg


def eirp_w(transmitter: Transmitter, mode: Mode) -> float:
    """Equivalent isotropic radiated power (W) in the direction of interest.

    EIRP = Pt * Gt / Lt, evaluated in dB:
        peak_power_dBW + gain_dBi - losses_db

    Gt is the transmit antenna gain toward the target. By default this is the
    boresight (peak) gain. Losses are given in dB and are transmit-side only (feed,
    radome, etc.).
    """
    return db_to_lin(
        mode.power.peak_power_dBW
        + transmitter.beam.gain_dBi
        - transmitter.losses_db
    )

def effective_receive_aperture_m2(
    receiver: Receiver, mode: Mode
) -> float:
    """Effective receive aperture (m^2) = Gr * lambda^2 / (4 * pi * Lr)."""
    return (
        db_to_lin(receiver.beam.gain_dBi - receiver.losses_db)
        * wavelength_m(mode) ** 2
        / (4 * math.pi)
    )


def echo_power_w(radar: RadarSystem, rcs_m2: float, range_m: float) -> float:
    """Received echo power (W) at the receiver input for any scatterer of RCS `rcs_m2`.

    EIRP * sigma * Ae / ((4*pi)^2 * R^4), boresight gain assumed.
    """

    return (
        eirp_w(radar.transmitter, radar.mode)
        * rcs_m2
        * effective_receive_aperture_m2(radar.receiver, radar.mode)
    ) / ((4 * math.pi) ** 2 * range_m**4)


def noise_power(radar: RadarSystem) -> float:
    """Receiver noise power (W) = k * T0 * B_sig * F.

    B_sig is the occupied bandwidth of the transmitted waveform, i.e. the
    matched-filter noise bandwidth. The IF bandwidth only needs to cover it
    (enforced by the Pulse validator) and does not set the post-compression
    noise.
    """
    noise_figure_linear = db_to_lin(radar.receiver.noise_figure)
    bandwidth_hz = signal_bandwidth_hz(radar.mode.pulse)
    return (
        BOLTZMANN_J_PER_K
        * REFERENCE_TEMPERATURE_K
        * bandwidth_hz
        * noise_figure_linear
    )


def minimum_discernible_signal(radar: RadarSystem) -> float:
    """Minimum discernible signal (W), referred to the receiver input.

    MDS is the input signal power equal to the receiver noise power, so antenna
    gains do not enter. `transmitter` is unused and only kept so the signature
    matches noise_power().
    """
    return noise_power(radar)


def snr_on_target(
    radar:RadarSystem, target: Target, range_m: float
) -> float:
    """Single-pulse SNR (linear) after pulse compression, before integration."""
    return (
        echo_power_w(radar, target.radar_cross_section_m2, range_m)
        * pulse_compression_gain(radar.mode)
        / noise_power(radar)
    )


def required_snr_db(
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float,
    swerling_case: SwerlingCase,
    pulse_count: int = 1,
) -> float:
    """Required per-pulse SNR (dB) for non-coherent integration of `pulse_count` pulses.

    Shnidman's equation. swerling_case: 0 = non-fluctuating, 1-4 = Swerling I-IV.
    Valid for 0.1 <= Pd <= 0.99, roughly 1e-9 <= Pfa <= 1e-3, 1 <= pulse_count <~ 100.
    """
    detection_probability: float = detection_probability_threshold
    false_alarm_probability: float = false_alarm_probability_threshold

    if not 0.1 <= detection_probability <= 0.99:
        raise ValueError("Shnidman is valid for 0.1 <= detection probability <= 0.99")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false alarm probability must be in (0, 1)")
    if pulse_count < 1:
        raise ValueError("pulse_count must be >= 1")

    # Fluctuation shape parameter: infinity means no fluctuation
    fluctuation_shape: float = {
        0: math.inf,
        1: 1.0,
        2: float(pulse_count),
        3: 2.0,
        4: 2.0 * pulse_count,
    }[swerling_case]

    # Detectability factor for a non-fluctuating target integrating pulse_count pulses
    detection_margin: float = math.sqrt(
        -0.8 * math.log(4 * false_alarm_probability * (1 - false_alarm_probability))
    ) + math.copysign(1.0, detection_probability - 0.5) * math.sqrt(
        -0.8 * math.log(4 * detection_probability * (1 - detection_probability))
    )
    many_pulse_correction: float = 0.25 if pulse_count >= 40 else 0.0
    detectability_factor: float = detection_margin * (
        detection_margin
        + 2.0 * math.sqrt(pulse_count / 2.0 + many_pulse_correction - 0.25)
    )

    # Fluctuation loss (dB)
    fluctuation_loss_db: float
    if math.isinf(fluctuation_shape):
        fluctuation_loss_db = 0.0
    else:
        fluctuation_loss_db = (
            (
                (17.7006 * detection_probability - 18.4496) * detection_probability
                + 14.5339
            )
            * detection_probability
            - 3.525
        ) / fluctuation_shape
        if detection_probability > 0.872:
            high_probability_extra_loss_db: float = (
                math.exp(27.31 * detection_probability - 25.14)
                + (detection_probability - 0.8)
                * (
                    0.7 * math.log(1e-5 / false_alarm_probability)
                    + (2 * pulse_count - 20) / 80.0
                )
            ) / fluctuation_shape
            fluctuation_loss_db += high_probability_extra_loss_db

    return fluctuation_loss_db + 10.0 * math.log10(detectability_factor / pulse_count)


def required_single_pulse_snr_db(
    receiver: Receiver,
    target: Target,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float = 1e-6,
) -> float:
    """Required SNR (dB) of one pulse, given how this receiver integrates.

    NonCoherent: Shnidman with the full pulse count.
    Coherent: treated as one pulse with N times the energy, so the requirement
    drops by 10*log10(N). Only physically valid if the echo stays correlated
    over the dwell, which rules out Swerling II and IV (pulse-to-pulse fluctuation).
    """
    pulse_count: int = receiver.integration_pulses
    method: IntegrationMethod = receiver.integration_method

    if method == "NonCoherent":
        return required_snr_db(
            detection_probability_threshold,
            false_alarm_probability_threshold,
            target.swerling_case,
            pulse_count,
        )

    if target.swerling_case in (2, 4):
        raise ValueError(
            "Coherent integration is invalid for Swerling II/IV: the echo decorrelates between pulses"
        )
    single_pulse_requirement_db: float = required_snr_db(
        detection_probability_threshold,
        false_alarm_probability_threshold,
        target.swerling_case,
        1,
    )
    return single_pulse_requirement_db - 10.0 * math.log10(pulse_count)


def closed_form_max_detection_range(
    radar: RadarSystem,
    target: Target,
    detection_probability_threshold: float,
    false_alarm_probability_threshold: float = 1e-6,
) -> float:
    """Maximum detection range (m): the radar range equation solved for range.

    Uses the same building blocks as snr_on_target(), so at the returned range
    snr_on_target() equals the required single-pulse SNR.
    """
    required_snr_linear: float = db_to_lin(
        required_single_pulse_snr_db(
            radar.receiver,
            target,
            detection_probability_threshold,
            false_alarm_probability_threshold,
        )
    )
    numerator: float = (
        eirp_w(radar.transmitter, radar.mode)
        * target.radar_cross_section_m2
        * effective_receive_aperture_m2(radar.receiver, radar.mode)
        * pulse_compression_gain(radar.mode)
    )
    denominator: float = (
        (4 * math.pi) ** 2 * noise_power(radar) * required_snr_linear
    )
    return (numerator / denominator) ** 0.25

"""In-memory waveform diagnostics; this module never writes IQ or plots."""

import math
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .models import BurstWaveform
from .ofdm import SAMPLE_RATE
from .quantization import normalize_iq

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _spectrum_frames(
    iq: ArrayLike, sample_rate: float, nfft: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    samples = np.asarray(iq, dtype=np.complex128)
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("Spectrum requires a nonempty, finite IQ vector")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be positive and finite")
    if isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer)) or nfft < 2:
        raise ValueError("FFT length must be an integer of at least two")
    if samples.size < nfft:
        samples = np.pad(samples, (0, nfft - samples.size))
    hop = nfft // 2
    frames = np.lib.stride_tricks.sliding_window_view(samples, nfft)[::hop]
    window = np.hanning(nfft + 1)[:-1]
    spectrum = np.fft.fftshift(np.fft.fft(frames * window, axis=-1), axes=-1)
    powers = np.abs(spectrum) ** 2 / (sample_rate * np.sum(window ** 2))
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / sample_rate))
    times = (np.arange(frames.shape[0]) * hop + nfft / 2) / sample_rate
    return frequencies, times, powers


def estimate_psd(
    iq: ArrayLike, sample_rate: float = SAMPLE_RATE, nfft: int = 1024
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two-sided Welch PSD, centered in ascending frequency, with 50% overlap."""
    frequencies, _, powers = _spectrum_frames(iq, sample_rate, nfft)
    return frequencies, np.mean(powers, axis=0)


def plot_waveform(burst: BurstWaveform, digital_amplitude: float = 0.6) -> "Figure":
    """Show packet envelopes, two-sided PSD and spectrogram, without saving."""
    import matplotlib.pyplot as plt

    iq = normalize_iq(burst.iq, digital_amplitude)
    if iq.size == 0:
        raise ValueError("Cannot plot an empty burst")
    figure, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    time_ms = np.arange(iq.size) / burst.sample_rate * 1000
    axes[0].plot(time_ms, np.abs(iq), linewidth=0.6)
    axes[0].set(title="Packet-like OFDM envelope", xlabel="Time (ms)", ylabel="IQ magnitude")
    axes[0].set_xlim(0, burst.duration_ms)
    axes[0].grid(alpha=0.25)

    frequencies, psd = estimate_psd(iq, burst.sample_rate)
    floor = np.finfo(float).tiny
    axes[1].plot(frequencies / 1e6, 10 * np.log10(np.maximum(psd, floor)), linewidth=0.8)
    axes[1].set(
        title="Two-sided baseband power spectral density",
        xlabel="Frequency relative to carrier (MHz)",
        ylabel="PSD (dB/Hz)",
        xlim=(-10, 10),
    )
    axes[1].grid(alpha=0.25)

    frequencies, times, powers = _spectrum_frames(iq, burst.sample_rate, 128)
    db = 10 * np.log10(np.maximum(powers, floor))
    highest = float(np.max(db))
    mesh = axes[2].pcolormesh(
        times * 1000,
        frequencies / 1e6,
        db.T,
        shading="auto",
        vmin=highest - 70,
        vmax=highest,
        cmap="magma",
    )
    axes[2].set(
        title="Packet activity and sample-timed gaps",
        xlabel="Time (ms)",
        ylabel="Frequency relative to carrier (MHz)",
        ylim=(-10, 10),
        xlim=(0, burst.duration_ms),
    )
    figure.colorbar(mesh, ax=axes[2], label="PSD (dB/Hz)")
    return figure

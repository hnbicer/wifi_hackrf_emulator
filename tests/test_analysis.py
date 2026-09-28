import numpy as np
import pytest

from wifi_emulator.analysis import estimate_psd, plot_waveform
from wifi_emulator.burst import generate_burst
from wifi_emulator.models import TrafficEvent
from wifi_emulator.ofdm import SAMPLE_RATE, generate_ofdm_symbols


@pytest.mark.parametrize("frequency_hz", [-3_125_000, 3_125_000])
def test_two_sided_psd_preserves_frequency_sign(frequency_hz):
    tone = np.exp(2j * np.pi * frequency_hz * np.arange(8192) / SAMPLE_RATE)
    frequencies, power = estimate_psd(tone)
    assert np.all(np.diff(frequencies) > 0)
    assert frequencies[0] == -10_000_000
    assert frequencies[np.argmax(power)] == frequency_hz
    assert np.sum(power) * (frequencies[1] - frequencies[0]) == pytest.approx(1.0)


def test_ofdm_energy_is_broad_and_within_expected_band():
    iq = generate_ofdm_symbols(1000, np.random.default_rng(3))
    frequencies, power = estimate_psd(iq)
    total = np.sum(power)
    assert np.sum(power[np.abs(frequencies) <= 8_700_000]) / total > 0.95
    assert np.sum(power[frequencies < 0]) / total == pytest.approx(0.5, abs=0.06)
    interior = power[(np.abs(frequencies) > 500_000) & (np.abs(frequencies) < 7_500_000)]
    guards = power[np.abs(frequencies) > 9_000_000]
    assert np.mean(interior) > 10 * np.mean(guards)


def test_plot_waveform_has_time_psd_and_spectrogram_without_writing(tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.chdir(tmp_path)
    burst = generate_burst(TrafficEvent(1, 6, 2_437_000_000, [100, 200], [50], waveform_seed=4))
    figure = plot_waveform(burst)
    try:
        figure.canvas.draw()
        assert len(figure.axes) == 4  # Three plots plus the spectrogram color bar.
        assert figure.axes[1].get_xlim() == (-10, 10)
        assert figure.axes[2].get_ylim() == (-10, 10)
        assert list(tmp_path.iterdir()) == []
    finally:
        plt.close(figure)

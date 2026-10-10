"""Unit tests for CommSimLab modules.

Run with: uv run pytest tests/ -v
"""

import numpy as np
import pytest

# ── Channel ──────────────────────────────────────────────────────────────────

class TestChannel:
    def test_awgn_channel_shape(self):
        from commlab.channel import awgn_channel
        sig = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        noisy, noise = awgn_channel(sig, 20)
        assert noisy.shape == sig.shape
        assert noise.shape == sig.shape

    def test_snr_db(self):
        from commlab.channel import awgn_channel, snr_db
        sig = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 10000))
        noisy, noise = awgn_channel(sig, 20)
        measured = snr_db(sig, noise)
        assert abs(measured - 20) < 1.0  # within 1 dB

    def test_noise_figure_db(self):
        from commlab.channel import noise_figure_db
        nf = noise_figure_db(30, 27)
        assert abs(nf - 3.0) < 0.01

    def test_cascaded_noise_figure(self):
        from commlab.channel import cascaded_noise_figure
        # Single stage should return that stage's NF
        nf = cascaded_noise_figure([10], [3])
        assert abs(nf - 3.0) < 0.01

    def test_noise_temperature(self):
        from commlab.channel import noise_temperature
        # NF = 0 dB → T_eq = 0 K
        t_eq = noise_temperature(0)
        assert abs(t_eq) < 0.01


# ── Capacity ─────────────────────────────────────────────────────────────────

class TestCapacity:
    def test_shannon_capacity(self):
        from commlab.capacity import shannon_capacity_db
        # B=3000 Hz, SNR=20 dB → C ≈ 19973 bps
        cap = shannon_capacity_db(3000, 20)
        assert 19000 < cap < 21000

    def test_nyquist_max_rate(self):
        from commlab.capacity import nyquist_max_rate
        # B=3000 Hz, M=4 → R = 2*3000*log2(4) = 12000 bps
        rate = nyquist_max_rate(3000, 4)
        assert abs(rate - 12000) < 0.01

    def test_min_nyquist_levels(self):
        from commlab.capacity import min_nyquist_levels
        # B=3000, R=12000 → M=4
        m = min_nyquist_levels(3000, 12000)
        assert m == 4

    def test_bandwidth_for_capacity(self):
        from commlab.capacity import bandwidth_for_capacity
        # C=19973, SNR=20 dB → B ≈ 3000
        bw = bandwidth_for_capacity(19973, 20)
        assert abs(bw - 3000) < 10


# ── Analog ───────────────────────────────────────────────────────────────────

class TestAnalog:
    def test_am_modulate_shape(self):
        from commlab.analog import generate_tone, am_modulate
        t, msg = generate_tone(20, 0.1, 8000)
        t2, am, car, env = am_modulate(msg, 500, 8000, 0.5)
        assert am.shape == msg.shape

    def test_am_power_formula(self):
        from commlab.analog import am_power
        # Pt = Pc(1 + m²/2) = 1*(1 + 0.25/2) = 1.125
        p = am_power(1.0, 0.5)
        assert abs(p - 1.125) < 1e-6

    def test_am_efficiency(self):
        from commlab.analog import am_efficiency
        # η = m²/(2+m²) = 0.25/2.25 ≈ 0.1111
        eff = am_efficiency(0.5)
        assert abs(eff - 0.25 / 2.25) < 1e-6

    def test_overmodulation(self):
        from commlab.analog import check_overmodulation
        assert check_overmodulation(1.5) is True
        assert check_overmodulation(0.5) is False

    def test_fm_modulation_index(self):
        from commlab.analog import fm_modulation_index
        beta = fm_modulation_index(75000, 15000)
        assert abs(beta - 5.0) < 0.01

    def test_fm_bandwidth_carson(self):
        from commlab.analog import fm_bandwidth_carson
        # BW = 2(Δf + fm) = 2(75000+15000) = 180000
        bw = fm_bandwidth_carson(75000, 15000)
        assert abs(bw - 180000) < 0.01

    def test_generate_tone(self):
        from commlab.analog import generate_tone
        t, sig = generate_tone(100, 0.5, 10000)
        assert len(t) == len(sig)
        assert len(t) == 5000


# ── Sampling ─────────────────────────────────────────────────────────────────

class TestSampling:
    def test_nyquist_rate(self):
        from commlab.sampling import nyquist_rate
        assert nyquist_rate(4000) == 8000

    def test_check_aliasing_true(self):
        from commlab.sampling import check_aliasing
        aliased, _ = check_aliasing(500, 800)
        assert aliased is True

    def test_check_aliasing_false(self):
        from commlab.sampling import check_aliasing
        aliased, _ = check_aliasing(500, 1200)
        assert aliased is False

    def test_sample_signal(self):
        from commlab.sampling import sample_signal
        t = np.linspace(0, 1, 10000)
        sig = np.sin(2 * np.pi * 10 * t)
        s_t, s_vals, s_idx = sample_signal(sig, t, 10000, 100)
        assert len(s_vals) > 0
        assert len(s_t) == len(s_vals)


# ── Quantization ─────────────────────────────────────────────────────────────

class TestQuantization:
    def test_sqnr_formula(self):
        from commlab.quantization import sqnr_db
        # SQNR = 6.02n + 1.76
        assert abs(sqnr_db(8) - 49.92) < 0.01
        assert abs(sqnr_db(16) - 98.08) < 0.01

    def test_uniform_quantize(self):
        from commlab.quantization import uniform_quantize
        sig = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        q, levels, step = uniform_quantize(sig, 2)
        assert len(q) == len(sig)
        assert step > 0

    def test_quantization_noise_power(self):
        from commlab.quantization import quantization_noise_power
        # Δ²/12
        noise = quantization_noise_power(1.0)
        assert abs(noise - 1.0 / 12) < 1e-6


# ── PCM ──────────────────────────────────────────────────────────────────────

class TestPCM:
    def test_pcm_bitrate(self):
        from commlab.pcm import pcm_bitrate
        assert pcm_bitrate(8000, 8) == 64000

    def test_pcm_bandwidth(self):
        from commlab.pcm import pcm_bandwidth
        assert pcm_bandwidth(8000, 8) == 32000

    def test_pcm_encode_decode(self):
        from commlab.pcm import pcm_encode, pcm_decode
        t = np.linspace(0, 0.1, 1000)
        sig = np.sin(2 * np.pi * 50 * t)
        result = pcm_encode(sig, t, 500, 8)
        decoded = pcm_decode(result['codes'], result['step_size'],
                             result['v_min'], 8)
        assert len(decoded) == len(result['samples'])
        assert np.allclose(decoded, result['quantized'])


# ── Delta Modulation ─────────────────────────────────────────────────────────

class TestDelta:
    def test_dm_encode_decode(self):
        from commlab.delta import dm_encode, dm_decode
        sig = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        dm_out, stair, err = dm_encode(sig, 0.1)
        decoded = dm_decode(dm_out, 0.1)
        assert len(decoded) == len(dm_out)

    def test_adaptive_dm(self):
        from commlab.delta import adaptive_dm_encode, adaptive_dm_decode
        sig = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        adm_out, stair, steps = adaptive_dm_encode(sig, 0.1)
        decoded = adaptive_dm_decode(adm_out, 0.1)
        assert len(decoded) == len(adm_out)

    def test_slope_overload_detection(self):
        from commlab.delta import dm_encode, detect_slope_overload
        # High-freq signal with small step → slope overload expected
        sig = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
        _, stair, _ = dm_encode(sig, 0.01)
        overload = detect_slope_overload(sig, stair)
        assert overload.any()  # some slope overload expected


# ── Line Coding ──────────────────────────────────────────────────────────────

class TestLineCoding:
    def test_nrz_unipolar(self):
        from commlab.linecode import nrz_unipolar
        bits = np.array([1, 0, 1, 1, 0])
        t, sig = nrz_unipolar(bits, 1000, 50000)
        assert len(t) == len(sig)
        assert np.max(sig) <= 1.0 + 1e-9
        assert np.min(sig) >= -1e-9

    def test_nrz_polar(self):
        from commlab.linecode import nrz_polar
        bits = np.array([1, 0, 1, 1, 0])
        t, sig = nrz_polar(bits, 1000, 50000)
        # Polar: values should be ±1
        unique_vals = np.unique(sig)
        assert len(unique_vals) == 2

    def test_manchester(self):
        from commlab.linecode import manchester
        bits = np.array([1, 0, 1])
        t, sig = manchester(bits, 1000, 50000)
        assert len(sig) > 0
        # Manchester always has transitions

    def test_random_bits(self):
        from commlab.linecode import random_bits
        bits = random_bits(100)
        assert len(bits) == 100
        assert set(bits).issubset({0, 1})


# ── Radio ────────────────────────────────────────────────────────────────────

class TestRadio:
    def test_superheterodyne_lo(self):
        from commlab.radio import superheterodyne_lo_freq
        # High-side: fLO = fRF + fIF
        lo = superheterodyne_lo_freq(1e6, 455e3, high_side=True)
        assert abs(lo - 1.455e6) < 1

    def test_image_frequency(self):
        from commlab.radio import image_frequency
        # High-side: fimage = fRF + 2*fIF
        img = image_frequency(1e6, 455e3, high_side=True)
        assert abs(img - 1.91e6) < 1

    def test_agc_gain(self):
        from commlab.radio import agc_gain
        gain = agc_gain(-10, -40)
        assert abs(gain - 30) < 0.01


# ── TV ───────────────────────────────────────────────────────────────────────

class TestTV:
    def test_raster_scan_params(self):
        from commlab.tv import raster_scan_params
        params = raster_scan_params(525, 30, True)
        assert 'h_freq' in params

    def test_vsb_tv_bandwidth(self):
        from commlab.tv import vsb_tv_bandwidth
        bw = vsb_tv_bandwidth()
        assert isinstance(bw, dict)

    def test_generate_raster_pattern(self):
        from commlab.tv import generate_raster_pattern
        pat = generate_raster_pattern(320, 240, 'bars')
        assert pat.shape[0] == 240
        assert pat.shape[1] == 320

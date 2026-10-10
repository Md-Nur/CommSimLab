"""CommSimLab – Streamlit Dashboard.

Interactive multi-page dashboard for Communication Systems simulation.
Each page corresponds to a theory module: channel/noise, analog modulation,
radio receivers, sampling, PCM, delta modulation, and line coding.

Run with:  uv run streamlit run app.py
"""

import streamlit as st
import numpy as np

st.set_page_config(
    page_title="CommSimLab",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
st.sidebar.title("📡 CommSimLab")
st.sidebar.markdown("*Communication Systems Simulation Lab*")

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "📶 Channel & Noise",
        "📊 Shannon Capacity",
        "🔊 Analog Modulation",
        "📻 Radio Receivers",
        "⏱️ Sampling",
        "🔢 Quantization",
        "💾 PCM",
        "📐 Delta Modulation",
        "🔲 Line Coding",
        "📺 TV Systems",
    ],
)


# ── Helper to show matplotlib figures in Streamlit ───────────────────────────
def show_fig(fig):
    """Display a matplotlib figure and close it to free memory."""
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📡 CommSimLab")
    st.markdown(
        """
        Welcome to **CommSimLab** – your interactive Communication Systems
        Simulation Laboratory!

        Use the sidebar to explore different modules:

        | Module | Topics |
        |--------|--------|
        | **Channel & Noise** | AWGN, SNR, noise figure, cascaded noise (Friis) |
        | **Shannon Capacity** | Channel capacity, Nyquist rate, data-rate calculators |
        | **Analog Modulation** | AM (mod index, multi-tone, power), FM, VSB, FM demod |
        | **Radio Receivers** | Pre/de-emphasis, superheterodyne, AGC |
        | **Sampling** | Sampling theorem, aliasing, natural vs flat-top |
        | **Quantization** | Uniform quantizer, SQNR, PAM |
        | **PCM** | Full PCM chain: sample → quantize → encode → reconstruct |
        | **Delta Modulation** | DPCM, DM, adaptive DM, slope overload vs granular noise |
        | **Line Coding** | NRZ/RZ/Manchester/AMI + PSD plots |
        | **TV Systems** | Raster scan, VSB spectrum |

        Each page lets you adjust parameters with sliders and see results
        update live.
        """
    )

# ═══════════════════════════════════════════════════════════════════════════════
# CHANNEL & NOISE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📶 Channel & Noise":
    from commlab.channel import (
        awgn_channel, snr_db, noise_figure_db,
        cascaded_noise_figure, noise_temperature, plot_signal_with_noise,
    )
    from commlab.analog import generate_tone

    st.title("📶 Channel & Noise")

    tab1, tab2 = st.tabs(["AWGN Channel", "Cascaded Noise Figure"])

    with tab1:
        st.subheader("Add AWGN to a Signal")
        col1, col2 = st.columns(2)
        with col1:
            freq = st.slider("Signal frequency (Hz)", 1, 100, 10, key="ch_freq")
            snr_val = st.slider("SNR (dB)", -10, 40, 20, key="ch_snr")
        with col2:
            duration = st.slider("Duration (s)", 0.1, 2.0, 0.5, step=0.1, key="ch_dur")
            fs = st.slider("Sample rate (Hz)", 500, 10000, 2000, key="ch_fs")

        t, signal = generate_tone(freq, duration, fs)
        noisy, noise = awgn_channel(signal, snr_val)
        measured_snr = snr_db(signal, noise)

        st.metric("Measured SNR (dB)", f"{measured_snr:.2f}")
        fig = plot_signal_with_noise(t, signal, noisy, title=f"Signal + AWGN (SNR = {snr_val} dB)")
        show_fig(fig)

    with tab2:
        st.subheader("Friis Formula – Cascaded Amplifiers")
        num_stages = st.slider("Number of stages", 2, 6, 3, key="ch_stages")
        gains = []
        nfs = []
        cols = st.columns(num_stages)
        for i, c in enumerate(cols):
            with c:
                g = st.number_input(f"Gain {i+1} (dB)", value=10.0, step=1.0, key=f"g_{i}")
                n = st.number_input(f"NF {i+1} (dB)", value=3.0, step=0.5, key=f"nf_{i}")
                gains.append(g)
                nfs.append(n)

        overall_nf = cascaded_noise_figure(gains, nfs)
        t_eq = noise_temperature(overall_nf)
        st.metric("Overall Noise Figure (dB)", f"{overall_nf:.3f}")
        st.metric("Equivalent Noise Temperature (K)", f"{t_eq:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SHANNON CAPACITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Shannon Capacity":
    from commlab.capacity import (
        shannon_capacity_db, nyquist_max_rate, min_nyquist_levels,
        bandwidth_for_capacity, plot_capacity_vs_snr, plot_capacity_vs_bandwidth,
    )

    st.title("📊 Shannon Capacity & Nyquist")

    tab1, tab2, tab3 = st.tabs(["Shannon Capacity", "Nyquist Calculator", "Plots"])

    with tab1:
        st.subheader("Shannon-Hartley Theorem")
        bw = st.number_input("Bandwidth (Hz)", value=3000.0, step=100.0, key="sc_bw")
        snr_input = st.slider("SNR (dB)", -5, 50, 20, key="sc_snr")
        cap = shannon_capacity_db(bw, snr_input)
        st.metric("Channel Capacity (bps)", f"{cap:,.1f}")
        st.latex(r"C = B \cdot \log_2(1 + \text{SNR})")

    with tab2:
        st.subheader("Nyquist Maximum Data Rate")
        bw2 = st.number_input("Bandwidth (Hz)", value=3000.0, step=100.0, key="ny_bw")
        levels = st.number_input("Signal levels (M)", value=4, min_value=2, step=1, key="ny_lev")
        rate = nyquist_max_rate(bw2, levels)
        st.metric("Max Data Rate (bps)", f"{rate:,.1f}")
        st.latex(r"R_{\max} = 2B \cdot \log_2(M)")

        st.divider()
        st.subheader("Reverse: Minimum Levels")
        bw3 = st.number_input("Bandwidth (Hz)", value=3000.0, step=100.0, key="ny_bw3")
        target_rate = st.number_input("Target data rate (bps)", value=24000.0, step=1000.0)
        min_m = min_nyquist_levels(bw3, target_rate)
        st.metric("Minimum levels M", str(min_m))

    with tab3:
        st.subheader("Capacity vs SNR")
        bw_plot = st.number_input("Bandwidth (Hz)", value=3000.0, step=100.0, key="sc_bw_plot")
        snr_range = np.arange(-5, 51, 1)
        fig = plot_capacity_vs_snr(bw_plot, snr_range)
        show_fig(fig)

        st.subheader("Capacity vs Bandwidth")
        snr_plot = st.slider("SNR (dB)", -5, 50, 20, key="sc_snr_plot")
        bw_range = np.arange(100, 10001, 100)
        fig2 = plot_capacity_vs_bandwidth(snr_plot, bw_range)
        show_fig(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALOG MODULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔊 Analog Modulation":
    from commlab.analog import (
        generate_tone, am_modulate, am_power, am_efficiency,
        am_demodulate_envelope, check_overmodulation,
        fm_modulate, fm_bandwidth_carson, fm_modulation_index,
        fm_demodulate_discriminator,
        plot_am, plot_fm, plot_spectrum,
    )

    st.title("🔊 Analog Modulation")

    tab1, tab2 = st.tabs(["AM Modulation", "FM Modulation"])

    with tab1:
        st.subheader("Amplitude Modulation")
        col1, col2 = st.columns(2)
        with col1:
            fm_freq = st.slider("Message freq (Hz)", 1, 200, 20, key="am_fm")
            fc_freq = st.slider("Carrier freq (Hz)", 100, 2000, 500, key="am_fc")
            mod_idx = st.slider("Modulation index (m)", 0.0, 2.0, 0.5, step=0.05, key="am_m")
        with col2:
            duration = st.slider("Duration (s)", 0.05, 1.0, 0.2, step=0.05, key="am_dur")
            fs = st.slider("Sample rate (Hz)", 2000, 20000, 8000, key="am_fs")

        t, message = generate_tone(fm_freq, duration, fs)
        t, am_sig, carrier, envelope = am_modulate(message, fc_freq, fs, mod_idx)

        is_over = check_overmodulation(mod_idx)
        if is_over:
            st.warning("⚠️ Over-modulation detected (m > 1.0)!")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Power / Pc", f"{1 + mod_idx**2/2:.3f}")
        col_m2.metric("Efficiency", f"{am_efficiency(mod_idx)*100:.1f}%")
        col_m3.metric("Over-modulated", "Yes" if is_over else "No")

        fig = plot_am(t, message, carrier, am_sig, envelope)
        show_fig(fig)

        st.subheader("AM Spectrum")
        fig_spec = plot_spectrum(am_sig, fs, title="AM Signal Spectrum")
        show_fig(fig_spec)

        st.subheader("Envelope Detection (Demodulation)")
        demod = am_demodulate_envelope(am_sig, fs, fm_freq * 3)
        import matplotlib.pyplot as plt
        fig_d, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, message / np.max(np.abs(message)), label="Original (normalized)", alpha=0.7)
        ax.plot(t, demod / np.max(np.abs(demod)), label="Demodulated (normalized)", alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.set_title("Envelope Detection Result")
        fig_d.tight_layout()
        show_fig(fig_d)

    with tab2:
        st.subheader("Frequency Modulation")
        col1, col2 = st.columns(2)
        with col1:
            fm_msg_freq = st.slider("Message freq (Hz)", 1, 200, 20, key="fm_fm")
            fm_car_freq = st.slider("Carrier freq (Hz)", 100, 5000, 1000, key="fm_fc")
            fm_dev = st.slider("Freq deviation (Hz)", 10, 2000, 200, key="fm_dev")
        with col2:
            fm_dur = st.slider("Duration (s)", 0.05, 1.0, 0.2, step=0.05, key="fm_dur")
            fm_fs = st.slider("Sample rate (Hz)", 5000, 50000, 20000, key="fm_fs")

        t_fm, msg_fm = generate_tone(fm_msg_freq, fm_dur, fm_fs)
        t_fm, fm_sig = fm_modulate(msg_fm, fm_car_freq, fm_fs, fm_dev)

        beta = fm_modulation_index(fm_dev, fm_msg_freq)
        bw_carson = fm_bandwidth_carson(fm_dev, fm_msg_freq)

        col_f1, col_f2 = st.columns(2)
        col_f1.metric("Modulation Index (β)", f"{beta:.2f}")
        col_f2.metric("Carson's BW (Hz)", f"{bw_carson:,.0f}")

        fig_fm = plot_fm(t_fm, msg_fm, fm_sig)
        show_fig(fig_fm)

        st.subheader("FM Spectrum")
        fig_fms = plot_spectrum(fm_sig, fm_fs, title="FM Signal Spectrum")
        show_fig(fig_fms)

# ═══════════════════════════════════════════════════════════════════════════════
# RADIO RECEIVERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📻 Radio Receivers":
    from commlab.radio import (
        pre_emphasis, de_emphasis, superheterodyne_lo_freq,
        superheterodyne_tuning_range, image_frequency, agc_gain,
        plot_pre_de_emphasis,
    )
    from commlab.analog import generate_tone

    st.title("📻 Radio Receiver Concepts")

    tab1, tab2, tab3 = st.tabs(["Superheterodyne", "Pre/De-emphasis", "AGC"])

    with tab1:
        st.subheader("Superheterodyne Tuning Range")
        col1, col2 = st.columns(2)
        with col1:
            rf_min = st.number_input("RF min (MHz)", value=540.0, step=10.0)
            rf_max = st.number_input("RF max (MHz)", value=1600.0, step=10.0)
        with col2:
            if_freq = st.number_input("IF (kHz)", value=455.0, step=5.0)
            high_side = st.checkbox("High-side injection", value=True)

        result = superheterodyne_tuning_range(
            rf_min * 1e6, rf_max * 1e6, if_freq * 1e3, high_side
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("LO min (MHz)", f"{result['lo_min']/1e6:.3f}")
        col_r2.metric("LO max (MHz)", f"{result['lo_max']/1e6:.3f}")
        col_r3.metric("Capacitor ratio", f"{result['capacitor_ratio']:.3f}")

        st.divider()
        st.subheader("Image Frequency")
        rf_test = st.number_input("RF freq (MHz)", value=1000.0, step=10.0)
        img = image_frequency(rf_test * 1e6, if_freq * 1e3, high_side)
        st.metric("Image Frequency (MHz)", f"{img/1e6:.3f}")

    with tab2:
        st.subheader("Pre-emphasis & De-emphasis Filter Response")
        tau = st.select_slider("Time constant τ (μs)", options=[25, 50, 75, 100, 150], value=75)
        fm_fs = st.slider("Sample rate (Hz)", 10000, 100000, 44100, key="radio_fs")
        fig = plot_pre_de_emphasis(fm_fs, tau * 1e-6)
        show_fig(fig)

        st.divider()
        st.subheader("Effect on a Signal")
        freq_sig = st.slider("Signal freq (Hz)", 100, 15000, 3000, key="radio_sig_freq")
        t, sig = generate_tone(freq_sig, 0.01, fm_fs)
        pre = pre_emphasis(sig, fm_fs, tau * 1e-6)
        de = de_emphasis(pre, fm_fs, tau * 1e-6)

        import matplotlib.pyplot as plt
        fig2, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t[:200], sig[:200])
        axes[0].set_title("Original")
        axes[1].plot(t[:200], pre[:200])
        axes[1].set_title("Pre-emphasized")
        axes[2].plot(t[:200], de[:200])
        axes[2].set_title("De-emphasized (recovered)")
        fig2.tight_layout()
        show_fig(fig2)

    with tab3:
        st.subheader("AGC Gain Calculator")
        target = st.number_input("Target level (dBm)", value=-10.0, step=1.0)
        input_lev = st.slider("Input level (dBm)", -80, 0, -40)
        gain = agc_gain(target, input_lev)
        st.metric("Required AGC Gain (dB)", f"{gain:.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Sampling":
    from commlab.sampling import (
        nyquist_rate, sample_signal, natural_sampling, flat_top_sampling,
        reconstruct_signal, check_aliasing, plot_sampling_demo,
    )
    from commlab.analog import generate_tone

    st.title("⏱️ Sampling Theorem")

    tab1, tab2 = st.tabs(["Sampling & Reconstruction", "Aliasing Demo"])

    with tab1:
        st.subheader("Sample, Reconstruct & Compare")
        col1, col2 = st.columns(2)
        with col1:
            sig_freq = st.slider("Signal freq (Hz)", 1, 500, 50, key="samp_freq")
            fs_orig = st.slider("Simulation rate (Hz)", 2000, 20000, 10000, key="samp_fsorig")
        with col2:
            fs_samp = st.slider("Sampling rate (Hz)", 10, 2000, 200, key="samp_fssamp")
            duration = st.slider("Duration (s)", 0.02, 0.5, 0.1, step=0.01, key="samp_dur")

        nyq = nyquist_rate(sig_freq)
        is_aliased, alias_freq = check_aliasing(sig_freq, fs_samp)

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Nyquist Rate (Hz)", f"{nyq}")
        col_m2.metric("Aliasing?", f"Yes (→ {alias_freq:.1f} Hz)" if is_aliased else "No")

        if is_aliased:
            st.warning(f"⚠️ Sampling rate {fs_samp} Hz < Nyquist rate {nyq} Hz — aliasing occurs!")

        t, signal = generate_tone(sig_freq, duration, fs_orig)
        s_times, samples, s_idx = sample_signal(signal, t, fs_orig, fs_samp)
        t_recon = t
        reconstructed = reconstruct_signal(samples, s_times, t_recon, fs_samp)

        fig = plot_sampling_demo(t, signal, (s_times, samples), reconstructed, title=f"Sampling at {fs_samp} Hz")
        show_fig(fig)

    with tab2:
        st.subheader("Aliasing Demonstration")
        sig_freq2 = st.slider("Signal freq (Hz)", 10, 500, 100, key="alias_freq")
        fs_list_str = st.text_input(
            "Sampling rates (Hz, comma-separated)",
            value="80, 150, 200, 500",
        )
        try:
            fs_list = [float(x.strip()) for x in fs_list_str.split(",")]
        except ValueError:
            fs_list = [80, 150, 200, 500]

        for fs_test in fs_list:
            aliased, af = check_aliasing(sig_freq2, fs_test)
            status = f"⚠️ Aliased → {af:.1f} Hz" if aliased else "✅ OK"
            st.write(f"**fs = {fs_test} Hz**: {status}")

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔢 Quantization":
    from commlab.quantization import (
        uniform_quantize, quantization_noise_power, sqnr_db, sqnr_actual,
        plot_quantization, plot_sqnr_vs_bits,
    )
    from commlab.analog import generate_tone

    st.title("🔢 Quantization")

    tab1, tab2 = st.tabs(["Uniform Quantizer", "SQNR Analysis"])

    with tab1:
        st.subheader("Uniform Mid-Rise Quantizer")
        col1, col2 = st.columns(2)
        with col1:
            freq = st.slider("Signal freq (Hz)", 1, 200, 20, key="q_freq")
            num_bits = st.slider("Number of bits", 1, 12, 4, key="q_bits")
        with col2:
            duration = st.slider("Duration (s)", 0.05, 1.0, 0.2, step=0.05, key="q_dur")
            fs = st.slider("Sample rate (Hz)", 1000, 20000, 8000, key="q_fs")

        t, signal = generate_tone(freq, duration, fs)
        quantized, levels, step = uniform_quantize(signal, num_bits)

        noise_pow = quantization_noise_power(step)
        actual_sqnr = sqnr_actual(signal, quantized)
        theoretical_sqnr = sqnr_db(num_bits)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Step size (Δ)", f"{step:.4f}")
        col_m2.metric("Actual SQNR (dB)", f"{actual_sqnr:.2f}")
        col_m3.metric("Theoretical SQNR (dB)", f"{theoretical_sqnr:.2f}")

        fig = plot_quantization(t, signal, quantized, levels, title=f"{num_bits}-bit Quantization")
        show_fig(fig)

    with tab2:
        st.subheader("SQNR vs Number of Bits")
        max_b = st.slider("Max bits", 4, 24, 16, key="q_maxb")
        fig2 = plot_sqnr_vs_bits(max_b)
        show_fig(fig2)

        st.latex(r"\text{SQNR} \approx 6.02n + 1.76 \text{ dB}")
        st.info("Each additional bit improves SQNR by ~6 dB.")

# ═══════════════════════════════════════════════════════════════════════════════
# PCM
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💾 PCM":
    from commlab.pcm import (
        pcm_encode, pcm_decode, pcm_reconstruct, pcm_bitrate,
        pcm_bandwidth, plot_pcm_chain,
    )
    from commlab.analog import generate_tone

    st.title("💾 Pulse Code Modulation")

    col1, col2 = st.columns(2)
    with col1:
        freq = st.slider("Signal freq (Hz)", 1, 200, 20, key="pcm_freq")
        num_bits = st.slider("Number of bits", 2, 12, 4, key="pcm_bits")
        fs_sample = st.slider("Sampling rate (Hz)", 50, 2000, 200, key="pcm_fss")
    with col2:
        duration = st.slider("Duration (s)", 0.05, 1.0, 0.2, step=0.05, key="pcm_dur")
        fs_orig = st.slider("Simulation rate (Hz)", 4000, 20000, 10000, key="pcm_fsorig")

    t, signal = generate_tone(freq, duration, fs_orig)
    result = pcm_encode(signal, t, fs_sample, num_bits)

    bitrate = pcm_bitrate(fs_sample, num_bits)
    min_bw = pcm_bandwidth(fs_sample, num_bits)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Bit Rate (bps)", f"{bitrate:,}")
    col_m2.metric("Min Bandwidth (Hz)", f"{min_bw:,}")
    col_m3.metric("Quantization Levels", f"{2**num_bits}")

    decoded = pcm_decode(result['codes'], result['step_size'], result['v_min'], num_bits)
    reconstructed = pcm_reconstruct(decoded, result['sample_times'], t, fs_sample)

    fig = plot_pcm_chain(
        t, signal, result['sample_times'], result['samples'],
        result['quantized'], reconstructed, title=f"{num_bits}-bit PCM"
    )
    show_fig(fig)

    with st.expander("Binary Codes (first 20 samples)"):
        for i, code in enumerate(result['codes'][:20]):
            st.code(f"Sample {i}: {code}", language=None)

# ═══════════════════════════════════════════════════════════════════════════════
# DELTA MODULATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Delta Modulation":
    from commlab.delta import (
        dm_encode, dm_decode, adaptive_dm_encode, adaptive_dm_decode,
        detect_slope_overload, detect_granular_noise,
        plot_dm_analysis, plot_adm_comparison,
    )
    from commlab.analog import generate_tone

    st.title("📐 Delta Modulation")

    tab1, tab2 = st.tabs(["DM vs ADM", "Analysis"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            freq = st.slider("Signal freq (Hz)", 1, 100, 10, key="dm_freq")
            step_size = st.slider("Step size (Δ)", 0.01, 1.0, 0.1, step=0.01, key="dm_step")
        with col2:
            duration = st.slider("Duration (s)", 0.1, 2.0, 0.5, step=0.1, key="dm_dur")
            fs = st.slider("Sample rate (Hz)", 500, 10000, 2000, key="dm_fs")

        t, signal = generate_tone(freq, duration, fs)
        dm_out, dm_stair, dm_err = dm_encode(signal, step_size)
        adm_out, adm_stair, adm_steps = adaptive_dm_encode(signal, step_size)

        fig = plot_adm_comparison(t, signal, dm_stair, adm_stair, title="DM vs Adaptive DM")
        show_fig(fig)

    with tab2:
        st.subheader("Slope Overload & Granular Noise Detection")
        col1, col2 = st.columns(2)
        with col1:
            freq2 = st.slider("Signal freq (Hz)", 1, 100, 10, key="dm_freq2")
            step2 = st.slider("Step size (Δ)", 0.01, 1.0, 0.1, step=0.01, key="dm_step2")
        with col2:
            dur2 = st.slider("Duration (s)", 0.1, 2.0, 0.5, step=0.1, key="dm_dur2")
            fs2 = st.slider("Sample rate (Hz)", 500, 10000, 2000, key="dm_fs2")

        t2, sig2 = generate_tone(freq2, dur2, fs2)
        dm_out2, dm_stair2, _ = dm_encode(sig2, step2)
        slope_ol = detect_slope_overload(sig2, dm_stair2)
        granular = detect_granular_noise(sig2, dm_stair2)

        fig2 = plot_dm_analysis(t2, sig2, dm_stair2, slope_ol, granular,
                                title="DM Analysis: Slope Overload & Granular Noise")
        show_fig(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# LINE CODING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔲 Line Coding":
    from commlab.linecode import (
        nrz_unipolar, nrz_polar, nrz_inverted,
        rz_unipolar, rz_polar, rz_ami,
        manchester, diff_manchester,
        random_bits, plot_all_line_codes, plot_all_psd,
    )

    st.title("🔲 Line Coding Schemes")

    tab1, tab2 = st.tabs(["Waveforms", "Power Spectral Density"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            num_bits = st.slider("Number of bits", 4, 20, 8, key="lc_nbits")
            bit_rate = st.slider("Bit rate (bps)", 100, 10000, 1000, key="lc_br")
        with col2:
            fs = st.slider("Sample rate (Hz)", 5000, 100000, 50000, key="lc_fs")
            use_random = st.checkbox("Random bits", value=True)

        if use_random:
            bits = random_bits(num_bits)
        else:
            bits_str = st.text_input("Bit pattern", value="10110010")
            bits = np.array([int(b) for b in bits_str if b in '01'])

        st.write(f"**Bit pattern:** `{''.join(map(str, bits))}`")

        fig = plot_all_line_codes(bits, bit_rate, fs)
        show_fig(fig)

    with tab2:
        st.subheader("Power Spectral Density Comparison")
        num_bits_psd = st.slider("Number of bits for PSD", 32, 256, 128, key="lc_psd_bits")
        br_psd = st.slider("Bit rate (bps)", 100, 10000, 1000, key="lc_psd_br")
        fs_psd = st.slider("Sample rate (Hz)", 10000, 200000, 100000, key="lc_psd_fs")

        bits_psd = random_bits(num_bits_psd)
        fig2 = plot_all_psd(bits_psd, br_psd, fs_psd)
        show_fig(fig2)

# ═══════════════════════════════════════════════════════════════════════════════
# TV SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📺 TV Systems":
    from commlab.tv import (
        raster_scan_params, vsb_tv_bandwidth, generate_raster_pattern,
        plot_tv_spectrum, plot_raster_scan,
    )

    st.title("📺 TV Systems")

    tab1, tab2 = st.tabs(["Raster Scan", "VSB Spectrum"])

    with tab1:
        st.subheader("Raster Scan Parameters")
        col1, col2 = st.columns(2)
        with col1:
            lines = st.number_input("Total lines", value=525, step=1)
            fps = st.number_input("Frames per second", value=30, step=1)
        with col2:
            interlaced = st.checkbox("Interlaced", value=True)

        params = raster_scan_params(lines, fps, interlaced)
        for k, v in params.items():
            st.metric(k.replace("_", " ").title(), f"{v}")

        st.divider()
        st.subheader("Test Patterns")
        pattern = st.selectbox("Pattern", ["bars", "checkerboard", "gradient"])
        pat = generate_raster_pattern(320, 240, pattern)
        fig = plot_raster_scan(pat, title=f"Test Pattern: {pattern}")
        show_fig(fig)

    with tab2:
        st.subheader("VSB TV Channel Spectrum")
        col1, col2 = st.columns(2)
        with col1:
            vid_bw = st.number_input("Video BW (MHz)", value=4.2, step=0.1)
            vestigial = st.number_input("Vestigial BW (MHz)", value=0.75, step=0.05)
        with col2:
            audio_off = st.number_input("Audio offset (MHz)", value=4.5, step=0.1)

        bw_info = vsb_tv_bandwidth(vid_bw, vestigial, audio_off)
        for k, v in bw_info.items():
            st.metric(k.replace("_", " ").title(), f"{v}")

        fig2 = plot_tv_spectrum(vid_bw, vestigial, audio_off)
        show_fig(fig2)

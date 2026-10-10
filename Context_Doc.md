**Project: `CommSimLab`**

```
CommSimLab/
├── app.py                  # Streamlit dashboard (one page per module)
├── commlab/
│   ├── channel.py          # AWGN, SNR, noise figure, cascaded noise (Friis)
│   ├── capacity.py         # Shannon capacity, Nyquist levels, data-rate calculators
│   ├── analog.py           # AM (mod index, multi-tone, power), FM, VSB filter, FM discriminator
│   ├── radio.py            # Pre-emphasis, superheterodyne tuning range, AGC/AVC, reactance FET modulator
│   ├── sampling.py         # Sampling theorem, aliasing, natural vs flat-top sampling
│   ├── quantization.py     # Uniform quantizer, quantization noise/SQNR, PAM
│   ├── pcm.py              # PCM encoder/decoder (sampler → quantizer → encoder)
│   ├── delta.py            # DPCM, DM, adaptive DM (slope overload vs granular noise)
│   ├── linecode.py         # NRZ/RZ unipolar/polar/bipolar, Manchester + PSD plots
│   └── tv.py               # Raster scan / VSB spectrum demo for TV
├── notebooks/              # One Jupyter notebook per unit for experiments
├── tests/
└── requirements.txt        # numpy, scipy, matplotlib, streamlit
```

**What each module would show**

- **Overview/Noise:** add AWGN at a chosen SNR and watch a signal degrade; compute noise figure for cascaded amplifiers; plot Shannon capacity vs. bandwidth/SNR.
- **Analog:** time and spectrum plots for AM (adjustable modulation index, over-modulation), FM (varying deviation), VSB filtering, and FM demodulation by frequency discrimination.
- **Radio:** superheterodyne tuning-range calculator (IF, local oscillator, capacitor range), pre-emphasis/de-emphasis effect on noise.
- **Sampling/PCM:** slider for sampling rate to see aliasing appear, SQNR vs. number of bits, full PCM chain with reconstructed waveform.
- **DPCM/DM/ADM:** show slope overload and granular noise, and how ADM fixes it.
- **Line coding:** all formats side by side for a bit pattern, with their power spectra.

**Suggested build order:** `channel.py` → `analog.py` → `sampling.py` → `pcm.py` → `delta.py` → `linecode.py`, then the Streamlit app on top. This way each step is testable on its own, and the numerical problems from your syllabus become built-in calculators and unit tests.

----

## What you'll actually be doing

You're building a **software lab for my communication systems course**. Each theory topic becomes code that generates a signal, passes it through a system, and plots the result. The workflow for each module is the same:

1. **Pick a theory topic**, for example AM modulation.
2. **Write the math as code.** The AM wave expression `s(t) = Ac[1 + m·cos(2πfm·t)]·cos(2πfc·t)` becomes a NumPy line.
3. **Generate signals.** Create a time axis, a message signal, and a carrier.
4. **Pass them through the system.** Modulate, add noise, filter, sample, quantize, or demodulate.
5. **Visualize.** Plot the time-domain waveform and its spectrum (FFT).
6. **Verify.** Check your results against your textbook numericals. For example, if the modulation index is 0.5, does the simulated power match the formula `Pt = Pc(1 + m²/2)`?

You'll end up with something you can run to see aliasing appear, watch FM bandwidth grow, or see quantization noise fall as you add bits. That makes the course concepts far easier to remember for exams, and the project works as a portfolio piece.

## Concepts you need alongside the libraries

Several skills matter as much as the libraries:

- **Sampling a "continuous" signal in code.** You simulate analog signals with a very high sample rate, so you need to keep the simulation rate separate from the sampling rate you are demonstrating.
- **FFT and spectrum reading.** Frequency axis, magnitude, and windowing.
- **dB conversions** for SNR and noise figure.
- **Basic filter design**, low-pass and band-pass.

A good first exercise is to simulate an AM wave with `m = 0.5`, plot it, plot its FFT, and confirm you see the carrier plus two sidebands. If that works, the rest of the project is variations on the same pattern.

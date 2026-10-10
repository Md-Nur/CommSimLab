# CommSimLab: Communication Systems Simulation Lab 📡

An interactive software simulation laboratory for communication systems engineering. Built with NumPy, SciPy, Matplotlib, and Streamlit, CommSimLab turns textbook theory into executable code, real-time interactive dashboards, and verifiable numerical experiments.

---

## 📁 Project Structure

```
CommSimLab/
├── app.py                  # Interactive multi-page Streamlit dashboard
├── commlab/                # Core communication systems library
│   ├── channel.py          # AWGN, SNR, noise figure, cascaded noise (Friis)
│   ├── capacity.py         # Shannon capacity, Nyquist levels, data-rate calculators
│   ├── analog.py           # AM (mod index, multi-tone, power), FM, VSB filter, FM demod
│   ├── radio.py            # Pre-emphasis/de-emphasis, superheterodyne tuning, AGC
│   ├── sampling.py         # Sampling theorem, aliasing, natural vs flat-top sampling
│   ├── quantization.py     # Uniform quantizer, quantization noise/SQNR, PAM
│   ├── pcm.py              # PCM encoder/decoder (sampler → quantizer → encoder)
│   ├── delta.py            # DPCM, DM, adaptive DM (slope overload vs granular noise)
│   ├── linecode.py         # NRZ/RZ (unipolar, polar, AMI), Manchester + PSD plots
│   └── tv.py               # Raster scan parameters & VSB spectrum demo for TV
├── notebooks/              # Unit experiment Jupyter notebooks
│   ├── 01_channel_and_capacity.ipynb
│   ├── 02_analog_modulation.ipynb
│   ├── 03_sampling_and_pcm.ipynb
│   └── 04_delta_modulation_and_linecoding.ipynb
├── tests/                  # Unit test suite with textbook formula verification
│   └── test_modules.py
├── pyproject.toml          # Project configuration & dependencies
├── requirements.txt        # Pinned requirements
└── Context_Doc.md          # Project design specification
```

---

## 🚀 Getting Started

This project is managed with [`uv`](https://github.com/astral-sh/uv), the fast Python package and virtual environment manager.

### 1. Prerequisites
Ensure `uv` is installed:
```bash
uv --version
```
*(If not installed, install via `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux)*

### 2. Virtual Environment & Dependencies
Create the virtual environment and sync all dependencies:
```bash
uv venv
uv sync
```

To activate the virtual environment manually:
```bash
source .venv/bin/activate
```
*(Or run commands directly through `uv run ...`)*

---

## 🖥️ Running the Interactive Streamlit Dashboard

CommSimLab includes a multi-page interactive Streamlit dashboard featuring parameter sliders, real-time waveform updates, and spectrum analyzers:

```bash
uv run streamlit run app.py
```

### Dashboard Modules:
- **📶 Channel & Noise:** Add AWGN noise at variable SNR, calculate Friis cascaded noise figure and noise temperature.
- **📊 Shannon Capacity:** Calculate Shannon channel capacity and Nyquist data rates with interactive parameter plots.
- **🔊 Analog Modulation:** Inspect AM modulation indices ($m < 1$, $m = 1$, $m > 1$), power distribution, envelope demodulation, and FM frequency deviation with Carson's bandwidth rule.
- **📻 Radio Receivers:** Calculate superheterodyne local oscillator ranges, capacitor tuning ratios, and pre/de-emphasis filters.
- **⏱️ Sampling:** Vary sampling rates to watch aliasing appear when $f_s < 2f_{\max}$ and verify sinc reconstruction.
- **🔢 Quantization:** Uniform mid-rise quantizer with SQNR vs. bit depth ($6.02n + 1.76$ dB rule).
- **💾 PCM:** Full pipeline: analog input $\to$ sampler $\to$ quantizer $\to$ binary encoder $\to$ decoder $\to$ reconstructed wave.
- **📐 Delta Modulation:** Compare standard DM with Adaptive Delta Modulation (ADM); detect slope overload and granular noise.
- **🔲 Line Coding:** Side-by-side comparison of Unipolar NRZ, Polar NRZ, NRZ-I, Unipolar RZ, Polar RZ, AMI, Manchester, and Differential Manchester with Welch PSD plots.
- **📺 TV Systems:** NTSC raster scan timing calculator and idealized VSB spectrum display.

---

## 🧪 Running Unit Tests

Run the complete test suite with `pytest`:
```bash
uv run pytest tests/ -v
```

All 39 unit tests verify core signal processing functions and mathematical formulas against textbook references (Haykin, Proakis, Lathi).

---

## 🔬 Jupyter Notebook Experiments

Explore the experiment notebooks inside `notebooks/`:
```bash
uv run jupyter lab notebooks/
```
Each notebook covers a core unit with executable code blocks and Matplotlib figures.

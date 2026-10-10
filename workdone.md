Viewed pcm.py:15-115
Viewed quantization.py:1-60
Viewed radio.py:100-130
Viewed pcm.py:115-179
Viewed pcm.py:1-30
Viewed app.py:340-355
Ran command: `grep -n "pcm_decode" app.py`
Edited pcm.py
Edited radio.py
Viewed test_modules.py:170-195
Viewed app.py:475-495
Edited app.py
Edited test_modules.py
Ran command: `uv run pytest tests/ -v`
Used tool: manage_task
Ran command: `ls -la notebooks/`
Ran command: `uv pip compile pyproject.toml -o requirements.txt`
Created requirements.txt
Ran command: `uv run python -c "
import json

def make_notebook(cells):
    return {
        'cells': cells,
        'metadata': {
            'language_info': {
                'name': 'python',
                'version': '3.13'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

def code_cell(code):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': code.strip().splitlines(keepends=True)
    }

def markdown_cell(md):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': md.strip().splitlines(keepends=True)
    }

# Notebook 1: Channel & Capacity
nb1_cells = [
    markdown_cell('# Unit 1: AWGN Channel & Shannon Capacity\n\nExperiments on noise simulation, Friis formula, and Shannon-Hartley capacity.'),
    code_cell('''import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from commlab.channel import awgn_channel, snr_db, cascaded_noise_figure, plot_signal_with_noise
from commlab.capacity import shannon_capacity_db, plot_capacity_vs_snr
from commlab.analog import generate_tone

# Generate 20 Hz tone
t, signal = generate_tone(freq=20, duration=0.2, fs=5000)

# Pass through AWGN channel with SNR = 15 dB
noisy, noise = awgn_channel(signal, snr_db=15)
print(f'Measured SNR: {snr_db(signal, noise):.2f} dB')

fig = plot_signal_with_noise(t, signal, noisy, title='AWGN Channel (SNR = 15 dB)')
plt.show()'''),
    code_cell('''# Friis Formula for 3-stage amplifier
gains = [12.0, 10.0, 8.0] # dB
nfs = [2.5, 4.0, 6.0]     # dB
overall_nf = cascaded_noise_figure(gains, nfs)
print(f'Overall Noise Figure: {overall_nf:.2f} dB')

# Shannon Capacity Plot
fig_cap = plot_capacity_vs_snr(bandwidth_hz=4000, snr_db_range=np.arange(-5, 45, 1))
plt.show()''')
]

with open('notebooks/01_channel_and_capacity.ipynb', 'w') as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# Notebook 2: Analog Modulation
nb2_cells = [
    markdown_cell('# Unit 2: Analog Modulation (AM & FM)\n\nExperiments with AM modulation index, overmodulation, spectrum, and FM Carson bandwidth.'),
    code_cell('''import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from commlab.analog import generate_tone, am_modulate, am_power, am_efficiency, plot_am, plot_spectrum, fm_modulate, plot_fm

# Generate message and AM modulate with m = 0.5
fs = 10000
t, msg = generate_tone(freq=20, duration=0.2, fs=fs)
t, am_sig, carrier, envelope = am_modulate(msg, carrier_freq=400, fs=fs, modulation_index=0.5)

print(f'Normalized Power Ratio Pt/Pc: {am_power(1.0, 0.5):.4f}')
print(f'Modulation Efficiency: {am_efficiency(0.5)*100:.2f}%')

fig = plot_am(t, msg, carrier, am_sig, envelope)
plt.show()

fig_spec = plot_spectrum(am_sig, fs, title='AM Spectrum (Carrier + 2 Sidebands)')
plt.show()'''),
    code_cell('''# FM Modulation
t_fm, msg_fm = generate_tone(freq=15, duration=0.2, fs=20000)
t_fm, fm_sig = fm_modulate(msg_fm, carrier_freq=800, fs=20000, freq_deviation=300)

fig_fm = plot_fm(t_fm, msg_fm, fm_sig)
plt.show()''')
]

with open('notebooks/02_analog_modulation.ipynb', 'w') as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# Notebook 3: Sampling & PCM
nb3_cells = [
    markdown_cell('# Unit 3: Sampling Theorem & Pulse Code Modulation (PCM)\n\nDemonstration of Nyquist rate, aliasing, quantization SQNR, and full PCM reconstruction.'),
    code_cell('''import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from commlab.analog import generate_tone
from commlab.sampling import sample_signal, reconstruct_signal, check_aliasing, plot_sampling_demo
from commlab.quantization import uniform_quantize, sqnr_db, sqnr_actual, plot_quantization
from commlab.pcm import pcm_encode, pcm_decode, pcm_reconstruct, plot_pcm_chain

# Sample a 40 Hz signal at 200 Hz (Nyquist is 80 Hz -> no aliasing)
fs_orig = 10000
fs_samp = 200
t, sig = generate_tone(freq=40, duration=0.1, fs=fs_orig)
s_times, samples, _ = sample_signal(sig, t, fs_orig, fs_samp)
reconstructed = reconstruct_signal(samples, s_times, t, fs_samp)

fig = plot_sampling_demo(t, sig, samples, reconstructed, title='Sampling & Sinc Reconstruction')
plt.show()'''),
    code_cell('''# Full PCM Pipeline (8-bit)
result = pcm_encode(sig, t, fs_sample=200, num_bits=8)
decoded = pcm_decode(result['codes'], result['step_size'], result['v_min'], 8)
pcm_recon = pcm_reconstruct(decoded, result['sample_times'], t, 200)

fig_pcm = plot_pcm_chain(t, sig, result['sample_times'], result['samples'], result['quantized'], pcm_recon)
plt.show()''')
]

with open('notebooks/03_sampling_and_pcm.ipynb', 'w') as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

# Notebook 4: Delta Modulation & Line Coding
nb4_cells = [
    markdown_cell('# Unit 4: Delta Modulation & Digital Line Coding\n\nComparing DM vs Adaptive DM, plus line coding waveforms (NRZ, RZ, Manchester) and PSDs.'),
    code_cell('''import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from commlab.analog import generate_tone
from commlab.delta import dm_encode, adaptive_dm_encode, plot_adm_comparison
from commlab.linecode import random_bits, plot_all_line_codes, plot_all_psd

t, sig = generate_tone(freq=10, duration=0.5, fs=2000)
_, dm_stair, _ = dm_encode(sig, step_size=0.1)
_, adm_stair, _ = adaptive_dm_encode(sig, step_size_init=0.1)

fig_dm = plot_adm_comparison(t, sig, dm_stair, adm_stair)
plt.show()'''),
    code_cell('''# Line Coding Waveforms & PSD
bits = random_bits(8)
fig_lc = plot_all_line_codes(bits, bit_rate=1000, fs=50000)
plt.show()

bits_psd = random_bits(128)
fig_psd = plot_all_psd(bits_psd, bit_rate=1000, fs=100000)
plt.show()''')
]

with open('notebooks/04_delta_modulation_and_linecoding.ipynb', 'w') as f:
    json.dump(make_notebook(nb4_cells), f, indent=2)

print('Successfully created 4 unit experiment notebooks!')
"`
Created README.md
Ran command: `ls -la && ls -la commlab/`
Ran command: `rm -f test_linecode.py`
Ran command: `uv run pytest tests/ -v`
Used tool: manage_task

The **CommSimLab** project has been set up following all requirements from [Context_Doc.md](./Context_Doc.md) using `uv` and a dedicated virtual environment.

---

### 📦 Setup & Environment
- **Virtual Environment:** Configured with `uv venv` at [.venv/](./.venv).
- **Dependencies Installed:** `numpy`, `scipy`, `matplotlib`, `streamlit`, and `pytest` synced via `uv sync`.
- **Project Config:** Updated [pyproject.toml](./pyproject.toml) and [requirements.txt](./requirements.txt).

---

### 📁 Implemented Modules (`commlab/`)

| Module | Core Features & Formulas |
|---|---|
| [`channel.py`](./commlab/channel.py) | AWGN generation at chosen SNR, SNR measurement, Friis cascaded noise figure ($F = F_1 + \frac{F_2-1}{G_1} + \dots$), equivalent noise temperature ($T_e$). |
| [`capacity.py`](./commlab/capacity.py) | Shannon capacity ($C = B \log_2(1 + \text{SNR})$), Nyquist max rate ($R = 2B \log_2 M$), minimum levels calculator, parameter curves. |
| [`analog.py`](./commlab/analog.py) | AM modulation ($s(t) = A_c[1+m\cdot x(t)]\cos(2\pi f_c t)$), power ratio ($P_t = P_c(1 + m^2/2)$), overmodulation check, envelope demodulation (Hilbert), FM modulation, Carson's rule ($\text{BW} = 2(\Delta f + f_m)$), discriminator demodulation, VSB filter. |
| [`radio.py`](./commlab/radio.py) | Superheterodyne local oscillator tuning range, capacitor ratio ($C_{\max}/C_{\min}$), image frequency, AGC gain, 75 $\mu\text{s}$ pre/de-emphasis filters. |
| [`sampling.py`](./commlab/sampling.py) | Nyquist rate ($2 f_{\max}$), natural gating & flat-top sampling, sinc interpolation reconstruction, aliasing detection. |
| [`quantization.py`](./commlab/quantization.py) | Uniform mid-rise quantizer, quantization noise power ($\Delta^2/12$), theoretical SQNR ($6.02n + 1.76$ dB), actual SQNR, PAM generation & demodulation. |
| [`pcm.py`](./commlab/pcm.py) | Full PCM pipeline: `pcm_encode` (sampler $\to$ quantizer $\to$ binary encoder), `pcm_decode`, `pcm_reconstruct`, bit rate & transmission bandwidth. |
| [`delta.py`](./commlab/delta.py) | DPCM with linear predictor, standard Delta Modulation (DM), Adaptive Delta Modulation (ADM), slope overload and granular noise detection. |
| [`linecode.py`](./commlab/linecode.py) | Unipolar NRZ, Polar NRZ, NRZ-I, Unipolar RZ, Polar RZ, AMI (Bipolar), Manchester (IEEE 802.3), Differential Manchester, and Welch PSD estimation. |
| [`tv.py`](./commlab/tv.py) | NTSC raster scan timing parameters, test pattern generator (color bars, checkerboard, gradient), and VSB TV channel spectrum model. |

---

### 🖥️ Interactive Streamlit Dashboard ([`app.py`](./app.py))

Run with `uv`:
```bash
uv run streamlit run app.py
```
Includes multi-page navigation across all modules with live parameter sliders (SNR, modulation indices, sampling rates, bit depths, filter time constants) and interactive Matplotlib figures.

---

### 📓 Experiment Notebooks (`notebooks/`)
Created 4 ready-to-run Jupyter notebooks under [notebooks/](./notebooks/):
- `01_channel_and_capacity.ipynb`
- `02_analog_modulation.ipynb`
- `03_sampling_and_pcm.ipynb`
- `04_delta_modulation_and_linecoding.ipynb`

---

### ✅ Verification
All 39 unit tests in [tests/test_modules.py](./tests/test_modules.py) pass cleanly:
```bash
uv run pytest tests/ -v
# Output: 39 passed in 11.62s
```
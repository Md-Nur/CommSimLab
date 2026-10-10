import numpy as np
import matplotlib.pyplot as plt
from commlab.linecode import *

bits = random_bits(10)
print("Bits:", bits)
t, s = nrz_unipolar(bits, 1000, 10000)
print("Shape of t and s:", t.shape, s.shape)
fig1 = plot_line_code(t, s, bits, 1000)
fig2 = plot_all_line_codes(bits, 1000, 10000)
fig3 = plot_all_psd(bits, 1000, 10000)
print("Plots generated successfully!")

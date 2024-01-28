from math import sqrt, pi, exp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

LINEWIDTH = 3
MARKERSIZE = 10
OPACITY = 1
COLORS = sns.color_palette("bright")

eps = .1
delta = .1
rho = .9
h = .49
r = 1.5

dimensions = np.array([1, 5, 10, 15, 20, 25, 30])
Ns = np.zeros(dimensions.shape)
Ms = np.zeros(dimensions.shape)
for i in range(0, len(dimensions)):
  d = dimensions[i]
  rho_r = 1 - rho**2 - (1 - rho**2) * (d + 1)/(r * d + 1)
  E_w = ( 1/( 2 * sqrt(2 * h) ) + sqrt(h/2) )**d
  R = r * d + 1
  K = (1 - rho**2) * (d + 1)
  Ns[i] = E_w * delta**(-1) * eps**(-2) * 8 * (rho**2 * R + 2 * K - 1)**2 / (1 - rho_r)**2
  Ms[i] = delta**(-1) * eps**(-2) * 2 * (R + K) * ( rho_r / (2 - rho_r) )**2

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

plt.plot(dimensions, Ns, 
             marker = "o",
             alpha = OPACITY,
             markersize = MARKERSIZE, 
             color = COLORS[0],
             linewidth = LINEWIDTH)

plt.ticklabel_format(style='plain')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Dimension", fontsize = 25, color="black")
plt.ylabel(r"N lower bound", fontsize = 25, color="black")
plt.savefig("ar_N_lb.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

plt.plot(dimensions, Ms, 
             marker = "o",
             alpha = OPACITY,
             markersize = MARKERSIZE, 
             color = COLORS[0],
             linewidth = LINEWIDTH)

plt.ticklabel_format(style='plain')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Dimension", fontsize = 25, color="black")
plt.ylabel(r"M lower bound", fontsize = 25, color="black")
plt.savefig("ar_M_lb.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')


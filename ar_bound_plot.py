from math import sqrt, pi, exp
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

LINEWIDTH = 3
MARKERSIZE = 10
OPACITY = 1
COLORS = sns.color_palette("bright")

plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

eps = .1
delta = .1
rho = .9
h = .4
r = 1.5

dimensions = np.array([1, 5, 10, 15, 20, 25, 30])
Ns = np.zeros(dimensions.shape)

for i in range(0, len(dimensions)):
  d = dimensions[i]
  rho_r = 1 - rho**2 - (1 - rho**2) * (d + 1)/(r * d + 1)
  Ns[i] = delta**(-1) * eps**(-2) * d/(1 - rho_r) * ( 1/( 2 * sqrt(2 * h) ) + sqrt(h/2) )**d
plt.plot(dimensions, Ns, 
             marker = "o",
             alpha = OPACITY,
             markersize = MARKERSIZE, 
             color = COLORS[0],
             linewidth = LINEWIDTH)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel(r"Dimension", fontsize = 25, color="black")
plt.ylabel(r"Number of independent Markov chains", fontsize = 25, color="black")
plt.savefig("ar_bound_plot.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight',)


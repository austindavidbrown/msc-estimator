import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def self_normalized_importance_weights(xs):
  s = np.sum((xs - m) * (xs - m), 1)
  neg_log_proposal = 1/(2 * (1/2 + h)) * s
  neg_log_target = 1/(2) * s
  ws = np.exp(neg_log_proposal - neg_log_target)
  return ws/np.sum( ws )

def sim_mcmc():
  run_lengths = np.zeros(M)

  proposal_xs = m + h**(1/2) * np.random.normal(size = (N, d))
  init_probs = self_normalized_importance_weights(proposal_xs)
  init_indices = np.random.choice(np.arange(0, N), 
                                  M, 
                                  p=init_probs, 
                                  replace = True)
  init_xs = proposal_xs[init_indices]

  V_computes = np.sum( (init_xs - m) * (init_xs - m) , 1)
  update_indices = np.where( V_computes <=  r * d )[0]
  
  x = np.zeros((M, d))
  x[update_indices] = init_xs[update_indices]

  # Generate sums
  sums = np.zeros((M, d))
  for t in range(1, 10**6):
    # sample
    x_new = np.zeros((M, d))
    xi = np.random.normal(size = x[update_indices].shape)
    x_new[update_indices] = (1 - rho) * m + rho * x[update_indices] + (1 - rho**2)**(1/2) * xi
    
    # Update
    x[update_indices] = x_new[update_indices]
    sums[update_indices] += x_new[update_indices]

    # Compute new V values and remove any update indices that have hit C
    V_computes[update_indices] = np.sum( (x_new[update_indices] - m) * (x_new[update_indices] - m), 1)
    update_indices = np.where( V_computes > r * d )[0]

    # Update run lengths
    run_lengths[np.logical_and(V_computes <= r * d, run_lengths == 0)] = t

    # Break if there are no update indices left
    if update_indices.shape[0] == 0:
      break

  return sums, run_lengths


###
# Constants
###
np.random.seed(1)
N =  10**6
M =  10**5
d = 2
h = .4 # h variance
m = 0
rho = .9
r = 1.5

samples, run_lengths = sim_mcmc()
true_samples = np.random.normal(size = (M, d), loc = m, scale = 1)


def compute_diagnostics(X):
  n_iterations = X.shape[0]
  d = X.shape[1]
  shift = int(n_iterations / 4)
  n_computes = 1 + int((n_iterations - shift) / shift)
  iterations = np.arange(shift, n_iterations + shift, shift)

  means = np.zeros((n_computes, d))
  stds = np.zeros((n_computes, d))

  for k in range(0, n_computes):
    m = (shift * (k + 1))
    X_m = X[0:m, :]

    means[k] = np.mean(X_m, 0)
    stds[k] = np.std(X_m, 0)/m**(1/2)

  return {
    "iterations": iterations,
    "means": means,
    "stds": stds
  }



###
# Plot the mean and stds
###
true_diagnostics = compute_diagnostics(true_samples)
true_means = true_diagnostics["means"]
true_stds = true_diagnostics["stds"]

diagnostics = compute_diagnostics(samples)
iterations = diagnostics["iterations"]
means = diagnostics["means"]
stds = diagnostics["stds"]

COLORS = sns.color_palette("bright")

LINEWIDTH = 3
MARKERSIZE = 8
OPACITY = 1

###
# Plot the run lengths
###
plt.clf()
plt.figure(figsize=(10, 8))
plt.style.use("seaborn-v0_8-whitegrid")

plt.vlines(x=np.arange(1, run_lengths.shape[0] + 1),
           alpha = .9,
           ymin=np.zeros(run_lengths.shape[0]), 
           ymax=run_lengths, 
           colors = COLORS[0], 
           ls = '-', 
           lw = 1)

plt.tick_params(axis="x", labelsize=20)
plt.tick_params(axis="y", labelsize=20)
plt.xlabel("Independent Markov chain M", fontsize=30)
plt.ylabel("Run length", fontsize=30)
plt.savefig("ar_runlength.pdf", 
            format='pdf', 
            pad_inches=0, 
            bbox_inches='tight')

###
# Plot samples
###
for k in range(0, d):
  true_means_k = true_means[:, k]
  true_stds_k = true_stds[:, k]

  means_k = means[:, k]
  stds_k = stds[:, k]

  plt.clf()
  plt.figure(figsize=(10, 8))
  plt.style.use("seaborn-v0_8-whitegrid")

  plt.plot(iterations, true_means_k, 
    '-', label="MC mean", 
    alpha = OPACITY, 
    marker = "s", 
    markersize = MARKERSIZE, 
    color = COLORS[2], 
    linewidth = LINEWIDTH)
  plt.fill_between(iterations, 
                   true_means_k - 2 * true_stds_k, 
                   true_means_k + 2 * true_stds_k, 
                   alpha=0.1,
                   color=COLORS[2])

  plt.plot(iterations, means_k, 
    '-', label="MSC mean", 
    alpha = OPACITY, 
    marker = "o", 
    markersize = MARKERSIZE, 
    color = COLORS[0], 
    linewidth = LINEWIDTH)
  plt.fill_between(iterations, 
                   means_k - 2 * stds_k, 
                   means_k + 2 * stds_k, 
                   alpha=0.1,
                   color=COLORS[0])

  plt.tick_params(axis="x", labelsize=20)
  plt.tick_params(axis="y", labelsize=20)
  plt.xlabel("M iterations", fontsize=30)
  plt.ylabel("Empirical mean", fontsize=30)
  plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
  plt.savefig("ar_means_%d.pdf" % (k + 1), 
              format='pdf', 
              pad_inches=0, 
              bbox_inches='tight')



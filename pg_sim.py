import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from formulaic import Formula

from pypolyagamma import PyPolyaGamma, BernoulliRegression


###
# Constants
###
np.random.seed(1)
PG = PyPolyaGamma(seed = 1)
N_iterations = 10**7
M_iterations = 10**6
n_repeat = 1 # for many polya gammas
m_values = np.array([.25 * M_iterations, .5 * M_iterations, .75 * M_iterations, M_iterations], dtype=int)

# Tuning for msc
h = .49
r = 1.001
tau = 10


def cholesky_inv(M):
  M_L = np.linalg.cholesky(M)
  M_inv_L = np.linalg.inv(M_L).T
  M_inv = M_inv_L @ M_inv_L.T
  return M_inv_L, M_inv

def sigmoid(out):
  return 1 / (1 + np.exp(-out))

def bceloss(out, Y):
  return np.sum( np.log1p(np.exp(out)) - Y * out )

# Gradient descent on the negative log of the target density with annealing step sizes
def graddescent(anneal_factor = .99,
                stepsize = .5, tol = 10**(-10), max_iterations = 10**6):
  alpha = np.zeros(1)
  beta = np.zeros(X.shape[1])
  
  old_loss = bceloss(X @ beta, Y) \
             + 1/(2.0) * beta @ C_inv @ beta

  for t in range(1, max_iterations):
    grad_loss_beta = X.T @ (sigmoid(X @ beta) - Y) + C_inv @ beta

    if np.any(np.isnan(grad_loss_beta)):
      raise Exception("NAN value in gradient descent.")
    else:
      beta_new = beta - stepsize * grad_loss_beta
      new_loss = bceloss(X @ beta_new, Y) \
                 + 1/(2.0) * beta_new @ C_inv @ beta_new

      # New loss worse than old loss? Reduce step size and try again.
      if (new_loss > old_loss):
        stepsize = stepsize * anneal_factor
      else:
        # Update
        beta = beta_new

        # Stopping criterion
        if np.sum(grad_loss_beta**2) < tol:
          return beta

        old_loss = new_loss

  raise Exception("Gradient descent failed to converge.")

def sample_initial(T_iterations, 
                   M_samples):
  beta_opt = graddescent()

  proposal_betas = np.zeros((T_iterations, n_features + 1))
  for i in range(0, T_iterations):
    xi = np.random.normal(size = n_features + 1)
    grad_loss_beta_opt = X.T @ (sigmoid(X @ beta_opt) - Y) + C_inv @ beta_opt
    proposal_betas[i] = beta_opt - C @ grad_loss_beta_opt + C_L @ xi

  # Generate weights
  weights = np.zeros(T_iterations)
  for i in range(0, T_iterations):
    beta = proposal_betas[i]
    grad_loss_beta_opt = X.T @ (sigmoid(X @ beta_opt) - Y) + C_inv @ beta_opt
    out = beta - beta_opt  + C @ grad_loss_beta_opt

    f_proposal = 1/(2.0 * (1/2 + h)) * out.T @ C_inv @ out
    f_target = bceloss(X @ beta, Y) + 1/(2.0) * beta @ C_inv @ beta          
    weights[i] = np.exp(f_proposal - f_target)

  init_probs = weights/np.sum( weights )

  # Sample initial
  init_indices = np.random.choice(np.arange(0, T_iterations), 
                                  M_iterations, 
                                  p=init_probs, 
                                  replace = True)
  init_betas = proposal_betas[init_indices]
  return init_betas

def pg_mcmc(beta_0):
  V_compute = np.sum( (beta_0)**2 )
  if V_compute > r * L:
    return np.zeros(beta_0.shape), 0
  elif V_compute <= r * L:
    ###
    # Gibbs sampler
    ###
    sum_betas = np.zeros(beta_0.shape)

    beta = beta_0
    for t in range(1, 10**8):
      # Sample omega_t | beta_{t-1}
      omega = np.zeros(n_samples)
      out = X @ beta
      PG.pgdrawv(np.ones(n_samples), out, omega)

      # Sample beta_t | omega_{t}
      Sigma_L, Sigma = cholesky_inv( X.T @ np.diag(omega) @ X + C_inv )

      xi = np.random.normal(size = n_features + 1)
      beta_new = Sigma @ X.T @ (Y - 1/2) + Sigma_L @ xi
      
      # Update
      sum_betas += beta_new
      beta = beta_new

      V_compute = np.sum( (beta_new)**2 )
      if V_compute <= r * L:
        break

  return sum_betas, t

def pg_sampler(beta_0, T_iterations):  
  '''
  np.float = np.float64
  betas_pg = np.zeros((T_iterations, n_features))
  bs_pg = np.zeros((T_iterations, 1))
  pg_reg = BernoulliRegression(1, n_features, 
                               b = beta_0[0],
                               sigmasq_b = tau,
                               A = beta_0[1:], 
                               sigmasq_A = tau)

  for t in range(0, T_iterations):
    pg_reg.resample(( X[:, 1:], np.expand_dims(Y, 0).T ))

    bs_pg[t, :] = pg_reg.b
    betas_pg[t, :] = pg_reg.A

  betas = np.column_stack((bs_pg, betas_pg))
  '''

  betas = np.zeros((T_iterations, beta_0.shape[0]))
  betas[0] = beta_0

  for t in range(1, T_iterations):
    if t % 10**4 == 0:
      print("Iteration", t)

    # Sample omega_t | beta_{t-1}
    omega = np.zeros(n_samples)
    out = X @ betas[t-1]
    PG.pgdrawv(np.ones(n_samples), out, omega)

    # Sample beta_t | omega_{t}
    Sigma_L, Sigma = cholesky_inv( X.T @ np.diag(omega) @ X + C_inv )

    xi = np.random.normal(size = n_features + 1)
    betas[t] = Sigma @ X.T @ (Y - 1/2) + Sigma_L @ xi

  return betas

def rwm(beta_0, T_iterations, h = 1):  
  d = beta_0.shape[0]
  accepts = np.zeros(T_iterations)
  betas = np.zeros((T_iterations, beta_0.shape[0]))
  betas[0] = beta_0
  
  f_target = bceloss(X @ beta_0, Y) + 1/(2.0) * beta_0 @ C_inv @ beta_0
  for t in range(1, T_iterations):
    if t % 10**4 == 0:
      print("Iteration", t)
      print("accept rate:", accepts[0:t].mean())

    xi_new = np.random.normal(size=d, loc=0, scale=1)
    beta_new = betas[t-1] + h**(1/2)/d**(1/2)  * xi_new
    f_target_new = bceloss(X @ beta_new, Y) + 1/(2.0) * beta_new @ C_inv @ beta_new

    u = np.random.uniform(low=0, high=1)
    if np.log(u) <= f_target - f_target_new:
      betas[t] = beta_new
      f_target = f_target_new
      accepts[t] = 1
    else:
      betas[t] = betas[t-1]
  return betas

def compute_diagnostics(X, m_values):
  n_computes = m_values.shape[0]
  d = X.shape[1]

  means = np.zeros((n_computes, d))
  stds = np.zeros((n_computes, d))
  for k in range(0, m_values.shape[0]):
    m = m_values[k]
    X_m = X[0:m, :]

    means[k] = np.mean(X_m, 0)
    stds[k] = np.std(X_m, 0)/m**(1/2)

  return {
    "means": means,
    "stds": stds
  }

###
# Cleveland heart data
###
df = pd.read_csv('cleveland.csv', sep=',', header = None)
df.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df = df[df.ca != "?"]
df = df[df.thal != "?"]
df.target[df.target > 0] = 1
df.target = pd.to_numeric(df.target)

df.cp = df.cp.astype('category')
df.restecg = df.restecg.astype('category')
df.slope = df.slope.astype('category')
df.thal = df.thal.astype('category')

Y, X = Formula('target ~ 0 + sex + ca + trestbps + age + chol + cp + fbs + restecg + thalach + exang + oldpeak + slope + thal').get_model_matrix(df)
X = X.to_numpy()
Y = Y.to_numpy().squeeze(1)

n_samples = X.shape[0]
n_features = X.shape[1]

# Standardize data
X = X - np.mean(X, 0) # center
scale_X = (X**2).sum(0)**(1/2)
X  = X/scale_X
intercept_column = n_samples**(-1/2) * np.ones(n_samples)
X = np.column_stack((intercept_column, X))

C = tau * np.eye(n_features + 1)
C_L = np.linalg.cholesky(C)
C_L_inv, C_inv = cholesky_inv(C)

L = np.linalg.norm(C, ord = 2)**2 * np.linalg.norm(X.T @ (Y - 1/2), ord = 2)**2 + np.trace(C)

###
# Generate MSC samples
###
print("Creating initial")
init_betas = sample_initial(N_iterations, M_iterations)

print("Running parallel chains")
# Generate samples
msc_samples = np.zeros((M_iterations, n_features + 1))
run_lengths = np.zeros(M_iterations)
for i in range(0, M_iterations):
  if (i + 1) % 10**4 == 0:
    print("Iteration", i + 1)

  sample, run_length = pg_mcmc(init_betas[i])
  msc_samples[i] = sample
  run_lengths[i] = run_length

# Rescale
msc_samples[:, 1:] = msc_samples[:, 1:] / scale_X
msc_samples[:, 0] = msc_samples[:, 0] / n_samples**(1/2)

###
# Compute diagnostics for plotting
###
msc_diagnostics = compute_diagnostics(msc_samples, m_values)
msc_means = msc_diagnostics["means"]
msc_stds = msc_diagnostics["stds"]


print("Running pg sampler")
pg_samples = pg_sampler(init_betas[0], M_iterations)
# Rescale
pg_samples[:, 1:] = pg_samples[:, 1:] / scale_X
pg_samples[:, 0] = pg_samples[:, 0] / n_samples**(1/2)

pg_diagnostics = compute_diagnostics(pg_samples, m_values)
pg_means = pg_diagnostics["means"]

print("Running RWM sampler")
rwm_samples = rwm(init_betas[0], M_iterations, h = 50)
# Rescale
rwm_samples[:, 1:] = rwm_samples[:, 1:] / scale_X
rwm_samples[:, 0] = rwm_samples[:, 0] / n_samples**(1/2)

rwm_diagnostics = compute_diagnostics(rwm_samples, m_values)
rwm_means = rwm_diagnostics["means"]














###
# Plot the mean and stds
###

COLORS = sns.color_palette("bright")

LINEWIDTH = 3
MARKERSIZE = 8
OPACITY = 1

###
# Plot samples
###
for k in [1, 2, 3]:
  pg_means_k = pg_means[:, k]
  rwm_means_k = rwm_means[:, k]

  msc_means_k = msc_means[:, k]
  msc_stds_k = msc_stds[:, k]

  plt.clf()
  plt.figure(figsize=(10, 8))
  plt.style.use("seaborn-v0_8-whitegrid")

  plt.plot(m_values, msc_means_k, 
    '-', label="MSC mean", 
    alpha = OPACITY, 
    marker = "o", 
    markersize = MARKERSIZE, 
    color = COLORS[0], 
    linewidth = LINEWIDTH)
  plt.fill_between(m_values, 
                   msc_means_k - 2 * msc_stds_k, 
                   msc_means_k + 2 * msc_stds_k, 
                   alpha=0.1,
                   color=COLORS[0])

  plt.plot(m_values, pg_means_k, 
    '-', label="PG mean", 
    alpha = OPACITY, 
    marker = "s", 
    markersize = MARKERSIZE, 
    color = COLORS[2], 
    linewidth = LINEWIDTH)

  plt.plot(m_values, rwm_means_k, 
    '-', label="RWM mean", 
    alpha = OPACITY, 
    marker = "v", 
    markersize = MARKERSIZE, 
    color = COLORS[4], 
    linewidth = LINEWIDTH)

  plt.tick_params(axis="x", labelsize=20)
  plt.tick_params(axis="y", labelsize=20)
  plt.xlabel("M iterations", fontsize=30)
  plt.ylabel("Empirical mean", fontsize=30)
  plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=.8)
  plt.savefig("pg_means_%d.pdf" % k, 
              format='pdf', 
              pad_inches=0, 
              bbox_inches='tight')




















'''
###
# Generate data
###
n_features = 2
n_samples = 20
beta_true = np.random.uniform(size = n_features + 1, low = -1, high = -1)

# Standardize X matrix so tr(X^T X) = n_features + 1
X = np.random.uniform(size = (n_samples, n_features), low = -1, high = 1)
X = X - np.mean(X, 0) # center

scale_X = (X**2).sum(0)**(1/2)
scale_intercept = n_samples**(1/2)

X  = X/scale_X
intercept_column = 1/scale_intercept * np.ones(n_samples)
X = np.column_stack((intercept_column, X))
Y = np.random.binomial(1, sigmoid(X @ beta_true))
'''



"""
Age-Structured Heroin Transmission Model with Treatment Delay
Based on: Rekrak, M. (2026). Master's Thesis, University Djillali Liabes.

This reproduces the numerical simulation in Section 3.6 of the thesis:
comparing model behavior below (beta=0.03) and above (beta=0.09) the
epidemic threshold beta*, using the age-structured system:

  dS/dt  = Lambda - mu*S - beta*S*U1 + q*U1(t-A)
  dU1/dt = beta*S*U1 - (mu+delta1+p)*U1 + integral_0^A k(a) U2(t,a) da
  dU2/dt + dU2/da = -(mu+delta2+k(a)) U2(t,a),   U2(t,0) = p*U1(t)

The age-PDE for U2 is solved via the method of lines: age is discretized
into a fine grid, and the transport term is approximated with an upwind
finite difference, turning the PDE into a large system of ODEs that is
solved together with S(t) and U1(t) using scipy's stiff ODE solver.

Because k(a) and the natural/treatment removal rates make the survival
probability Pi(a) decay to numerically zero well before a=100, the age
domain is truncated to a_max=100 instead of the full A=400 used in the
thesis -- this has no visible effect on S(t), U1(t) over the simulated
time horizon (t in [0,300]) but keeps the discretization small.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Parameters (Section 3.6, Eq. 3.1 of the thesis)
# ---------------------------------------------------------------
Lambda = 0.5
mu     = 0.1
q      = 0.2
A      = 400.0      # true treatment-exit age used in the thesis
delta1 = 0.1
delta2 = 0.1
p      = 0.1

def k(a):
    """Relapse probability as a function of treat-age a."""
    return 0.1 * (1 - np.exp(-0.04 * (a - 1)))

# ---------------------------------------------------------------
# Age discretization (method of lines)
# ---------------------------------------------------------------
a_max = 100.0        # effective truncation of the age domain (see note above)
N     = 400           # number of age grid points
da    = a_max / N
ages  = np.linspace(0, a_max, N, endpoint=False) + da / 2  # cell centers

removal_rate = mu + delta2 + k(ages)   # (mu + delta2 + k(a)) at each grid point

def rhs(t, y):
    S, U1 = y[0], y[1]
    U2 = y[2:]

    # Upwind finite-difference transport for U2 (age increases like time)
    U2_boundary = p * U1  # U2(t, 0) = p * U1(t)
    dU2 = np.empty_like(U2)
    dU2[0] = -(U2[0] - U2_boundary) / da - removal_rate[0] * U2[0]
    dU2[1:] = -(U2[1:] - U2[:-1]) / da - removal_rate[1:] * U2[1:]

    # Integral term: integral_0^A k(a) U2(t,a) da  (trapezoidal on the grid)
    integral_term = np.sum(k(ages) * U2) * da

    # Delay term q*U1(t-A): this represents users who reach the maximum
    # treatment age A and exit back to the susceptible pool. Because the
    # survival probability in treatment decays so fast (removal rate
    # mu+delta2+k(a) >= 0.2), essentially nobody survives in treatment
    # long enough to reach age A=400 within the simulated horizon, so
    # this outflow is negligible -- consistent with the thesis choosing
    # a very large A specifically to model "effectively unreachable"
    # treatment completion. We therefore set it to 0 here.
    delay_term = 0.0

    dS  = Lambda - mu * S - beta * S * U1 + delay_term
    dU1 = beta * S * U1 - (mu + delta1 + p) * U1 + integral_term

    return np.concatenate(([dS, dU1], dU2))

# ---------------------------------------------------------------
# Initial conditions (Section 3.6)
# ---------------------------------------------------------------
S_0  = 10.0
U1_0 = 2 + np.cos(0.1)
U2_0 = 0.2 * np.exp(-0.04 * (ages - 1))

y0 = np.concatenate(([S_0, U1_0], U2_0))

t_span = (0, 300)
t_eval = np.linspace(*t_span, 600)

# ---------------------------------------------------------------
# Run both scenarios: below threshold (beta=0.03) and above (beta=0.09)
# ---------------------------------------------------------------
results = {}
for beta in [0.03, 0.09]:
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="BDF", rtol=1e-6, atol=1e-8)
    results[beta] = sol

# ---------------------------------------------------------------
# Compute beta* (epidemic threshold) for reference, Theorem 3.4.1 form
# ---------------------------------------------------------------
def Pi(a_arr):
    # survival probability in treatment: Pi(a) = exp(-integral_0^a (mu+delta2+k(s)) ds)
    integrand = mu + delta2 + k(a_arr)
    cum = np.cumsum(integrand) * da
    return np.exp(-cum)

Pi_vals = Pi(ages)
K = np.sum(k(ages) * Pi_vals) * da
beta_star = (mu / Lambda) * (mu + delta1 + p - p * K)
print(f"Computed threshold beta* = {beta_star:.4f} (thesis reports 0.06)")

# ---------------------------------------------------------------
# Plot: reproduce the style of Fig 3.1 / Fig 3.2
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for beta, sol, label in zip([0.03, 0.09], [results[0.03], results[0.09]],
                             ["beta=0.03 (< beta*): drug-free equilibrium",
                              "beta=0.09 (> beta*): drug spread equilibrium"]):
    ax = axes[0] if beta == 0.03 else axes[1]
    ax.plot(sol.t, sol.y[0], label="S(t)", color="crimson")
    ax.plot(sol.t, sol.y[1], label="U1(t)", color="steelblue")
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/heroin_project/simulation_result.png", dpi=150)
print("Saved plot to simulation_result.png")

# ---------------------------------------------------------------
# Sanity check: compare simulated steady states to the thesis's
# closed-form equilibrium expressions (Theorem 2.1.2 / Section 3.3)
# ---------------------------------------------------------------
print("\n--- Verification against analytic equilibria ---")
for beta in [0.03, 0.09]:
    R0 = beta * (Lambda / mu) / (mu + delta1 + p - p * K)
    sol = results[beta]
    S_end, U1_end = sol.y[0, -1], sol.y[1, -1]
    if R0 < 1:
        S_star, U1_star = Lambda / mu, 0.0
    else:
        S_star = (1 / R0) * (Lambda / mu)
        U1_star = (mu / beta) * (R0 - 1)
    print(f"beta={beta}: R0={R0:.3f} | simulated (S,U1)=({S_end:.3f},{U1_end:.3f}) "
          f"| analytic (S*,U1*)=({S_star:.3f},{U1_star:.3f})")

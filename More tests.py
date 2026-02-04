#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ==========================================================
# SMR++ Sparse Manifold Regulation Framework
# ==========================================================

np.random.seed()

N = 1800
W = np.linspace(-6, 6, N)
DW = W[1] - W[0]
TARGET = 0.75
STEPS = 1400
DT = 0.002

# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def normalize(rho):
    rho = np.maximum(rho, 0)
    return rho / np.sum(rho * DW)

def entropy(p):
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))

def diffusion_matrix(D):
    a = D * DT / DW**2
    ab = np.zeros((3, N))
    ab[0,1:] = -a
    ab[1,:]  = 1 + 2*a
    ab[2,:-1]= -a
    ab[1,0] = ab[1,-1] = 1 + a
    return ab

# ----------------------------------------------------------
# Controllers
# ----------------------------------------------------------

def smr_controller(rho, tau, integ):
    active = np.sum(rho[np.abs(W) > tau] * DW)
    sigma = 1 - active
    e = TARGET - sigma
    integ += e * DT
    tau = max(0.02, tau + 0.18*e + 0.03*integ)
    return tau, sigma, integ

def soft_threshold(rho, lam=1.4):
    return normalize(np.exp(-lam*np.abs(W))*rho)

def hard_cut(rho, tau=1.2):
    rho[np.abs(W) < tau] = 0
    return normalize(rho)

# ----------------------------------------------------------
# Simulation core
# ----------------------------------------------------------

def simulate(method):
    rho = normalize(np.exp(-0.5*W**2))
    tau, integ = 1.0, 0.0
    sigmas, entropies = [], []

    for k in range(STEPS):
        if method == "SMR":
            tau, sigma, integ = smr_controller(rho, tau, integ)
        elif method == "SOFT":
            rho = soft_threshold(rho)
            sigma = np.sum(rho[np.abs(W)<1]*DW)
        elif method == "HARD":
            rho = hard_cut(rho)
            sigma = np.sum(rho[np.abs(W)<1]*DW)
        else:
            sigma = np.sum(rho[np.abs(W)<1]*DW)

        D = 0.03 + 0.25*(1-sigma)
        ab = diffusion_matrix(D)
        rho = solve_banded((1,1), ab, rho)
        rho = normalize(rho)

        # Noise shock mid-run
        if k == STEPS//2:
            rho += 0.02*np.random.randn(N)
            rho = normalize(rho)

        sigmas.append(sigma)
        entropies.append(entropy(rho))

    return rho, np.array(sigmas), np.array(entropies)

# ----------------------------------------------------------
# Run all methods
# ----------------------------------------------------------

methods = ["NONE","SMR","SOFT","HARD"]
results = {m: simulate(m) for m in methods}

# ----------------------------------------------------------
# Visualization
# ----------------------------------------------------------

fig, ax = plt.subplots(3,1, figsize=(11,12))

for m in methods:
    ax[0].plot(W, results[m][0], label=m)
ax[0].set_title("Final Density Profiles")
ax[0].legend()

for m in methods:
    ax[1].plot(results[m][1], label=m)
ax[1].axhline(TARGET, ls="--", c="k")
ax[1].set_title("Sparsity Convergence")
ax[1].legend()

for m in methods:
    ax[2].plot(results[m][2], label=m)
ax[2].set_title("Entropy (Lyapunov-like Stability)")
ax[2].legend()

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n=== FINAL METRICS ===")
for m in methods:
    print(f"{m:6}  sigma={results[m][1][-1]:.4f}   entropy={results[m][2][-1]:.4f}")

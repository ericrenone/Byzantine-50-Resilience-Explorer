#!/usr/bin/env python3
"""
Sparse Manifold Regulator (SMR)
Simulates adaptive thresholding in a diffusive density field using PI control.
Mathematical Model: 
- Diffusion: ∂ρ/∂t = D * ∂²ρ/∂w²
- Sparsity Control: τ(t) = Kp*e(t) + Ki*∫e(t)dt
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import tkinter as tk
from tkinter import messagebox

# Backend configuration
matplotlib.use("TkAgg")

class ManifoldRegulator:
    def __init__(self, n=2000, target_silence=0.75):
        self.n = n
        self.target_silence = target_silence
        self.w = np.linspace(-5.0, 5.0, n)
        self.dw = self.w[1] - self.w[0]
        
        # Initial Density (Normal Distribution)
        self.rho = np.exp(-0.5 * self.w**2)
        self.rho /= np.sum(self.rho * self.dw)
        
        # History Logs
        self.history = {"sigma": [], "tau": [], "energy": []}

    def _setup_diffusion_matrix(self, dt, sigma_diff):
        """Sets up the tridiagonal matrix for implicit Crank-Nicolson or Backward Euler."""
        alpha = sigma_diff * dt / self.dw**2
        # Banded format: [upper, diag, lower]
        ab = np.zeros((3, self.n))
        ab[0, 1:] = -alpha          # Upper
        ab[1, :] = 1 + 2 * alpha    # Diag
        ab[2, :-1] = -alpha         # Lower
        
        # Neumann Boundary Conditions (Zero flux)
        ab[1, 0] = 1 + alpha
        ab[1, -1] = 1 + alpha
        return ab

    def run(self, steps=1200, dt=0.002, kp=0.15, ki=0.02, sigma_diff=0.05):
        """Executes the simulation loop."""
        tau = 1.0
        integral_error = 0.0
        ab = self._setup_diffusion_matrix(dt, sigma_diff)

        for _ in range(steps):
            # 1. Measure Sparsity (Sigma)
            active_mask = (np.abs(self.w) > tau)
            active_mass = np.sum(self.rho[active_mask] * self.dw)
            sigma = 1.0 - active_mass
            
            # 2. Entropy Energy Calculation
            energy = 0.0
            if active_mass > 1e-9: energy -= active_mass * np.log(active_mass)
            if sigma > 1e-9: energy -= sigma * np.log(sigma)
            
            # 3. PI Control Update for Threshold Tau
            error = self.target_silence - sigma
            integral_error += error * dt
            tau = max(0.01, tau + (kp * error + ki * integral_error))
            
            # 4. Implicit Diffusion Step
            self.rho = solve_banded((1, 1), ab, self.rho)
            
            # 5. Numerical Clean-up
            self.rho = np.maximum(self.rho, 0)
            self.rho /= np.sum(self.rho * self.dw)
            
            # Record keeping
            self.history["sigma"].append(sigma)
            self.history["tau"].append(tau)
            self.history["energy"].append(energy)

        return tau

    def plot(self):
        """Generates the 4-panel analysis figure."""
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))
        h = self.history
        t_axis = np.arange(len(h["sigma"]))

        # ρ(w)
        axes[0].fill_between(self.w, self.rho, color='royalblue', alpha=0.3)
        axes[0].plot(self.w, self.rho, color='blue', label="Density ρ(w)")
        axes[0].axvline(h["tau"][-1], color='red', linestyle="--", label="Final τ")
        axes[0].axvline(-h["tau"][-1], color='red', linestyle="--")
        axes[0].set_title("Manifold Density Equilibrium")
        axes[0].legend()

        # σ(t)
        axes[1].plot(t_axis, h["sigma"], color='green')
        axes[1].axhline(self.target_silence, color='black', ls="--")
        axes[1].set_title("Sparsity Regulation (Target vs Actual)")

        # τ(t)
        axes[2].plot(t_axis, h["tau"], color='orange')
        axes[2].set_title("Threshold Evolution")

        # H(t)
        axes[3].plot(t_axis, h["energy"], color='purple')
        axes[3].set_title("System Entropy Energy")
        
        plt.tight_layout()
        plt.show()

def notify(msg):
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("SMR Status", msg)
    root.destroy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sparse Manifold Simulation")
    parser.add_argument("--silence", type=float, default=0.75, help="Target silence ratio (0-1)")
    args = parser.parse_args()

    sim = ManifoldRegulator(target_silence=args.silence)
    final_tau = sim.run()
    
    notify(f"Simulation Converged\nFinal Threshold: {final_tau:.4f}\nTarget Silence: {args.silence}")
    sim.plot()
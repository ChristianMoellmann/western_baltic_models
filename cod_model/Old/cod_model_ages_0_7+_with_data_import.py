import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# Load ICES Data
# ------------------------------

# File paths (replace with your own if needed)
catch_df = pd.read_csv("wb_catch.csv", sep=";", index_col=0)
mat_df = pd.read_csv("wb_mat.csv", sep=";", index_col=0)
weights_df = pd.read_csv("wb_weights.csv", sep=";", index_col=0)
mortal_df = pd.read_csv("wb_natmort.csv", sep=";", index_col=0)

# Add age0 to catch and reorder to match other data
catch_df.insert(0, 'age0', 0.0)
catch_df = catch_df[mat_df.columns]

# Align all datasets to common years
common_years = sorted(set(catch_df.index) & set(mat_df.index) & set(weights_df.index) & set(mortal_df.index))
catch_df = catch_df.loc[common_years]
mat_df = mat_df.loc[common_years]
weights_df = weights_df.loc[common_years]
mortal_df = mortal_df.loc[common_years]

# ------------------------------
# Set Model Parameters
# ------------------------------

years = catch_df.index.astype(int).values
n_years = len(years)
age_labels = list(catch_df.columns)
n_ages = len(age_labels)

# Natural mortality, weight, maturity loaded per year
# Fishing mortality: fixed at 0.207 for ages 3–5, 0.1 for others
F_vec = np.full(n_ages, 0.1)
F_vec[3:6] = 0.207

# Beverton–Holt recruitment parameters (example values)
a_rec = 50000
b_rec = 30000

# Environmental noise on recruitment
np.random.seed(42)
env = np.random.uniform(0.9, 1.1, n_years)

# ------------------------------
# Estimate Initial Population
# ------------------------------

F_est_init = np.full(n_ages, 0.2)
M_init = mortal_df.iloc[0].values
Z_init = F_est_init + M_init
N0 = np.zeros(n_ages)

for a in range(n_ages):
    if Z_init[a] > 0 and F_est_init[a] > 0:
        N0[a] = catch_df.iloc[0, a] * Z_init[a] / (F_est_init[a] * (1 - np.exp(-Z_init[a])) + 1e-10)

# ------------------------------
# Initialize Outputs
# ------------------------------

N = np.zeros((n_years, n_ages))      # Numbers at age
SSB = np.zeros(n_years)              # Spawning stock biomass
Catch = np.zeros((n_years, n_ages))  # Catch at age
N[0] = N0

# ------------------------------
# Simulation Loop
# ------------------------------

for y in range(n_years - 1):
    W = weights_df.iloc[y].values
    Mat = mat_df.iloc[y].values
    M = mortal_df.iloc[y].values

    # SSB
    SSB[y] = np.sum(N[y] * W * Mat)

    # Recruitment (Beverton-Holt + environment)
    recruits = (a_rec * SSB[y] / (b_rec + SSB[y])) * env[y]
    N[y + 1, 0] = recruits

    # Survival and aging
    for a in range(n_ages - 2):
        Z = F_vec[a] + M[a]
        N[y + 1, a + 1] = N[y, a] * np.exp(-Z)
        Catch[y, a] = (F_vec[a] / Z) * (1 - np.exp(-Z)) * N[y, a]

    # Plus group (age 7+)
    Z_plus = F_vec[-1] + M[-1]
    N[y + 1, -1] += N[y, -2] * np.exp(-Z_plus)
    N[y + 1, -1] += N[y, -1] * np.exp(-Z_plus)
    Catch[y, -1] = (F_vec[-1] / Z_plus) * (1 - np.exp(-Z_plus)) * (N[y, -2] + N[y, -1])

# Final year SSB
W_last = weights_df.iloc[-1].values
Mat_last = mat_df.iloc[-1].values
SSB[-1] = np.sum(N[-1] * W_last * Mat_last)

# ------------------------------
# Plot Results
# ------------------------------

# SSB plot
plt.figure(figsize=(10, 5))
plt.plot(years, SSB, label="Simulated SSB", color='green', marker='o')
plt.axhline(23492, color='red', linestyle='--', label='MSY B_trigger')
plt.title("Simulated Spawning Stock Biomass (SSB)")
plt.xlabel("Year")
plt.ylabel("SSB (tonnes)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Catch plot
total_catch = Catch.sum(axis=1)
plt.figure(figsize=(10, 5))
plt.plot(years, total_catch, label="Simulated Total Catch", color='blue', marker='s')
plt.title("Simulated Total Catch")
plt.xlabel("Year")
plt.ylabel("Catch (numbers)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

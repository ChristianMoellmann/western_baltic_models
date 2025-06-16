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
weights_catch_df = pd.read_csv("wb_weights_catch.csv", sep=";", index_col=0)
mortal_df = pd.read_csv("wb_natmort.csv", sep=";", index_col=0)

# Add age0 to catch and reorder to match other data
if np.shape(catch_df)[1] != np.shape(mat_df)[1]:
    catch_df.insert(0, 'age0', 0.0)

# Align all datasets to common years
common_years = sorted(set(catch_df.index) & set(mat_df.index) & set(weights_df.index) & set(mortal_df.index))
catch_df = catch_df.loc[common_years]
mat_df = mat_df.loc[common_years]
weights_df = weights_df.loc[common_years]
weights_catch_df = weights_catch_df.loc[common_years]
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
F_vec[0] = 0

# Beverton–Holt recruitment parameters (example values)
a_rec_r1 = 2.3081756699
b_rec_r1 = 0.0000184802

a_rec_r0 = 1.274305e+01
b_rec_r0 = 4.935334e-05

# Environmental noise on recruitment
np.random.seed(42)
env = np.random.uniform(0.9, 1.1, n_years)

# ------------------------------
# Set Initial Population
# ------------------------------

N0 = np.array([458813, 49263, 24421, 21912, 4204, 1128, 364, 208])

# ------------------------------
# Initialize Outputs
# ------------------------------

N = np.zeros((n_years, n_ages))      # Numbers at age
SSB = np.zeros(n_years)              # Spawning stock biomass
Catch = np.zeros((n_years, n_ages))  # Catch at age
N[0] = N0

# ------------------------------
# Set recruitment age class

age_r = 0

# ------------------------------

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
    if age_r == 1:
        recruits = (a_rec_r1 * SSB[y] / (1 + b_rec_r1 * SSB[y]))
        recruits = recruits * env[y]
        N[y + 1, 1] = recruits
    elif age_r == 0:
        recruits = (a_rec_r0 * SSB[y] / (1 + b_rec_r0 * SSB[y]))
        recruits = recruits * env[y]
        if N[y, 0] == 0: # in the first year, recruits are already given by N0
            N[y, 0] = recruits
    
    # Back-calculate age-zero fish
    if age_r == 1:
        N[y + 1, 0] = 0
        if y >= 2: # in the first year, recruits are already given by N0
            N[y - 1, 0] = (1 / (np.exp(-(mortal_df.iloc[y-1].values[0] + F_vec[0])))) * N[y, 1]

    # Survival and aging
    if age_r == 1:
        start_age = 1
    elif age_r == 0:
        start_age = 0
        
    for a in np.arange(start_age, n_ages - 2):
        Z = F_vec[a] + M[a]
        N[y + 1, a + 1] = N[y, a] * np.exp(-Z)
        
    for a in np.arange(start_age, n_ages):
        Z = F_vec[a] + M[a]
        Catch[y, a] = (F_vec[a] / Z) * (1 - np.exp(-Z)) * N[y, a]

    # Plus group (age 7+)
    Z_plus = F_vec[-2] + M[-2]
    N[y + 1, -1] += N[y, -2] * np.exp(-Z_plus)
    Z_plus = F_vec[-1] + M[-1]
    N[y + 1, -1] += N[y, -1] * np.exp(-Z_plus)
    
# Final year SSB
W_last = weights_df.iloc[-1].values
Mat_last = mat_df.iloc[-1].values
SSB[-1] = np.sum(N[-1] * W_last * Mat_last)

# # Final year catches
# M = mortal_df.iloc[-1].values
# for a in np.arange(start_age, n_ages):
#     Z = F_vec[a] + M[a]
#     Catch[-1, a] = (F_vec[a] / Z) * (1 - np.exp(-Z)) * N[-1, a]

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

# Catch-in-weight plot
total_catch = np.sum(Catch * np.array(weights_catch_df), axis = 1)
plt.figure(figsize=(10, 5))
plt.plot(years, total_catch, label="Simulated Total Catch in Tonnes", color='blue', marker='s')
plt.title("Simulated Total Catch (tonnes)")
plt.xlabel("Year")
plt.ylabel("Catch (numbers)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()












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
mortal_df = pd.read_csv("wb_natmort_var.csv", sep=";", index_col=0)
numbers_df = pd.read_csv("wb_numbers.csv", sep=";", index_col=0)
fishmort_df = pd.read_csv("wb_fishmort.csv", sep=";", index_col=0)

# Add age0 to catch and reorder to match other data
if np.shape(catch_df)[1] != np.shape(mat_df)[1]:
    catch_df.insert(0, 'age0', 0.0)

# Align all datasets to common years
common_years = sorted(set(catch_df.index) & set(mat_df.index) & set(weights_df.index) & set(mortal_df.index))
common_years = (np.array(common_years)[np.array(common_years) <= 2022]).tolist()
catch_df = catch_df.loc[common_years]
mat_df = mat_df.loc[common_years]
weights_df = weights_df.loc[common_years]
weights_catch_df = weights_catch_df.loc[common_years]
mortal_df = mortal_df.loc[common_years]
numbers_df = numbers_df.loc[common_years]
fishmort_df = fishmort_df.loc[common_years]

# ------------------------------
# Set Model Parameters
# ------------------------------

n_years = 100
age_labels = list(catch_df.columns)
n_ages = len(age_labels)

# Fishing mortality: fixed at 0.207 for ages 3–5, 0.1 for others

# F_vec = np.full(n_ages, 0.1)
# F_vec[3:6] = 0.207
# F_vec[0] = 0

# F_vec[:] = 0

f_set = 0
f_msy_bar = np.array(fishmort_df)[-1,3:6] / np.mean(np.array(fishmort_df)[-1,3:6]) * f_set
f_rel = np.array(fishmort_df)[-1,4] / np.array(fishmort_df)[-1,:]
F_vec = np.concatenate([np.reshape(0, [1,]), f_msy_bar[2]/f_rel[[1,2]], f_msy_bar, f_msy_bar[2]/f_rel[[6,7]]])

# ------------------------------
# Set Initial Population
# ------------------------------

N0 = np.array(numbers_df)[-1,:]

# ------------------------------
# Initialize Outputs
# ------------------------------

N = np.zeros((n_years, n_ages))      # Numbers at age
SSB = np.zeros(n_years)              # Spawning stock biomass
Catch = np.zeros((n_years, n_ages))  # Catch at age
N[0] = N0

W = np.array(weights_df)[-1,:]
Mat = np.array(mat_df)[-1,:]
M = np.array(mortal_df)[-1,:]

# ------------------------------

# ------------------------------
# Simulation Loop
# ------------------------------

for y in range(n_years - 1):
    # SSB
    SSB[y] = np.sum(N[y] * W * Mat)

    N[y + 1, 1] = 68064.14 # 136168.4 for Bmsy
    
    # Back-calculate age-zero fish
    N[y + 1, 0] = 0
    if y >= 2: # in the first year, recruits are already given by N0
        N[y - 1, 0] = (1 / (np.exp(-(M[0] + F_vec[0])))) * N[y, 1]

    # Survival and aging
    start_age = 1
        
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
SSB[-1] = np.sum(N[-1] * W * Mat)

# # Final year catches
# M = mortal_df.iloc[-1].values
# for a in np.arange(start_age, n_ages):
#     Z = F_vec[a] + M[a]
#     Catch[-1, a] = (F_vec[a] / Z) * (1 - np.exp(-Z)) * N[-1, a]

# ------------------------------
# Plot Results
# ------------------------------

# SSB plot
ssb_hist = np.sum(np.array(numbers_df) * np.array(weights_df) * np.array(mat_df), axis = 1)

plt.rcParams.update({'font.size': 20})

plt.figure(figsize=(10, 7))
plt.plot(np.arange(1985,2023), ssb_hist)
plt.plot(2022+np.arange(0,n_years), SSB, label="Simulated SSB", color='green', marker='o')
plt.axhline(23492, color='red', linestyle='--', label='MSY B_trigger')
plt.axhline(23492*2, color='red', linestyle=':', label='B MSY')
plt.axhline(15067, color='red', linestyle=':', label='MSY B_trigger')
plt.xlabel("Year")
plt.ylabel("SSB (tonnes)")
plt.ylim([0,110000])
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(np.arange(1985,2023), numbers_df['age1'])
plt.plot(2022+np.arange(0,n_years), N[:,1], label="Age-1 recruitment (millions)", color='green', marker='o')
plt.xlabel("Year")
plt.ylabel("Age-1 recruitment (millions)")
plt.grid(True)
plt.tight_layout()
plt.show()












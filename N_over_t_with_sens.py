#%%
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
import ipywidgets as widgets
from ipywidgets.embed import embed_minimal_html
import pandas as pd
from init_rcParams import set_mpl_settings
import numpy as np
#%%
import matplotlib.pyplot as plt
try: set_mpl_settings()
except ValueError: pass
import matplotlib as mpl
mpl.rc("figure", dpi=330) 

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# %%
from scipy.integrate import odeint

# Define the growth function
def growth(N, r, K):
	return r * N * (1 - (N / K))

# Define the predation function
def predation(N, A, B):
	return (B * N**2) / (A**2 + N**2)

# Define the total function with time-dependent parameters and extra differential equation for r
def total(y, t, params):
	N, r = y  # Unpack the current values of N and r
	
	# Determine the current parameter values based on time 't'
	current_params = {key: value['base'] for key, value in params.items()}

 
	# Handle any time-dependent adjustments to parameters
	for key, adjustments in params.items():
		for adjustment in adjustments.get('adjustments', []):
			if adjustment['start'] <= t <= adjustment['end']:
				current_params[key] = adjustment['value']
				break  # Use the first matching adjustment found
	
	drdt = current_params['r_g'] * N - current_params['r_d'] * r
	# Calculate the changes due to growth and predation for N
	dNdt = growth(N, r, current_params['K']) - \
		   predation(N, current_params['A'], current_params['B'])
		   
	return [dNdt, drdt]
#%%
# Initial conditions for N and r
N0 = 1  # Example initial population
r0 = 0.1  # Example initial rate of growth

# Parameters including r_g and r_d instead of r
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Assuming total function and initial conditions N0, r0 are defined elsewhere
# params dictionary as you provided
params = {
	'K': {'base': 10},      # Carrying capacity
	'A': {'base': 3},      # Predation parameter A
	'B': {'base': 1},      # Predation parameter B
	'r_g':{'base': 0.004},  # Growth rate of r
	'r_d':{'base': 0.03},  # Decay rate of r
	# Add 'adjustments' for any time-dependent parameters if needed
}
# params = {
# 	'K': {'base': 5},      # Carrying capacity
# 	'A': {'base': 1},      # Predation parameter A
# 	'B': {'base': 1},      # Predation parameter B
# 	'r_g':{'base': 0.04},  # Growth rate of r
# 	'r_d':{'base': 0.03},  # Decay rate of r
# 	# Add 'adjustments' for any time-dependent parameters if needed
# }

# Time points at which to solve the system
t = range(0, 250)

# Solve the system of equations (assuming the 'total' function is defined elsewhere)
solution = odeint(total, [N0, r0], t, args=(params,))

# Extract solutions
N_solution = solution[:, 0]
r_solution = solution[:, 1]

# Create subplots beneath each other
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)  # 2 rows, 1 column, sharing x-axis

# Plot N_solution on the first subplot
ax1.plot(t, N_solution)
# ax1.set_title('N Solution Over Time')
ax1.set_ylabel('N')

# # Vertical dashed lines with labels for the first subplot
# times = [25, 75, 125, 175]
# labels = ['a', 'b', 'c', 'd']
# for time, label in zip(times, labels):
# 	ax1.axvline(x=time, color='#A663CC', linestyle='--')
# 	ax1.text(time, ax1.get_ylim()[1], label, ha='center', va='bottom', fontweight='bold', )

# Plot r_solution on the second subplot
ax2.plot(t, r_solution)
# ax2.set_title('r Solution Over Time')
ax2.set_xlabel('Time')
ax2.set_ylabel('r')

# Vertical dashed lines with labels for the second subplot
# for time, label in zip(times, labels):
# 	ax2.axvline(x=time, color='#A663CC', linestyle='--')
	# No need to add text again, as they share the x-axis

# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plots
plt.show()

#%%
##! INTERESTING: RELAPSE?
# Parameters including r_g and r_d instead of r
# params = {
# 	'K': {'base': 10},  # Carrying capacity
# 	'A': {'base': 1},    # Predation parameter A
# 	'B': {'base': 1},  # Predation parameter B
# 	'r_g':{'base': 0.05},  # Growth rate of r
# 	'r_d':{'base': 0.05}, # Decay rate of r
# 	# Add 'adjustments' for any time-dependent parameters if needed
# }
# # Time points at which to solve the system
# t = range(0, 100)



# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

# Dynamical model function
def dN_dt(N, r, K, A, B):
	return r * N * (1 - N / K) - (B * (N ** 2)) / (A**2 + N ** 2)

# New constant values for A and B
A = 3
B = 1

# Define ranges for r and K
r_values = np.linspace(0.13, 0.8, 100)
K_values = np.linspace(3.5, 20, 100)

# Function to compute attractor roots
def compute_attractor_roots(K, r, A, B):
	attractor_roots = []
	search_intervals = np.linspace(0, K, 100)
	for i in range(len(search_intervals) - 1):
		a = search_intervals[i]
		b = search_intervals[i + 1]
		try:
			root = brentq(dN_dt, a, b, args=(r, K, A, B))
			# Check criteria and add root if it meets conditions
			if dN_dt(root+0.05, r, K, A, B) < 0 and not np.isclose(root, 0, atol=1e-3):
				attractor_roots.append(root)
		except ValueError:
			pass
	return attractor_roots

# Convert the attractor roots into a matrix form for plotting
sols = np.full((len(K_values), len(r_values)), np.nan)
for k_index, K in enumerate(K_values):
	for r_index, r in enumerate(r_values):
		result = compute_attractor_roots(K, r, A, B)
		if len(result) == 1:
			sols[k_index, r_index] = result[0]

# Plotting the results
fig = plt.figure(figsize=(8, 6))
cmap = plt.get_cmap('gold_black')
cmap.set_bad(color='#CCE8CC')
plt.pcolormesh(K_values, r_values, sols.T, shading='auto', cmap=cmap)
plt.title(f'A = {A}, B = {B}')
plt.xlabel('K')
plt.ylabel('r')

# Add a color bar on the right side of the plot
cbar = plt.colorbar(orientation='vertical', pad=0.01, fraction=0.05)
cbar.ax.set_ylabel('N value')
plt.show()

# %%

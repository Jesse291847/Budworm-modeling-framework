import numpy as np
import matplotlib.pyplot as plt
from init_rcParams import set_mpl_settings
import matplotlib as mpl

try: set_mpl_settings()
except ValueError: pass

mpl.rc("figure", dpi=330) 


# Set up the figure with two subplots next to each other
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Set parameters
r_values = [0.3, 0.45, 0.7]
K = 10
dt = 0.01
max_t = 30
time_values = np.arange(0, max_t, dt)

# Appetitive motivation
def growth(r, x, K):
    return r * x * (1 - (x / K))

# Get consumption for different r values
for r in r_values:
    N_values = np.zeros_like(time_values)
    N_values[0] = 0.1
    for i in range(1, len(time_values)):
        N_values[i] = N_values[i-1] + growth(r, N_values[i-1], K) * dt
    axs[0].plot(time_values, N_values, label=f'r={r}')

axs[0].axhline(y=K, color='#A663CC', linestyle='--')
axs[0].text(time_values[1], K-0.5, 'K', color='#A663CC', va='center')
axs[0].set_xlabel('Time (t)')
axs[0].set_ylabel('Consumption (N)')
axs[0].legend()
axs[0].set_title('')

# Set parameters for control
A_values = [0.5, 1, 1.5]
B = 1
p = np.linspace(0, 5, 200)

# Get control for different A values
for A in A_values:
    control = (B * p**2) / (A**2 + p**2)
    axs[1].plot(p, control, label=f'A={A}')

axs[1].axhline(y=B, color='#A663CC', linestyle='--')
axs[1].text(p[1], B-0.05, 'B', color='#A663CC', va='center')
axs[1].set_xlabel('Consumption (N)')
axs[1].set_ylabel('Control')
axs[1].legend()
axs[1].set_title('')

plt.tight_layout()
#plt.savefig("figures/fig2_C.pdf", format="pdf", bbox_inches="tight")
plt.show()

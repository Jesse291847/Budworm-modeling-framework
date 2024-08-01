import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual, FloatSlider
from ipywidgets.embed import embed_minimal_html
import pandas as pd
from init_rcParams import set_mpl_settings
import numpy as np
from functools import cache
#%%
import matplotlib.pyplot as plt
try: set_mpl_settings()
except ValueError: pass
import matplotlib as mpl
import numpy as np
from scipy.optimize import brentq
mpl.rc("figure", dpi=330) 
#%%
# ###### SIGMOID PLOT #########################
import matplotlib.pyplot as plt

#%%
import numpy as np
import matplotlib.pyplot as plt

# Assuming these are constants or given values
B_i_base = 0  # Replace this with actual B_i^base value
beta_i = 1  # Replace this with actual beta_i value
C_i = [1]  # Replace this with actual C_i values
h_B = 2.5  # Replace this with actual halfway point h_B

def B_i(N_j, T_B):
    outcome = 1 / (1 + np.exp((N_j - h_B) / T_B))
    return outcome

# Range of N_j values
N_j_values = np.linspace(0, 5, 500)

# Different values of T_B
T_B_values = [0.1, 0.3, 0.5]

# Plot each T_B
linestyles = ['--', '-', '-.']

# Plot each T_B with a different linestyle
for T_B, ls in zip(T_B_values, linestyles):
    B_i_values = [B_i(N_j, T_B) for N_j in N_j_values]
    plt.plot(N_j_values, B_i_values, label =rf'$T_{{B}}={T_B}$', linestyle=ls)

# Mark the halfway point h_B
plt.scatter([h_B], [B_i(h_B, T_B_values[0])], color='#A663CC', zorder=5, s=100)

# Label the marker with $h_B$
plt.text(h_B+0.4, B_i(h_B, T_B_values[0])+0.03, '$h_B$',
         color='#A663CC', verticalalignment='bottom', horizontalalignment='right')


# Adding labels and title
#plt.axhline(y=0, color='grey', linewidth=2, linestyle='-')

# Setting LaTeX formatted labels and title
plt.xlabel(r'$N_j$')  # LaTeX formatted x-axis label
plt.ylabel(r'$N_j \rightarrow B_i$')

plt.legend()
# plt.grid(True)

# Show the plot
plt.show()
#%%

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

#%%
from scipy.optimize import fsolve 
import scipy.optimize as opt

def get_roots(func, ax, plotBool = True):
    # Get list of mu values
    mu_values = np.linspace(-1, 10, 1000)  # change range as required

    # Initial guess for the roots
    x_initial_guesses = [-1, 0, 0.1,0.3,0.5,0.8,1,1.5,2,3,4, 5,6,7,8]  # change as per your requirement

    # roots = []
    # for x0 in x_initial_guesses:
    #     root = opt.root(func, x0, method = 'broyden1',)
    #     roots.append(root[0])
    
    root_arr = opt.root(func, x_initial_guesses, method = 'broyden1')
    # root_arr = opt.brentq(func, 0, 14)
    mask = np.isclose(root_arr.fun, 0, atol=1e-6) # you can adjust the 
    roots = root_arr.x[mask]
    roots.sort()
    grouped = [roots[0]]
    
    for i in range(1, len(roots)):
        if abs(roots[i] - grouped[-1]) > 0.2:
            grouped.append(roots[i])
    roots = grouped

    # Plotting
    # fig, ax = plt.subplots()

    # Plot function
    # totlist = list(map(func,mu_values))
    # if plotBool: ax.plot(mu_values, totlist, label='tot')
    attractors = []
    # Plot roots
    i,j = 0,0
    for root in roots:
        
        y_val_plus = func(root+0.01)
        y_val = func(root)
        if y_val_plus < 0: 
            if i == 0:
                if plotBool: ax.scatter(root, y_val, color='#1C110A', zorder=5, label='Attractor', s=80)  # Attractor (round points)
                i+= 1
            elif plotBool: ax.scatter(root, y_val, color='#1C110A', zorder=5, s=80)  

            attractors.append(root)
        else:
            if j == 0:
                if plotBool: ax.scatter(root, y_val, facecolors='none', edgecolors='#1C110A', linewidths = 2, zorder=5, label='Repellor', s= 80)  # Repellor (open round)
                j += 1
            elif plotBool: ax.scatter(root, y_val, facecolors='none', edgecolors='#1C110A', linewidths = 2, zorder=5, s= 80)  #


    return attractors

#%%

#%%
#%%

############################################### ?INTERACTIVE #########################
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import interactive, FloatSlider

def budworm(r, K, A, B):
    mu_max = 10
    mu = np.linspace(0,mu_max,100)

    fig, axs = plt.subplots(1)#, figsize = (16,12))
    
    
    _mu = lambda _mu : r * _mu *(1-(_mu/K))
    mulist = list(map(_mu,mu))
    axs.plot(mu, mulist, label = 'Growth')
    axs.set_xlabel('N')
    axs.set_ylabel('dN')

    pred = lambda p: (B*p**2) / (A**2 + p**2) #- C * p
    predlist = list(map(pred,mu))
    axs.plot(mu, predlist, label= 'Control', color = '#880D1E')

    tot = lambda tot : _mu(tot) - pred(tot)  #r*_mu* (1-(_mu/K)) - (B* _mu**2/(A**2+_mu**2))
    totlist = list(map(tot,mu))
    axs.plot(mu, totlist, label = 'Nullcline', color = '#A663CC')
    get_roots(tot, axs)
    
    axs.set_ylim((-0.5,2))
    # axs.set_xlim((-0.5,2))
    axs.axhline(0, linestyle = '--', color = '#1C110A', alpha = 0.5)
    plt.legend()

B_slider = FloatSlider(min=0.25, max=3, step=0.25, value=1.00)    
A_slider = FloatSlider(min=0.25, max=3, step=0.25, value=1.00)   
  
interactive(budworm, r=(0.0,1,0.025), K = (0.0,25,1),  A = A_slider, B = B_slider) #, C = (0.0,0.5,0.01))

#########################################################################
#%%
######################## ? SUBPLOTS ########################
import numpy as np
from matplotlib import pyplot as plt

def budworm(r, K, A, B, r_index = None, axs = None):
    if r_index == None:
        axs = [axs]
        r_index = 0
    mu_max = 10
    mu = np.linspace(0,mu_max,100)

    _mu = lambda _mu : r * _mu *(1-(_mu/K))
    mulist = list(map(_mu,mu))
    
    axs[r_index].plot(mu, mulist, label = 'Growth')
    axs[r_index].set_xlabel('N')
    axs[r_index].set_ylabel('dN')

    pred = lambda p: (B*p**2) / (A**2 + p**2)
    predlist = list(map(pred,mu))
    axs[r_index].plot(mu, predlist, label= 'Control', color = '#880D1E')

    tot = lambda _mu : r*_mu* (1-(_mu/K)) - (B* _mu**2/(A**2+_mu**2))
    totlist = list(map(tot,mu))
    axs[r_index].plot(mu, totlist, label = 'Consumption', color = '#A663CC')
    get_roots(tot, axs[r_index])
    axs[r_index].set_title(f'r = {r}')
    axs[r_index].set_title(f'{abcd[r_index]}', fontweight = 'bold')



    axs[r_index].set_ylim((-0.5,1.3))
    # axs[r_index].set_xlim((-0.5,5))

    axs[r_index].axhline(0, linestyle = '--', color = '#1C110A', alpha = 0.5)
    # if r_index == 0: axs[r_index].legend(ncol=2)


# Create an array of subplots
############# ?SUBPLOTS AND ANIMATION #######################
# Define the values for parameters
K = 10
A = 1
B = 1.00
r = 0.8
param = np.linspace(0.4,6,4)
abcd = ['a','b','c','d']

# PLOT SLOW DECLINE TO HEAVY DRINKING: K param = np.linspace(0.4,6,4), A = 1, B = 1.00, r = 0.8

#%%
fig, axs = plt.subplots(4, figsize=(11,14))
# Iterate over the r_values, and create a plot for each
for r_index, param_i in enumerate(param):
    budworm(r, param_i, A, B, r_index, axs)
    
# plt.subplots_adjust(hspace=-0.1)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, loc = 'upper right', ncol = 2)

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def growth_control_nullcline(r, B, A, K, initial_N, tmax = 20):
    # Define the differential equations
    def _mu(mu):
        return r * mu * (1 - (mu / K))
    
    def pred(p):
        return (B * p**2) / (A**2 + p**2)
    
    def tot(tot, t):
        return _mu(tot) - pred(tot)
    
    # Time points to evaluate
    t = np.linspace(0, tmax, 200)  # You can adjust these time boundaries and number of points
    
    # Solve Differential Equation for each initial condition
    for i0 in initial_N:
        sol = odeint(tot, i0, t)
        plt.plot(t, sol, label=f'Initial N={i0}')
    
    # Plotting
    plt.title('Total Population Over Time')
    plt.xlabel('Time')
    plt.ylabel('N (Population)')
    # plt.legend()
    plt.grid(True)
    plt.show()
#%%
# Example usage:
growth_control_nullcline(r=0.8, B=1, A=1, K=10, initial_N=[0,2,4,6,8,10,12,14])
# K = 10
# A = 1
# B = 1.00
# r = 0.8















# #%%
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib.animation import PillowWriter
# fig, axs = plt.subplots(1)
# def update(frame_number, param):
#     axs.clear()  # Clear the current figure
#     budworm(r, param[frame_number], A, B)  # Plot with the new parameter value
# ani = animation.FuncAnimation(plt.gcf(), update, frames=len(param), fargs=(param,), interval=1000)
# ani.save('../plots/animation.gif', writer=PillowWriter(fps=2))


# #%%



#%%
## GROWTH SYSTEM
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Define parameters
r_values = [0.3, 0.45, 0.7]  # List of different r values
K = 10
dt = 0.01  # time step
max_t = 30  # maximum time

# Initialize arrays to store time values
time_values = np.arange(0, max_t, dt)

# Define function _mu
def _mu(r, x, K):
    return r * x * (1 - (x / K))

# Create figure and axes
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot for each value of r
for r in r_values:
    # Initialize N_values array for this r
    N_values = np.zeros(time_values.shape)
    
    # Set initial condition
    N_values[0] = 0.1  # adjust as needed
    
    # Implement Euler's method
    for i in range(1, len(time_values)):
        N_values[i] = N_values[i-1] + _mu(r, N_values[i-1], K) * dt
    
    # Calculate mu for a range of N values
    N_range = np.linspace(0, K, len(time_values))
    mu_list = [_mu(r, x, K) for x in N_range]
    
    # Plot growth rate versus N
    axs[0].plot(N_range, mu_list, label=f'r={r}')
    
    # Plot Cumulated Growth
    axs[1].plot(time_values, N_values, label=f'r={r}')

# Add dashed line and 'K' text to right plot (axs[1])
axs[1].axhline(y=K, color = '#A663CC', linestyle='--')
axs[1].text(time_values[1], K-0.5, 'K', color = '#A663CC', verticalalignment='center')

# Set labels for left plot (axs[0])
axs[0].set_xlabel('Consumption (N)')
axs[0].set_ylabel('Increase in Consumption (dN)')
axs[0].legend()

# Set labels for right plot (axs[1])
axs[1].set_xlabel('Time (t)')
axs[1].set_ylabel('Consumption (N)')
axs[1].legend()

# Adjust layout
plt.tight_layout()

# Display plot
plt.show()


# ####
# r = 0.07,
# r = 0.40
# r = 0.56






#%%%%
#%%
######## PREDATOR SYSTEM #######
#? dC over C, and C over time
import matplotlib.pyplot as plt
import numpy as np

# Define parameters
B_values = [1]  # List containing B value
A_values = [0.5, 1, 1.5]  # List of different A values

# Define p range
p = np.linspace(0, 5, 200)
dt = 0.01  # time step
max_t = 30  # maximum time

# Create figure and axes
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
# plt.subplots_adjust(hspace=0.5)

# Plot for each value of A and B
for A in A_values:
    # Define function pred for this A
    pred = lambda p: (B_values[0]*p**2) / (A**2 + p**2)
    
    # Apply function to p values
    plist = list(map(pred, p))
    
    # Plot results on the left subplot
    axs[0].plot(p, plist, label=f'Growth A={A}')

    # Initialize arrays to store time and P values
    time_values = np.arange(0, max_t, dt)
    P_values = np.zeros(time_values.shape)
    
    # Set initial condition
    P_values[0] = 0.1  # adjust as needed
    
    # Implement Euler's method
    for i in range(1, len(time_values)):
        P_values[i] = P_values[i-1] + pred(P_values[i-1]) * dt
    
    # Plot results on the right subplot
    axs[1].plot(time_values, P_values, label=f'A={A}')

# Add dashed line and 'B' text to left plot (axs[0])
axs[0].axhline(y=B_values[0], color = '#A663CC', linestyle='--')
axs[0].text(p[1], B_values[0]-0.05, 'B', color = '#A663CC', verticalalignment='center')

# Set labels for left plot (axs[0])
axs[0].set_ylabel('Change of control (dC)')
axs[0].set_xlabel('Consumption (N)')
axs[0].legend()

# Set labels for right plot (axs[1])
axs[1].set_xlabel('Time (t)')
axs[1].set_ylabel('Control')
axs[1].legend()

plt.tight_layout()

# Display plot
plt.show()

#%%%%
##### GROWTH VS CONTROL
# ? Quiver plot
import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 0.5
K = 10
A = 1
B = 1


# System equations
dgrowth_dt = lambda N: r * N * (1-N/K)
dcontrol_dt = lambda N: (B*N**2) / (A **2 + N**2)

# Grid for plotting
N = np.linspace(0, 12, 13)
N1, N2 = np.meshgrid(N, N)

# Vector fields
U = dgrowth_dt(N1)
V = dcontrol_dt(N2)

# Quiver plot
plt.quiver(N1, N2, V, U, color='r')
plt.ylabel('Growth over time')
plt.xlabel('Control over time')
plt.title('Quiver Plot of the System')
plt.grid(True)
plt.show()













#%%
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
import numpy as np
from scipy.optimize import brentq
mpl.rc("figure", dpi=330) 
#%%
# ###### TRIPLE PLOT #########################
import matplotlib.pyplot as plt

A, B = 1, 1

def dN_dt(N, r, K):
	return r * N * (1 - N / K) - (B * (N ** 2)) / (A**2 + N ** 2)

r_values = np.linspace(0.13, 0.8, 300)
K_values = np.linspace(3.5, 12, 300)
sols = np.full((len(K_values), len(r_values)), np.nan)

for k_index, K in enumerate(K_values):
	for r_index, r in enumerate(r_values):
		attractor_roots = []
		search_intervals = np.linspace(0, K, 100)

		for i in range(len(search_intervals) - 1):
			a = search_intervals[i]
			b = search_intervals[i + 1]
			try:
				root = brentq(dN_dt, a, b, args=(r, K))
				if dN_dt(root+0.05, r,K) < 0 and not np.isclose(root, 0, atol=1e-3) \
						and not any(np.isclose(root, r,rtol = 1e-3) for r in attractor_roots):
					attractor_roots.append(root)
			except ValueError:
				pass

		if len(attractor_roots) == 1:
			sols[k_index, r_index] = attractor_roots[0]

X, Y = np.meshgrid(K_values, r_values)
#%%
def plot_bifurcation(K_values, r_values, sols, ax):
	X, Y = np.meshgrid(K_values, r_values)
	cmap = plt.get_cmap('black_gold')
	cmap.set_bad(color='#CCE8CC')
	mesh = ax.pcolormesh(X, Y, sols.T, shading='auto', cmap=cmap)

	arrow_properties = dict(
		facecolor="#880D1E",
		edgecolor="#880D1E",
		head_width=0.035,
		head_length=0.2,
		shape='full',
		lw=2.3,
		zorder=5,
		overhang=0.2,
		length_includes_head=True
	)

	# Draw horizontal arrow
	ax.arrow(4.5, 0.62, 3, 0, **arrow_properties)
	ax.arrow(7.5, 0.62, -3, 0, **arrow_properties)

	arrow_properties = dict(
		facecolor="#A663CC",
		edgecolor="#A663CC",
		head_width=0.2,
		head_length=0.04,
		shape='full',
		lw=2.5,
		zorder=5,
		overhang=0.2,
		length_includes_head=True
	)

	# Draw vertical arrow
	ax.arrow(10, 0.23, 0, 0.54, **arrow_properties)
	ax.arrow(10, 0.77, 0, -0.54, **arrow_properties)

	ax.set_xlabel(r'$K/A$')
	ax.set_ylabel(r'$\frac{Ar}{B}$', rotation=0, labelpad=15)

	# Plot points and labels
	# points_r_values = [0.3, 0.5, 0.7]
	# labels = ['c', 'b', 'a']
	points_r_values = [0.3, 0.385, 0.56, 0.7]
	labels = ['d', 'c', 'b', 'a']
	for r_value, label in zip(points_r_values, labels):
		ax.plot(10, r_value, 'o', markersize='8', color='#1C110A', zorder=6)
		ax.text(9.6, r_value - 0.02, label, color='#1C110A', fontsize=22, zorder=6)

	return mesh

def plot_line1(r_values, K, dN_dt, ax):
	sols_low = np.zeros(len(r_values))
	global sols_high 
	sols_high = np.zeros(len(r_values))

	# Perform the root finding
	for r_index, r in enumerate(r_values):
		attractor_roots = []

		# Use a finer search grid
		search_intervals = np.linspace(0, K, 100)

		# Search for roots in each small interval
		for i in range(len(search_intervals) - 1):
			a = search_intervals[i]
			b = search_intervals[i + 1]
			try:
				root = brentq(dN_dt, a, b, args=(r, K))
				# Check if the root is an attractor and not already found
				if dN_dt(root+0.05, r,K) < 0 and not np.isclose(root, 0, atol=1e-3) \
						and not any(np.isclose(root, r,rtol = 1e-3) for r in attractor_roots):
					attractor_roots.append(root)
			except ValueError:
				pass  # No root in this interval
		sols_low[r_index] = np.min(attractor_roots)
		sols_high[r_index] = np.max(attractor_roots)	

	    # Find max of sols_low and min of sols_high
	sols_low_max_index = np.argmax(sols_low[sols_low < 3])
	sols_high_min_index = np.argwhere(r_values == np.min(r_values[sols_high > 3]))

	# Plot the lines
	ax.plot(r_values[sols_low < 3], sols_low[sols_low < 3], '#A663CC', linewidth=3)
	ax.plot(r_values[sols_high > 3], sols_high[sols_high > 3], '#A663CC', linestyle='-', linewidth=3)


	# Set labels
	ax.set_xlabel(r'$Ar/B$')
	ax.set_ylabel('N')

	## ADD B MARKER
	ax.plot(r_values[sols_low_max_index], sols_low[sols_low_max_index], 'o', color='#1C110A', markersize='5') 
	# Blue marker at the lowest point of sols_high
	ax.plot(r_values[sols_high_min_index], sols_high[sols_high_min_index], 'o', color='#1C110A', markersize='5')  
	ax.text(r_values[sols_low_max_index], sols_low[sols_low_max_index], 'b', color='#1C110A',
			fontsize=22, verticalalignment='bottom', horizontalalignment='right')
	ax.text(r_values[sols_high_min_index], sols_high[sols_high_min_index], 'c', color='#1C110A',
			fontsize=22, verticalalignment='top', horizontalalignment='right')

	#### ADD A MARKER ####
	ax.text(r_values[np.abs(r_values-0.67).argmin()], sols_low[np.abs(r_values-0.67).argmin()], 'a', color='#1C110A', fontsize=22, verticalalignment='bottom', horizontalalignment='right')
	ax.plot(r_values[np.abs(r_values-0.67).argmin()], sols_low[np.abs(r_values-0.67).argmin()], 'o', color='#1C110A', markersize='5')  

	#### ADD D MARKER ####
	ax.text(r_values[np.abs(r_values-0.25).argmin()], sols_low[np.abs(r_values-0.25).argmin()], 'd', color='#1C110A', fontsize=22, verticalalignment='bottom', horizontalalignment='right')
	ax.plot(r_values[np.abs(r_values-0.25).argmin()], sols_low[np.abs(r_values-0.25).argmin()], 'o', color='#1C110A', markersize='5')  
 
	###### 
	ax.text(0.387, 0.48, 'c', color='#1C110A', fontsize=22, verticalalignment='bottom', horizontalalignment='right')
	ax.plot(0.387, 0.48,  'o', color='#1C110A', markersize='5')  

	ax.text(0.55, 7.65, 'b', color='#1C110A', fontsize=22, verticalalignment='bottom', horizontalalignment='right')
	ax.plot(0.55, 7.65, 'o', color='#1C110A', markersize='5')  
 
	ax.set_xlabel(r'$Ar/B$')
	ax.set_ylabel('N')

def plot_line2(K_values, dN_dt, r, ax):

	# Function to find attractors for given r and K
	def find_attractors(r, K):
		attractor_roots = []
		search_intervals = np.linspace(0, K, 100)
		for a, b in zip(search_intervals[:-1], search_intervals[1:]):
			try:
				root = brentq(dN_dt, a, b, args=(r, K))
				if dN_dt(root + 0.05, r, K) < 0 and not any(np.isclose(root, attractor_roots, atol=1e-3)):
					attractor_roots.append(root)
			except ValueError:
				pass
		return [np.min(attractor_roots), np.max(attractor_roots)] if attractor_roots else [np.nan, np.nan]

	# Fixed value for r and set the range for K
	r = 0.61
	K_values = np.linspace(4, 7.4, 500)
	sols_low = []
	sols_high = []

	# Perform the root finding for each K value
	for K in K_values:

		low, high = find_attractors(r, K)
		sols_low.append(low)
		sols_high.append(high)

	# Convert lists to NumPy arrays for indexing
	sols_low = np.array(sols_low)
	sols_high = np.array(sols_high)
	# ax.plot(K_values[sols_low < 2], sols_low[sols_low < 2], '#1C110A', linewidth=3)
	# ax.plot(K_values[sols_high > 2], sols_high[sols_high > 2], '#880D1E', linestyle='-.', linewidth=3)
	ax.plot(K_values[sols_low < 2], sols_low[sols_low < 2], '#880D1E', linewidth=3)
	ax.plot(K_values[sols_high > 2], sols_high[sols_high > 2], '#880D1E', linewidth=3)
	ax.set_xlabel(r'$K/A$')
	# ax.set_xlabel(r'$\dfrac{K}{A}$')
	ax.set_ylabel('N')

fig = plt.figure(figsize=(12, 12))
# Create a GridSpec with 3 rows and 2 columns,
# and set the ratio such that the first row takes up half the height of the figure
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.4)

# Bifurcation plot at the top spanning both columns
ax_bifurcation = plt.subplot(gs[0, :])
mesh = plot_bifurcation(K_values, r_values, sols, ax_bifurcation)
# Create colorbar
cbar = fig.colorbar(mesh, ax=ax_bifurcation, pad=0.01, location='right', fraction=0.05)
# cbar.ax.set_xlim(0, 0.5)
K = 10
# Line plots at the bottom left
ax_line1 = plt.subplot(gs[1, 1])
plot_line1(r_values, K, dN_dt, ax_line1)

r = 0.62
# Line plots at the bottom right
ax_line2 = plt.subplot(gs[1, 0])
plot_line2(K_values, dN_dt, r, ax_line2)

plt.tight_layout()
plt.show()
#%%

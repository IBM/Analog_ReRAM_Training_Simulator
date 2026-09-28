
# Auxiliary functions to plot data extracted from SPICE outputs

# Import
import matplotlib.pyplot as plt
from pathlib import Path
import pandas
import numpy
from src.spice_data_analyzer import SimulationDataAnalyzer
import dominate
import plotly.express as px
import seaborn as sns
import dominate.tags as tags
import os
from matplotlib.colors import LinearSegmentedColormap


# Colorbar settings
colors = ["#08306B", "#FFFFFF"]
color_map = LinearSegmentedColormap.from_list("color_map", colors, N=256)


# Plot conductance map during reading phase
def plot_conductance_map(filtered_resistances: pandas.DataFrame, size: int) -> None:
    conductance_map = numpy.zeros((size, size))
    init_g_map = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            R = filtered_resistances[(filtered_resistances['n_row'] == i) & (filtered_resistances['n_col'] == j)]['R'].values
            G = 1e6 / R[-1]
            conductance_map[i, j] = G
    plt.figure(figsize=(10, 6))
    annot_data = numpy.array([["{:.0f}".format(val) for val in row] for row in conductance_map])
    ax = sns.heatmap(conductance_map, annot=annot_data, cmap=color_map, fmt="",
                    cbar_kws={'label': 'Conductance (uS)'}, annot_kws={"size": 16},
                    vmin=20, vmax=90, linewidths=1, linecolor='black')
    ax.collections[0].colorbar.outline.set_edgecolor('black')
    ax.collections[0].colorbar.outline.set_linewidth(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    plt.title('Conductance Map')
    plt.xlabel('Columns (SLs - nFET S)')
    plt.ylabel('Rows (BLs and WLs - ReRAM TE and nFET G)')
#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
import pandas
from pathlib import Path
import plotly.express as px
import warnings
import dominate
import dominate.tags as tags
import os
import numpy


# Get unit cell current from sense resistor
def compute_current_through(data:pandas.DataFrame, path_to_ammeter:str, R_sense:float):
	return (data[f'{path_to_ammeter}/plus)']-data[f'{path_to_ammeter}/minus)'])/R_sense


# Post-process SPICE output from crosspoint array circuit
class SimulationDataAnalyzer:
    #Initialization
	def __init__(self, path_to_simulation_data:Path,
						t_read:float =10e-6, t_write: float=500e-9,
						array_size:int=10):
		self.array_size = 10
		self.path_to_file= path_to_simulation_data
		self.t_read = t_read
		self.t_write = t_write

	# Get unit cell current from sense resistor
	def compute_current_through(self, path_to_ammeter:str, R_sense:float=.1):
		return compute_current_through(data=self.data, path_to_ammeter=path_to_ammeter, R_sense=R_sense)

	# Retrieve timestamps while performing a reading operation
	def reading_time(self, path_to_file: str):
		with open(path_to_file, 'r') as ifile:
			lines = ifile.readlines()
		reading_times = []
		for i in range(len(lines)):
			if lines[i].startswith('* read'):
				reading_times.append(float(lines[i+2].split()[0]))
		return reading_times
	
	# Retrieve electrical values of each cell while performing a reading operation
	def read_data(self):
		with open(self.path_to_file, 'r') as ifile:
			cols = ifile.readline()
		cols = cols.split()
		cols_to_read = [
			i for i,col in enumerate(cols) if
			i == 0 or # This is the time. For some unknown reason it is exported twice, in this way we only read it once.
			#col[1:5] == "1t1r"  # This represents all the voltages in the nodes that were given an explicit name.
			col[0:2] == "v("  # This represents all the voltages in the nodes that were given an explicit name.
		]
		self.data = pandas.read_csv(self.path_to_file, sep=r'\s+', usecols=cols_to_read)

	# Retrieve filtered timestamps while performing a reading operation (make sure is within the actual reading pulse)
	def sample_read_resistance(self, dataframe:pandas.DataFrame):
		read_cycle = self.reading_time(path_to_file="waveforms/col1.txt")
		df = []
		for t in read_cycle:
			new = dataframe[(dataframe["time"] >= (t+(0.3*self.t_read))) & (dataframe["time"] <= (t+(0.7*self.t_read)))]
			df.append(new.groupby(['n_row', 'n_col']).mean())
		new_df = pandas.concat(df, axis=0)
		return new_df.reset_index()

	# Register the switching events
	def register_switching(self, filtered_resistances: pandas.DataFrame, n_rows:int=5, n_cols:int=5):
		switching_events = []
		total =0
		for i in range(n_rows):
			for j in range(n_cols):
				R = filtered_resistances[(filtered_resistances['n_row'] == i) & (filtered_resistances['n_col'] == j)]['R'].values
				G = 1e6/R
				for k in range(1, len(G)):
					if abs(G[k] - G[k-1]) > 2:
						total += 1
		return total

	# Generate filtered resistance map: computation of the resistance during read operation (voltage over current), and averaging over all resistance values filtered to be within the actual read pulse
	def generate_matrix_data(self):
		self.read_data()
		self.data.set_index('time', inplace=True)
		matrix_data = {}
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			for n_row in range(self.array_size):
				for n_col in range(self.array_size):
					self.data[f'I({n_row},{n_col})'] = self.compute_current_through(f'v(/1t1r_10x10/r{n_row}/c{n_col}/ammeter_drain')
					self.data[f'V({n_row},{n_col})'] = self.data[f'v(/1t1r_10x10/r{n_row}/c{n_col}/top_electrode_node)'] - self.data[f'v(/1t1r_10x10/r{n_row}/c{n_col}/source_node)']
					self.data[f'R({n_row},{n_col})'] = self.data[f'V({n_row},{n_col})']/self.data[f'I({n_row},{n_col})']
			for variable in {'R','V','I'}:
				matrix_data[variable] = []
				for n_row in range(self.array_size):
					for n_col in range(self.array_size):
						df = self.data[[f'{variable}({n_row},{n_col})']]
						df['n_row'] = n_row
						df['n_col'] = n_col
						df.rename(columns={f'{variable}({n_row},{n_col})': variable}, inplace=True)
						df.set_index(['n_row','n_col'], append=True, inplace=True)
						matrix_data[variable].append(df)
				matrix_data[variable] = pandas.concat(matrix_data[variable])
		filtered_resistances = self.sample_read_resistance(dataframe=matrix_data["R"].reset_index(drop=False))
		return filtered_resistances

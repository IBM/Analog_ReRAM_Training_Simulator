#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
# Auxiliary functions to compute bit streams for parallel weight update schemes as a result of x and δ vectors

from pathlib import Path
import numpy
from dataclasses import dataclass


# Generate a piece-wise linear sequence of time and voltage values to reproduce a pulse with finite rise time.
def generate_pwl_pulse(V_pulse:float, V_standby:float, t_0:float, t_rise:float, t_on:float, t_fall:float, t_end:float)->(tuple,tuple):
	"""
	Arguments
	---------
	V_pulse:
		Voltage value when the pulse is active
	V_standby:
		Voltage value when the pulse is not active
	t_0:
		Time at which the pulse starts
	t_rise:
		Rise time
	t_on:
		The time during which to maintain `V_pulse`
	t_fall:
		Fall time

	Returns
	-------
	times:
		The four time values (float) for the pulse.
	voltages:
		The four voltage values (float) for the pulse.
	"""
	times = [
		t_0,
		t_0+t_rise,
		t_0+t_rise+t_on,
		t_0+t_rise+t_on+t_fall,
		t_0+t_rise+t_on+t_fall+t_end,
	]
	voltages = [
		V_standby,
		V_pulse,
		V_pulse,
		V_standby,
		V_standby,
	]
	return times, voltages


# Waveform generator for a single unit cell, writing x3 .txt files (x1 BL, x1 SL, x1 WL)
class PulseCreator:
	def __init__(self, t_rise:float, t_write:float, t_read:float, t_end:float,V_pot:float, V_dep:float, V_read:float, t_separation:float, V_standby:float):
		self.t_rise = t_rise
		self.t_write = t_write
		self.t_read = t_read
		self.V_pot = V_pot
		self.V_dep = V_dep
		self.V_read = V_read
		self.V_standby = V_standby
		self.t_separation = t_separation
		self.t_end = t_end
		self.points = ['* time voltage',(0,0.0)] # Initialize.

	def append_pot(self, V_pot:float=None, V_standby:float=None):
		times, voltages = generate_pwl_pulse(
			t_0 = self.points[-1][0] + self.t_separation,
			t_on = self.t_write,
			V_pulse = self.V_pot if V_pot is None else V_pot,
			t_rise = self.t_rise,
			V_standby = self.V_standby if V_standby is None else V_standby,
			t_fall = self.t_rise,
			t_end = self.t_end,
		)
		points = ['* pot_write'] + [(t,V) for t,V in zip(times,voltages)]
		self.points.extend(points)

	def append_dep(self, V_dep:float=None, V_standby: float=None):
		times, voltages = generate_pwl_pulse(
			t_0 = self.points[-1][0] + self.t_separation,
			t_on = self.t_write,
			V_pulse = self.V_dep if V_dep is None else V_dep,
			t_rise = self.t_rise,
			V_standby = self.V_standby if V_standby is None else V_standby,
			t_fall = self.t_rise,
			t_end = self.t_end,
		)
		points = ['* dep_write'] + [(t,V) for t,V in zip(times,voltages)]
		self.points.extend(points)

	def append_read(self, V_read:float=None, V_standby: float=None):
		times, voltages = generate_pwl_pulse(
			t_0 = self.points[-1][0] + self.t_separation,
			t_on = self.t_read,
			V_pulse = self.V_read if V_read is None else V_read,
			t_rise = self.t_rise,
			#V_standby = self.V_standby,
			V_standby=self.V_standby if V_standby is None else V_standby,
			t_fall = self.t_rise,
			t_end = self.t_end,
		)
		points = ['* read'] + [(t,V) for t,V in zip(times,voltages)]
		self.points.extend(points)

	def append_point(self, Delta_t:float, V:float):
		points = ['* point'] + [(self.points[-1][0] + Delta_t,V)]
		self.points.extend(points)
	def append_bias(self, V_gate:float=None, operation:str='* read'):
		t_on = self.t_write
		if operation=="* read":
			t_on = self.t_read
		times, voltages = generate_pwl_pulse(
			t_0 = self.points[-1][0] + self.t_separation,
			t_on = t_on,
			V_pulse = self.V_dep if V_gate is None else V_gate,
			t_rise = self.t_rise,
			V_standby = self.V_standby,
			t_fall = self.t_rise,
		)
		points = [operation] + [(t,V) for t,V in zip(times,voltages)]
		self.points.extend(points)

	def to_file(self, path_to_file:Path):
		# This point is appended because otherwise ngspice brings the voltage to 0 right after the last point.
		self.append_point(
			Delta_t = 1e99,
			V = self.points[-1][1],
		)
		with open(path_to_file, 'w') as ofile:
			for data in self.points:
				if isinstance(data, str):
					print(data, file=ofile)
				elif isinstance(data, tuple) and len(data)==2:
					print(f'{data[0]:.6e} {data[1]:.6e}', file=ofile)
				else:
					raise RuntimeError('Something is wrong.')


# Waveform generator orchestrating the entire crosspoint array, writing as many .txt files as the array lines
@dataclass
class ReRAMMatrixPulseGenerator:
	n_rows: int
	n_cols: int
	t_rise: float
	t_write: float
	t_read: float
	t_rise_gate: float
	t_wait: float
	V_gates_pot: float
	V_gates_dep: float
	V_row_pot: float
	V_row_dep: float
	V_row_read: float
	V_col_pot: float
	V_col_dep: float
	V_col_read: float
	V_standby_dep: float
	V_standby_pot: float
	Gate_Coincidence: bool = False
	two_phases: bool = True

	def pulse_creator(self):
		self._rows = [PulseCreator(
			t_rise = self.t_rise,
			t_write = self.t_write,
			t_read = self.t_read,
			t_separation = 2*self.t_wait + self.t_rise_gate,
			V_pot = self.V_row_pot,
			V_dep = self.V_row_dep,
			V_read = self.V_row_read,
			V_standby = 0,
			t_end = self.t_wait + self.t_rise_gate,
		) for _ in range(self.n_rows)]
		self._cols = [PulseCreator(
			t_rise = self.t_rise,
			t_write = self.t_write,
			t_read = self.t_read,
			t_separation = (2*self.t_wait) + self.t_rise_gate,
			V_pot = self.V_col_pot,
			V_dep = self.V_col_dep,
			V_read = self.V_col_read,
			V_standby = self.V_standby_dep,
			t_end = self.t_wait + self.t_rise_gate,
		) for _ in range(self.n_cols)]
		self._gates = [PulseCreator(
			t_separation = self.t_wait,
			t_rise = self.t_rise_gate,
			t_write = 2*self.t_wait + 2* self.t_rise + self.t_write,
			t_read =  2*self.t_wait + 2* self.t_rise + self.t_read,
			V_pot = self.V_gates_pot,
			V_dep = self.V_gates_dep,
			V_read = self.V_gates_dep,
			V_standby = 0,
			t_end = 0,
		) for _ in range(self.n_rows)]

	def set_waveform_folder(self, path: Path):
		self.path_to_directory = path

	def append_pot(self, n_row:int, n_col:int):
		for n,gate in enumerate(self._gates):
			if self.Gate_Coincidence:
				gate.append_pot(V_pot=self.V_gates_pot if n==n_row else 0, V_standby=0)
			else:
				gate.append_pot(V_pot=self.V_gates_pot, V_standby=0)
			#gate.append_bias(V_gate=None, operation='* pot_write')
			#gate.append_pot(V_pot=0 if n!=n_row else None)
		for n,row in enumerate(self._rows):
			if self.Gate_Coincidence:
				row.append_pot(V_pot=self.V_row_pot, V_standby = self.V_standby_pot)
			else:
				row.append_pot(V_pot=self.V_row_pot if n==n_row else self.V_standby_pot, V_standby=self.V_standby_pot)
		for n,col in enumerate(self._cols):
			col.append_pot(V_pot=self.V_standby_pot if n!=n_col else self.V_col_pot, V_standby=self.V_standby_pot)

	def append_all_pot(self):
		for n,gate in enumerate(self._gates):
			gate.append_pot(V_pot=None, V_standby=0)
		for n,row in enumerate(self._rows):
			row.append_pot(V_pot=None, V_standby=self.V_standby_pot)
		for n,col in enumerate(self._cols):
			col.append_pot(V_pot=None, V_standby=self.V_standby_pot)

	def append_all_dep(self):
		for n,gate in enumerate(self._gates):
			gate.append_dep(V_dep=None)
		for n,row in enumerate(self._rows):
			row.append_dep(V_dep=None)
		for n,col in enumerate(self._cols):
			col.append_dep(V_dep=None)

	def append_dep(self, n_row:int, n_col:int):
		for n,gate in enumerate(self._gates):
			if self.Gate_Coincidence:
				gate.append_dep(V_dep=self.V_gates_dep if n==n_row else 0, V_standby=0)
			else:
				gate.append_dep(V_dep=self.V_gates_dep, V_standby=0)
		for n,row in enumerate(self._rows):
			if self.Gate_Coincidence:
				row.append_dep(V_dep=self.V_row_dep, V_standby = self.V_standby_dep)
			else:
				row.append_dep(V_dep=self.V_row_dep if n==n_row else self.V_standby_dep, V_standby=self.V_standby_dep)
		
		for n,col in enumerate(self._cols):
			col.append_dep(V_dep=self.V_standby_dep if n!=n_col else self.V_col_dep, V_standby=self.V_standby_dep)

	def append_read_whole_matrix(self):
		for gate in self._gates:
			#gate.append_bias(V_gate=None, operation='* read')
			#gate.append_read(V_standby=self.V_gates_dep)
			gate.append_read()
		for row in self._rows:
			row.append_read(V_standby=0)
		for col in self._cols:
			col.append_read(V_standby=0)

	def to_files(self):
		self.path_to_directory.mkdir(exist_ok=True)
		if self.path_to_directory.is_dir():
			for p in self.path_to_directory.iterdir():
				p.unlink()

		for i,row in enumerate(self._rows):
			row.to_file(self.path_to_directory/f'row{i}.txt')
		for i,col in enumerate(self._cols):
			col.to_file(self.path_to_directory/f'col{i}.txt')
		for i,gate in enumerate(self._gates):
			gate.to_file(self.path_to_directory/f'gates{i}.txt')

	def generate_parallel_pot(self, x:numpy.ndarray, err:numpy.ndarray):
		for n, n_row in enumerate(self._rows):
			if self.Gate_Coincidence:
				n_row.append_pot(V_pot=self.V_row_pot, V_standby=self.V_standby_pot)
				continue
			else:
				if x[n] != 0:
					n_row.append_pot(V_pot=self.V_row_pot, V_standby=self.V_standby_pot)
				else:
					n_row.append_pot(V_pot = self.V_standby_pot, V_standby=self.V_standby_pot)
		for n, n_col in enumerate(self._cols):
			if err[n] != 0:
				n_col.append_pot(V_pot=self.V_col_pot, V_standby=self.V_standby_pot)
			else:
				n_col.append_pot(V_pot = self.V_standby_pot, V_standby=self.V_standby_pot)
		for n, n_gate in enumerate(self._gates):
			if  not self.Gate_Coincidence:
				n_gate.append_pot(V_pot=self.V_gates_pot, V_standby=self.V_gates_pot)
				continue
			if x[n] != 0:
				n_gate.append_pot(V_pot=self.V_gates_pot, V_standby=0)
			else:
				n_gate.append_pot(V_pot=0, V_standby=0)

	def generate_parallel_dep(self, x:numpy.ndarray, err:numpy.ndarray):
		for n, n_row in enumerate(self._rows):
			if self.Gate_Coincidence:
				n_row.append_dep(V_dep=self.V_row_dep, V_standby=self.V_standby_dep)
				continue
			else:
				if x[n] != 0:
					n_row.append_dep(V_dep=None)
				else:
					n_row.append_dep(V_dep = self.V_standby_dep)
		for n, n_col in enumerate(self._cols):
			if err[n] != 0:
				n_col.append_dep(V_dep=self.V_col_dep, V_standby=self.V_standby_dep)
			else:
				n_col.append_dep(V_dep = self.V_standby_dep, V_standby=self.V_standby_dep)
		for n, n_gate in enumerate(self._gates):
			if not self.Gate_Coincidence:
				n_gate.append_dep(V_dep=self.V_gates_dep, V_standby=self.V_gates_dep)
				continue
			if x[n] != 0:
				n_gate.append_dep(V_dep=self.V_gates_dep, V_standby=0)
			else:
				n_gate.append_dep(V_dep=0, V_standby=0)
    
	def reset_voltages(self):
		self._rows = [PulseCreator(
			t_rise = self.t_rise,
			t_write = self.t_write,
			t_read = self.t_read,
			t_separation = 2*self.t_wait + self.t_rise_gate,
			V_pot = self.V_row_pot,
			V_dep = self.V_row_dep,
			V_read = self.V_row_read,
			V_standby = self.V_standby_pot,
			t_end = self.t_wait + self.t_rise_gate,
		) for _ in range(self.n_rows)]
		self._cols = [PulseCreator(
			t_rise = self.t_rise,
			t_write = self.t_write,
			t_read = self.t_read,
			t_separation = (2*self.t_wait) + self.t_rise_gate,
			V_pot = self.V_col_pot,
			V_dep = self.V_col_dep,
			V_read = self.V_col_read,
			V_standby = self.V_standby_dep,
			t_end = self.t_wait + self.t_rise_gate,
		) for _ in range(self.n_cols)]
		self._gates = [PulseCreator(
			t_separation = self.t_wait,
			t_rise = self.t_rise_gate,
			t_write = 2*self.t_wait + 2* self.t_rise + self.t_write,
			t_read =  2*self.t_wait + 2* self.t_rise + self.t_read,
			V_pot = self.V_gates_pot,
			V_dep = self.V_gates_dep,
			V_read = self.V_gates_dep,
			V_standby = self.V_standby_dep,
			t_end = 0,
		) for _ in range(self.n_rows)]

	def gen_stochastic_waveforms(self, x_bitstreams, err_bitstreams):
		self.append_read_whole_matrix()
		#cycle (+, +)
		for bit in range(x_bitstreams.shape[1]):
			self.generate_parallel_pot(x=x_bitstreams[0, bit, :], err=err_bitstreams[0, bit, :])
		#self.append_read_whole_matrix()
		#cycle (+, -)
		for bit in range(x_bitstreams.shape[1]):
			self.generate_parallel_dep(x=x_bitstreams[1, bit, :], err=err_bitstreams[1, bit, :])
		#self.append_read_whole_matrix()
		#cycle (-, +)
		if not self.two_phases:
			for bit in range(x_bitstreams.shape[1]):
				self.generate_parallel_dep(x=x_bitstreams[2, bit, :], err=err_bitstreams[2, bit, :])
				#self.append_read_whole_matrix()
			#cycle (-, -)
			for bit in range(x_bitstreams.shape[1]):
				self.generate_parallel_pot(x=x_bitstreams[3, bit, :], err=err_bitstreams[3, bit, :])
		self.append_read_whole_matrix()	
		self.to_files()
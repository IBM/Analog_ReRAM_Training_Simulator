
# Auxiliary functions to support SPICE simulation

# Import
import subprocess
from pathlib import Path
from src.pulse_generator import ReRAMMatrixPulseGenerator


# Run ngspice process with terminal command as input (see main.ipynb)
def run_ngspice(ngspice_command):
    try:
        print("Starting SPICE simulation...")
        subprocess.run(ngspice_command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ngspice failed with exit code {e.returncode}:\n{e.stderr}"
        ) from e
    print("SPICE simulation finished successfully.")


# Generate waveform files, patch the transient line in the crossbar netlist, run the initial simulation with initial conditions set, then clean up
def initialize_cir_file_and_read(crossbar: Path,
						ngspice_command: str,
						pulse_generator: ReRAMMatrixPulseGenerator):
	pulse_generator.append_read_whole_matrix()
	pulse_generator.to_files()
	restore_line = ".include simulation_output/nodes.txt"
	timestep_units = format_si(25e-9)
	end_time = pulse_generator._rows[-1].points[-3][0]
	time_units = format_si(end_time)
	transient_line = f"tran {timestep_units} {time_units} {timestep_units}"
	with open(crossbar, 'r') as file:	
		lines = file.readlines()
	new_lines = []
	for line in lines:
		if line.strip() == restore_line:
			continue
		elif line.strip().startswith("tran"):
			new_lines.append(transient_line + " uic\n")
		else:
			new_lines.append(line)
	with open(crossbar, 'w') as file:
		file.writelines(new_lines)
	# Add .ic (initial conditions) for the ReRAM model in the circuit netlist file
	ic_lines = ['\t.ic V(C_V) = C_init', '\t.ic V(T) = T_init']
	submodel_file = Path('simulation_models/IBM_CMO_HfOx_ReRAM_ngspice.sub')
	if not submodel_file.is_file():
		print(f"Submodel file {submodel_file} not found. Please check the path.")
		return
	with open(submodel_file, 'r') as file:
		lines = file.readlines()
	# Insert initial conditions after line that starts with '\t\t +R_th (parameters line)'
	for i, line in enumerate(lines):
		if line.startswith('\t\t+ R_th'):
			lines.insert(i + 1, ic_lines[0] + '\n')
			lines.insert(i + 2, ic_lines[1] + '\n')
			break
	with open(submodel_file, 'w') as file:
		file.writelines(lines)
	run_ngspice(ngspice_command)
	pulse_generator.reset_voltages()
	# Remove the initialization of nodes as a new circuit should recover values from nodes.txt
	# Remove lines that start with '.ic'
	with open(submodel_file, 'r') as f:
		lines = f.readlines()
	lines = [line for line in lines if not line.startswith('\t.ic')]
	with open(submodel_file, 'w') as f:
		f.writelines(lines)


# Define units
def format_si(x):
    for scale, suffix in [(1e-3, "m"), (1e-6, "u"), (1e-9, "n")]:
        if abs(x) >= scale:
            return f"{x / scale:g}{suffix}"
    return f"{x:g}"


# Update the transient simulation line in the netlist and restore node states from a previous run
def update_transient_time(crossbar, time:float, timestep:float=25e-9):
	timestep_units = format_si(timestep)
	time_units = format_si(time)
	update_transient_line = f"tran {timestep_units} {time_units} {timestep_units}"
	restore_nodes = ".include simulation_output/nodes.txt"
	with open(crossbar, 'r') as f:
		lines = f.readlines()
		lines.insert(3, restore_nodes + '\n')
	for i,line in enumerate(lines):
		if line.strip().startswith('tran'):
			lines.pop(i)
			lines.insert(i, update_transient_line+ '\n')
	with open(crossbar, 'w') as f:
		f.writelines(lines)
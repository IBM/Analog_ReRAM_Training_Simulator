# Fully Parallel Outer Product in a ReRAM Crosspoint Array for Analog In-Memory Training

An ngspice implementation using the IBM analog filamentary Conductive-Metal-Oxide/HfOx ReRAM device.

Victoria Clerico, Matteo Galetta, Matias Senger, Bert J. Offrein, Valeria Bragaglia


IBM Research Europe – Zurich

This repository accompanies:

> [1] V. Clerico, M. Galetta, M. Senger, W. Choi, D.F. Falcone, A. Todri-Sanial, B.J. Offrein, and V. Bragaglia. "Proving Analog In-Memory Training with Fully Parallel Outer Product in ReRAM Crosspoint Array." Under review at *npj Unconventional Computing* (2026).

## Overview

A SPICE-python engine implementing a fully parallel outer product (weight update) operation on a 10x10 crosspoint array, simulated in ngspice. Each crosspoint cell is a 1Transistor-1ReRAM (1T1ReRAM) cell, combining an nFET model with the ngspice implementation of the IBM analog filamentary CMO/HfOx ReRAM device (in `simulation_models/`).

The outer-product engine can be chained with a forward/backward pass and gradient computation in python to implement a full analog in-memory training loop.

## Weight update schemes

Two stochastic parallel update schemes are supported, selected in `ReRAMMatrixPulseGenerator`:

- **Conventional stochastic parallel weight update scheme** — the original scheme from [2]. A cell updates whenever the full programming voltage (SET or RESET) is applied across the bitline (row) and sourceline (column) alone; wordlines (gates) stay enabled throughout the update cycle.
- **Gate-coincidence stochastic parallel weight update scheme** (this work, [1]) — a cell updates only when the full programming voltage is applied across the bitline/sourceline *and* a positive gate voltage is simultaneously applied on the wordline, mitigating IR drop during array updates.

Reference:
> [2] T. Gokmen and Y. Vlasov. "Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices: Design Considerations." *Frontiers in Neuroscience*, v.10 p.333 (2016). https://doi.org/10.3389/fnins.2016.00333

## Repo structure

| File | Content |
|---|---|
| `main.ipynb` | Reference/tutorial notebook: single outer-product update walkthrough (read → generate waveforms → SPICE run (weight update) → read updated conductance map) |
| `src/pulse_generator.py` | `ReRAMMatrixPulseGenerator` / `PulseCreator` — Builds piecewise voltage waveforms (bitline/sourceline/wordline) per update scheme |
| `src/weight_update.py` | `BitstreamGenerator` — Encodes input (x) and error (δ) vectors as stochastic bitstreams |
| `src/utils/plotting.py` | Conductance heatmaps plotting |
| `src/utils/ngspice_file_parser.py` | ngspice invocation and `.cir` netlist manipulation (`run_ngspice`, `initialize_cir_file_and_read`, `update_transient_time`) |
| `src/spice_data_analyzer.py` | Parses ngspice output into per-cell current/voltage/resistance/conductance data |
| `src/circuit/10x10.cir` | 10x10 1T1ReRAM crosspoint array netlist |
| `simulation_models/` | ngspice subcircuit for the IBM CMO/HfOx ReRAM device |
| `simulation_parameters.yaml` | Device switching parameters and pulse timing/voltage parameters |
| `figures/` | Example waveform diagrams referenced in `main.ipynb` |

## Requirements

- Python 3
- [ngspice](https://ngspice.sourceforge.io/) on `PATH`

## Running

**Quick single-update walkthrough:** open `main.ipynb` and run cells in order. It performs one read, one parallel outer-product update, and shows the conductance map before/after.

## Citation

If you plan to use the tool or adapt it for academic work or publication, a citation to the following references is required:

> V. Clerico, M. Galetta, M. Senger, W. Choi, D.F. Falcone, A. Todri-Sanial, B.J. Offrein, and V. Bragaglia. "Proving Analog In-Memory Training with Fully Parallel Outer Product in ReRAM Crosspoint Array." Under review at *npj Unconventional Computing* (2026).


> M. Galetta, D.F. Falcone, V. Clerico, W. Choi, S. Menzel, A. La Porta, T. Stecconi, F. Horst, B.J. Offrein, and V. Bragaglia. "Study of Resistive Switching Dynamics and Memory States Equilibria in Analog Filamentary Conductive-Metal-Oxide/HfOx ReRAM via Compact Modeling". Advanced Electronic Materials 12 (7), e00373. (2025)

## License

This project is licensed under the Apache License 2.0.
It permits use, modification, and distribution for both academic and commercial purposes. Users are required to include the original license and provide proper attribution when redistributing this work.




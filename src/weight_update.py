#!/usr/bin/env python3
#
# Copyright IBM Corp. 2025, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""
Parallel Update Scheme
-----------------------
How to translate the input (x_i) and the error (δ_j) into stochastic bit streams of pulses of equal amplitude

From:
[2] T. Gokmen, and Y. Vlasov.
	“Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices: Design Considerations”
	Frontiers in Neuroscience, v.10 p.333 (2016). https://doi.org/10.3389/fnins.2016.00333

The weight update is defined by:
    w_ij  <- w_ij ± Δw_min (1 to BSL)∑ A_j ∧ B_j

Where BSL is the length of the stochastic bit stream, Δw_min is the change in the weight value
due to a coincidence event, A_j and B_j are random Bernoulli variables, n is the bit position in the sequence. 
The probabilities that A_j and B_j are equal to unity are given by x_i and δ_j
"""

# Import
import numpy


# Translate x and δ vectors into bit streams
class BitstreamGenerator():
    def __init__(self,
                bit_length:int):
        self.bit_length = int(bit_length)
    
    def get_bistreams(self, x:numpy.ndarray, err:numpy.ndarray, two_phases:bool = False):
        # Four cycles of pulses (x, err) --> (++, +-, -+, --)
        # Get indexes of positive and negative values
        x_pos = numpy.where(x > 0.05)[0]
        err_pos = numpy.where(err > 0.05)[0]
        x_neg = numpy.where(x < -0.05)[0]
        err_neg = numpy.where(err < -0.05)[0]
        # Create bit streams
        x_bitstreams = numpy.zeros((2 if two_phases else 4, self.bit_length, len(x)), dtype=numpy.int8)
        err_bitstreams = numpy.zeros((2 if two_phases else 4, self.bit_length, len(err)), dtype=numpy.int8)
        # Case of positive x, positive δ --> Update cycle with SET operations
        if x_pos.size >0 and err_pos.size>0:
            x_bitstreams[0, :, x_pos] = (numpy.random.rand(len(x_pos), self.bit_length) < x[x_pos][:, None]).astype(numpy.int8)
            err_bitstreams[0, :, err_pos] = (numpy.random.rand(len(err_pos), self.bit_length) < err[err_pos][:, None]).astype(numpy.int8)*-1
        # Case of positive x, negative δ --> Update cycle with RESET operations
        if x_pos.size>0 and err_neg.size>0:
            x_bitstreams[1, :, x_pos] = (numpy.random.rand(len(x_pos), self.bit_length) < x[x_pos][:, None]).astype(numpy.int8)*-1
            err_bitstreams[1, :, err_neg] = (numpy.random.rand(len(err_neg), self.bit_length) < -err[err_neg][:, None]).astype(numpy.int8)
        if two_phases:
            return x_bitstreams, err_bitstreams
        # Case of negative x, positive δ --> Update cycle with RESET operations
        if x_neg.size>0 and err_pos.size>0:
            x_bitstreams[2, :, x_neg] = (numpy.random.rand(len(x_neg), self.bit_length) < -x[x_neg][:, None]).astype(numpy.int8) *-1
            err_bitstreams[2, :, err_pos] = (numpy.random.rand(len(err_pos), self.bit_length) < err[err_pos][:, None]).astype(numpy.int8)
        # Case of negative x, negative δ --> Update cycle with SET operations
        if x_neg.size>0 and err_neg.size>0:
            x_bitstreams[3, :, x_neg] = (numpy.random.rand(len(x_neg), self.bit_length) < -x[x_neg][:, None]).astype(numpy.int8)
            err_bitstreams[3, :, err_neg] = (numpy.random.rand(len(err_neg), self.bit_length) < -err[err_neg][:, None]).astype(numpy.int8)*-1
        return x_bitstreams, err_bitstreams
# MIT License
#
# Copyright (c) 2018, Andrew Warrington.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
mhSolver.py
AW

TL;DR -
"""

# Import stock modules.
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules.
import homogenousKF.evaluateSchedule as es


def run_chain():
	
	# Define parameters of MH.
	mh_steps = 10000
	transition_prob = [0.2, 0.6, 0.2]  # The local transition kernel.
	transition_kern = [-1, 0, 1]       # Transition kernel space.
	
	# Do automated stuff.
	samples = [None] * mh_steps
	
	n = np.random.randint(es.t_max)
	samples[0] = es.calculate_kalman_cost(n)
	
	for i in range(1, mh_steps):
		
		if n == 0:
			n_new = n + np.random.choice(transition_kern, p=[0, 1 - transition_prob[2], transition_prob[2]])
		else:
			n_new = n + np.random.choice(transition_kern, p=transition_prob)
		
		s = es.calculate_kalman_cost(n_new)
		
		# Now calculate the MH ratio.
		A = np.min((1., s['r'] / samples[i-1]['r']))
		if np.random.rand() < A:
			n = n_new
			samples[i] = s
		else:
			samples[i] = samples[i-1]
	
	return samples


if __name__ == '__main__':

	print('Weird bug that optimum is flat...')
	
	mhSamples = run_chain()

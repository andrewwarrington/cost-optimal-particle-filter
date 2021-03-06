# MIT License
#
# Copyright (c) 2018, Andrew Warrington
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

'''
repeatSampleProblem.py
AW

TLDR -- This script creates the results shown in Warrington and Dhir 2018,
for the `repeat sample problem', with the intention of demonstrating the purpose
of the work and allowing for users to explore the ideas expressed in the
aforementioned work.
'''

# Configure MatPlotLib for use on clusters. If the code is run on linux,
# we assume it is being run on a cluster and hence more processes (agents)
# should be used. The switch below can be removed if this is not applicable
# for you, and the number of processes to be used can be modified by changing
# the agents variable.
import matplotlib
from sys import platform
import multiprocessing as proc
if platform == "linux" or platform == "linux2":
	# On a linux cluster.
	matplotlib.use('Agg')
	agents = 40
else:
	# On a local machine.
	agents = None
	
# Set up the pool of workers.
# If multiprocessing shouldnt be used, set the number of agents to None.
if agents is not None:
	batch_pool = proc.Pool(processes=agents)
else:
	batch_pool = None
	
# Import stock modules.
import numpy as np
import matplotlib.pyplot as plt
import itertools
import h5py as h5
import time
import os

# Import custom modules.
import particleFilter as pf
import util

# We dump all experimental logs into a folder, named here.
# If this folder doesnt exist (it shouldn't...), create it.
folder_output_name = "RSP_" + time.strftime("%Y_%m_%d") + "_" \
						+ time.strftime("%H_%M_%S")
report_name = folder_output_name + '/report.txt'
if not os.path.exists(folder_output_name):
	os.makedirs(folder_output_name)

# For development, seeding the generation may be helpful.
# Set the seed to None to stop seeding.
seed = None
np.random.seed(seed)

# Define some parameters of the simulation.
sigmaPlant = 0.01  # S.D. of the gaussian plant noise.
tMax = 100  # Number of timesteps to simulate over.

# Define the parameters of the cost-level optimisation.
# configurations = range(tMax + 1)  # Evaluate all possible configurations.
configurations = np.arange(0, tMax, 11)
lambdaHyper = 0.1  # Value of lambda hyperparameter.

# Define some slightly less important inference level parameters.
# Add one for sensible spacing to create nice interpolated plot.
nPointsToInterp = 10000 + 1

# Do some automated computation to save time later.
n_samples = len(configurations)
interp_bins = np.linspace(0, tMax, nPointsToInterp)

# Define some holders to be filled in later once the values have
# been calculated.
errors = np.zeros((n_samples,))
costs = np.zeros((n_samples,))
total_cost = np.zeros((n_samples,))


def calculate_var(_n, _plant_var, _t_max):
	'''
	AW.
	calculate_var -
	:param _n:          Number of observations to take during the trace.
	:param _plant_var:  Variance of the plant model.
	:param _t_max:      The length of the simulation.
	:return: The sum of the variances over the trace.
	'''
	# Equally space the observations, and add one
	# at the start and finish, just to make the calculations work.
	obs_plan = np.linspace(0, _t_max, _n+2)
	differences = np.diff(obs_plan)  # Find the spacing between each observe.
	variances = differences * differences * _plant_var * 0.5  # Calculate area.
	sum_var = np.sum(variances)  # Sum the variance triangles.
	return sum_var


def get_var(_n, _plant_var, _t_max, _interp_bins):
	'''
	AW.
	get_var -
	:param _n:
	:param _plant_var:
	:param _t_max:
	:param _interp_bins:
	:return:
	'''
	obs_plan = np.linspace(0, _t_max+1, _n+2)
	obs_plan = np.round(obs_plan[1:,])
	
	dt = np.diff(interp_bins)
	dt = np.mean(dt)
	
	vars = np.zeros((len(_interp_bins), 1))
	vars[1] = 0  # Set the initial variance.
	
	for _i in range(len(_interp_bins)):
		t = interp_bins[_i]
		if any(obs_plan == t):
			vars[_i] = 0
		else:
			vars[_i] = vars[_i-1] + dt * _plant_var
			
	vars = vars[:-1]
	return vars


# Main loop.
for _i in range(len(configurations)):
	n = configurations[_i]
	
	errors[_i] = calculate_var(n, sigmaPlant, tMax)
	costs[_i] = _i
	temp = get_var(n, sigmaPlant, tMax, interp_bins)
	
	total_cost[_i] = errors[_i] + lambdaHyper * costs[_i]


opt_val = np.min(total_cost)
opt_loc = np.argmin(total_cost)
print(opt_loc)

# Save the results to an output file for recording and post-processing.
f = h5.File(folder_output_name + '/results.h5', 'w')
f.create_dataset("spacing", data=configurations)
f.create_dataset("costs", data=costs)
f.create_dataset("errors", data=errors)
f.create_dataset("total_cost", data=total_cost)
f.close()

# Quickly throw up a figure representing the reconstruction errors.
fig = plt.figure()
plt.plot(configurations, errors)
plt.plot(configurations, costs)
plt.plot(configurations, total_cost)
plt.savefig(folder_output_name + '/kalman_cost.png')

p = 0


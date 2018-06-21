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
one-sample-problem.py
AW

TL;DR -- This script creates the results shown in Warrington and Dhir 2018,
for the `one sample problem', with the intention of demonstrating the purpose 
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
folder_output_name = "sub_1d_" + time.strftime("%Y_%m_%d") + "_" \
						+ time.strftime("%H_%M_%S")
report_name = folder_output_name + '/report.txt'
if not os.path.exists(folder_output_name):
	os.makedirs(folder_output_name)

# For development, seeding the generation may be helpful.
# Set the seed to None to stop seeding.
seed = None
np.random.seed(seed)

# Configure the 'cost' level parameters of the inference procedure.
# (Generally, bigger number here mean longer run times, but less noisy
# estimates of quantities.)
nTrueTraces = 10

# Set the parameters of the generative procedure. 
tMax = 10                # The time of the simulation.
dt = 0.1                 # The granularity of the simulation.
initialPositionMean = 0  # Initial position mean.  
initialPositionVar = 0   # Variance of initial position estimate.
initialVelocityMean = 2  # Initial velocity mean.
initialVelocityVar = 0   # Variance of initial velocity estimate.
velocityVar = 0.1        # Variance of the plant model.
nParticles = 100         # How many particles to use in inference.

# Configure the gridding setup for performing the `inference'.
# Increment each one by one for logical placing.
tObsBins = 10 + 1  # The number of observation times to evaluate.
xObsBins = 20 + 1  # The number of locations to evaluate.

# Do some automatic computation to save time later.
tSteps = np.arange(0, tMax+dt, dt)   
nSteps = len(tSteps)

# Select the times at which to evaluate.
tObs = np.round(np.linspace(0, nSteps-1, tObsBins)).astype(np.int)

# Need to find the 99% point of where the sub ends up.
expectedFinalPosition = tMax * initialVelocityMean
	
# Select the spatial position of the buoy.
xObs = np.linspace(initialPositionMean, expectedFinalPosition, xObsBins)

# A slightly more extensive set of positions to examine at
# allows the buoy to be placed outside of the expected range, but this
# is a bit of a corner case. However, to do this, use the code below:
# finalModelVar = velocityVar * nSteps
# final3SdUpper = expectedFinalPosition + 3 * finalModelVar
# final3SdLower = expectedFinalPosition - 3 * finalModelVar
# xObs = np.linspace(initialPositionMean, final3SdUpper, xObsBins)


def iterate_1(_x, _v):
	'''
	AW.
	iterate_1 - iterate the model a single step in a Gaussian random
	walk on velocity.
	:param _x: the current position (vector of particles).
	:param _v: the current velocity (vector of particles).
	:return: tuple of position and velocity.
	'''
	_v += np.random.normal(0, velocityVar, size=np.shape(_v))
	_x += _v * dt
	return np.asarray((_x, _v)).transpose()


def iterate_n(_x_0, _v_0, _n, _n_particles=nParticles):
	'''
	AW.
	iterate_n - iterate the model n steps in a Gaussian random walk.
	:param _x_0: The initial position (vector of particles).
	:param _v_0: The initial velocity (vector of particles).
	:param _n:   The number of time steps to iterate (positive int).
	:param _n_particles: The number of particles to iterate.
	:return: the newly sampled particles.
	'''
	hist = np.zeros((_n_particles, _n+1, 2))
	hist[:, 0, 0] = _x_0
	hist[:, 0, 1] = _v_0
	for _i in range(1, _n+1):
		hist[:, _i, :] = iterate_1(hist[:, _i - 1, 0], hist[:, _i - 1, 1])
	return hist[:, 1:, :]


def observation_var(_param):
	'''
	AW.
	observation_var - calculate the variance of the observation.
	Normally used as a truth model in simulation.
	:param _param: list of parameters used in calculation.
	:return:
	'''
	diff = _param[0]
	eps = 0.01
	slope = 0.5
	return slope * np.abs(diff) + eps


def get_observation(_x, _n, _true_trajectory):
	'''
	AW.
	get_observation - simulate an observation from the true trajectory.
	This is a function that is used in simulation _only_, since it is
	used to falsify an observation that would be otherwise available,
	and therefore it is okay for us to use true_trajectory here!
	:param _x: the current location of the buoy.
	:param _n: the timestep we are at.
	:param _true_trajectory: the true locations of the submarine.
	:return:
	'''
	_var = observation_var([_x - _true_trajectory[0, _n, 0]])             # Get the variance.
	return np.random.normal(_true_trajectory[0, _n, 0], np.sqrt(_var))  # Return an noisy observation.


def solve_for_one_trace(_k=None):
	'''
	AW.
	solve_for_one_trace - solves for the expected reconstruction error
	for each of the masks specified in the setup. This is the expected
	reconstruction error when compared to a single, synthetic ground
	truth which we generate upfront, and then perform inference to
	recover.
	:param _k: run number.
	:return: expected reconstruction error for whole time series.
				for all values of buoy location and sampling time.
	'''
	
	# Output and also seed if we are seeding.
	util.echo_to_file(report_name, 'Simulating run ' + str(_k))
	if not seed:
		np.random.seed()
	else:
		np.random.seed(_k + seed)
	
	# Define some holders for variables of interest.
	# The reconstruction error for this true trace
	single_reconstruction_error = np.zeros((tObsBins, xObsBins))
	# dims of: (particles (one for GT), time step, state).
	trueTrajectory = np.zeros((1, nSteps, 2))  
	
	# Sample the initial position and velocity.
	trueTrajectory[0, 0, 0] = np.random.normal(initialPositionMean, initialPositionVar)
	trueTrajectory[0, 0, 1] = np.random.normal(initialVelocityMean, initialVelocityVar)
	trueTrajectory[:, 1:, :] = iterate_n(trueTrajectory[0, 0, 0], trueTrajectory[0, 0, 1], nSteps - 1, 1)  # Use one particle.
	
	# Loop over all the masks specified.
	for t, x in masks:
		# Extract the mask.
		tSamp = np.int(tObs[t])
		xSamp = xObs[x]
		
		# Define a holder for the particles at all time steps, for all states,
		# states being position and velocity.
		particles = np.zeros((nParticles, nSteps, 2))
		
		# Sample the initial conditions.
		particles[:, 0, 0] = np.random.normal(initialPositionMean, initialPositionVar, size=nParticles)
		particles[:, 0, 1] = np.random.normal(initialVelocityMean, initialVelocityVar, size=nParticles)
		
		# Iterate the model until the sample time.
		particles[:, range(1, tSamp + 1), :] = iterate_n(particles[:, 0, 0], particles[:, 0, 1], tSamp)
		formatted_particles = np.reshape(particles[:, tSamp, 0], (nParticles, 1))
		
		# Calculate the expected position at the time of sampling.
		# This expected position is then used to calculate the estimate 
		# of observation noise.
		distances = xSamp - particles[:, tSamp, 0]
		var = observation_var([distances])
		expectedVar = np.mean(var)
		
		# Generate/falsify the observation.
		# Permitted to use true trajectory.
		observation = get_observation(xSamp, tSamp, trueTrajectory)  
		
		# Resample using the particle filter and inscribe the resampled particles 
		# into the particle matrix.
		resampled_idx = pf.iterate(formatted_particles, observation, np.sqrt(expectedVar)).get('resampled_indices')		
		resampled_particles = particles[resampled_idx, tSamp, :]
		particles[:, tSamp, :] = resampled_particles
		
		# Now iterate out the remainder of the particles to the end of the
		# time series.
		particles[:, tSamp + 1:, :] = iterate_n(particles[:, tSamp, 0], particles[:, tSamp, 1], nSteps - tSamp - 1)
		
		# Solve for the expected location at each time step.
		expected_location = np.squeeze(np.mean(particles[:, :, 0], axis=0))
		
		# Calculate the expected sum reconstruction error for the current mask.
		single_reconstruction_error[t, x] = np.sum(np.square(expected_location - trueTrajectory[0, :, 0]))
		
	return single_reconstruction_error

# Define a list for the the masks that we are evaluating over.
masks = list(itertools.product(range(tObsBins), range(xObsBins)))

# Write some stuff to the file.
util.echo_to_file(report_name, 'Beginning gridding inference.')
util.echo_to_file(report_name, (str(nTrueTraces) + ' traces to simulate.'))

if batch_pool is not None:
	rec = batch_pool.map_async(solve_for_one_trace, range(nTrueTraces))
	rec = rec.get()
else:
	rec = [solve_for_one_trace(_i) for _i in range(nTrueTraces)]
	# reconstruction_error[range(k * batch_size, (k+1) * batch_size), :, :] = rec
reconstruction_error = np.asarray(rec)

# Save the results to an output file for recording and post-processing.
f = h5.File(folder_output_name + '/results.h5', 'w')
f.attrs.create('Ordering', 'x-obs, t-obs, MC-Samples.'.encode('utf8'))  # h5 reverses the order.
f.create_dataset("reconstruction_errors", data=reconstruction_error.transpose([2, 1, 0]))
f.create_dataset("time_bins", data=tSteps)
f.create_dataset("x_bins", data=xObs)
f.close()

# Quickly throw up a figure representing the reconstruction errors.
# Should be a 3-D graph realistically.
a = np.mean(reconstruction_error, axis=0)
fig = plt.figure()
[plt.plot(xObs, _a) for _a in a]
plt.savefig(folder_output_name + '/reconstruction_final.png')

# Plot a heatmap of the error.
fig = plt.figure()
plt.imshow(np.mean(reconstruction_error, axis=0), cmap='hot', interpolation='nearest')
plt.pause(0.01)
plt.savefig(folder_output_name + '/heatmap_final.png')

p = 0


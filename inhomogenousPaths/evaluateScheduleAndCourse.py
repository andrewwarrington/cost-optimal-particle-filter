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
evaluateScheduleAndCourse.py
AW

TL;DR -
"""

# Import stock modules.
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import custom modules.
import inhomogenousPaths.generateRandomCourse as grc

# Define simulation parameters.
dimensions = 2
dt = 1

# Define the hyperparameter.
lambda_hyp = 0.1  # Cost weight.
phi_hyp = 0.001    # Var weight.

# Plant model.
vVarForwards = 0.00                        # Variance of the (forward) velocity in the plant model.
vVarSidewaysConst = 0.2                    # Constant part of the variance of the sideways velocity of the object.
vVarSidewaysGrad = 50                      # Gradient that scales the curvature to a variance.
# plantModelIterate = lambda currentLocation, currentVelocity: currentLocation + currentVelocity * dt
plantModelNoiseCov = lambda curvature: np.asarray([[vVarForwards, 0], [0, vVarSidewaysConst + np.abs(vVarSidewaysGrad*curvature)]])
# plantModelNoise = lambda currentVelocity, curvature: np.random.multivariate_normal(currentVelocity, plantModelNoiseCov(curvature))

# Observation model.
observationVar = 0.0
observationModel = lambda _gt_state: _gt_state

# Cost model.
unitCost = 1
costModel = lambda: unitCost


def evaluate(_schedule, _course, _return_just_value=False, _plot=False):
	'''
	AW - evaluate
	:param _schedule: WHERE(M==True).
	:param _course:   dict of course, bearing, curvature, and velocity.
	:param _plot:     plot figures (default = False).
	:return:
	'''
	# Extract some analytes from the course.
	t_max = len(_course['x'][:, 0])
	
	# Make sure the schedule is just a vector.
	_schedule = np.round(np.squeeze(_schedule))
	
	# Set up some holders for the inference results.
	variance_trajectory = np.zeros((t_max, 2))  # Forward, sideways.
	cost = len(_schedule)
	
	# Do initialisation.
	variance_trajectory[0, :] = [0, 0]  # Know initialisation exactly.
	
	# Now step through time series.
	for t in range(1, t_max):
		# Iterate.
		covariance = plantModelNoiseCov(_course['c'][t])
		variances = np.diagonal(covariance)
		variance_trajectory[t, :] = variance_trajectory[t - 1, :] + variances
		
		# # Check for observe.
		if t in _schedule:
			# Resample.
			variance_trajectory[t, :] = [0.0] * dimensions  # TODO - Currently noise free.
			
	total_cost = cost
	total_benefit = - np.sum(variance_trajectory[:, 1])  # Note - only interested in lateral variance.
	total_reward = np.exp(phi_hyp * total_benefit - lambda_hyp * total_cost)
	
	if _plot:
		plt.figure()
		plt.plot(variance_trajectory)
		plt.pause(0.1)
	
	rewards = {'r': total_reward,
				'c': total_cost,
				'b': total_benefit,
				's': _schedule}
	
	if _return_just_value:
		rewards = total_reward
	
	return rewards


if __name__ == '__main__':
	course = grc.generate_course(_seed=1)
	schedule = np.round(np.linspace(10, 900, 100))
	
	print(evaluate(schedule, course))
	
	plt.figure()
	plt.scatter(course['x'][:, 0], course['x'][:, 1])
	plt.axis('equal')
	plt.pause(0.1)
	
	p = 0



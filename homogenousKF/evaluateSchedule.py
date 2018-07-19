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
evaluateSchedule.py
AW

TL;DR -
"""

import numpy as np


# Define constants.

# Simulation params.
t_max = 100

# Model params.
plant_var = 0.01
lambda_hyp = 0.1


def calculate_kalman_cost(n):

	# Overload n so that we can directly pass in an observation schedule
	# constructed outside, or we can generate an equally spaced observation
	# plan.
	
	if np.isscalar(n):
		obs_plan = equally_space(round(n), t_max)
	else:
		obs_plan = n
		obs_plan = [0.0, obs_plan, t_max]
	
	differences = np.diff(obs_plan)
	variances = differences * differences * plant_var * 0.5
	
	sum_var = np.sum(variances)
	cost = len(obs_plan) - 2
	total_cost = np.exp(- sum_var - lambda_hyp * cost)
	
	return {'r': total_cost, 'u': sum_var, 'c': cost, 'n': n}


def equally_space(n, t_max):
	obs_plan = np.linspace(0, t_max+1, n+2)
	obs_plan = np.round(obs_plan)
	return obs_plan
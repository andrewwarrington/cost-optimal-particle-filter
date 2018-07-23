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
import time
import os
import copy

# Import custom modules.
import inhomogenousPaths.generateRandomCourse as grc
import inhomogenousPaths.evaluateScheduleAndCourse as esac

samplesToMake = 2

course = grc.generate_course(2)
plt.figure()
plt.scatter(course['x'][:, 0], course['x'][:, 1])
plt.axis('equal')
plt.pause(0.1)

optimal_schedule = np.round(np.linspace(0, grc.t_max, samplesToMake + 2)).astype(np.int)
optimal_schedule = optimal_schedule[1:-1]
optimal_reward = esac.evaluate(optimal_schedule, course)

schedule = np.sort(np.random.randint(0, grc.t_max, samplesToMake))

# Define parameters of MH.
mh_steps = 10000
transition_prob = [0.2, 0.6, 0.2]  # The local transition kernel.
transition_kern = [-1, 0, 1]  # Transition kernel space.

# Do automated stuff.
samples = [None] * mh_steps
samples[0] = esac.evaluate(schedule, course)

for i in range(1, mh_steps):
	
	s_new = copy.deepcopy(schedule)
	for _s in range(samplesToMake):
		if s_new[_s] == 0:
			s_new[_s] = schedule[_s] + np.random.choice(transition_kern, p=[0, 1 - transition_prob[2], transition_prob[2]])
		else:
			s_new[_s] = schedule[_s] + np.random.choice(transition_kern, p=transition_prob)
	
	s = esac.evaluate(s_new, course)
	
	# Now calculate the MH ratio.
	A = np.min((0., s['r'] - samples[i - 1]['r']))  # Note - performed in log space.
	if np.log(np.random.rand()) < A:
		schedule = s_new
		samples[i] = s
	else:
		samples[i] = samples[i - 1]


p = 0


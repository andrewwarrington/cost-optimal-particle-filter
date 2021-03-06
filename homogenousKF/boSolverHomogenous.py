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
boSolverHomogenous.py
AW

TL;DR -
"""

# Import stock modules.
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import GPy
import GPyOpt
from functools import partial

# Import custom modules.
import homogenousKF.evaluateHomogenousSchedule as es

# Set up gpyopt stuff.
domain = [{'name': 'samples', 'type': 'continuous', 'domain': (0, es.t_max)}]


def f(_s):
	return es.calculate_kalman_cost(_s, _return_just_value=True, _pop=True)


myBopt = GPyOpt.methods.BayesianOptimization(f=f, domain=domain, maximize=True)
myBopt.run_optimization(max_iter=100)
myBopt.plot_acquisition()

Y_s = np.squeeze(myBopt.Y)
m = np.argmax(Y_s)
Y_max = Y_s[m]
X_max = myBopt.X[m, :]




p = 0


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
generateRandomCourse.py
AW

TL;DR -- This script contains the utilities for generating a random course
(or a fixed course using a seed).
'''

import numpy as np
import matplotlib.pyplot as plt

t_max = 500

def generate_course(_seed=None, _straight=False):
	
	# Seed np if we are generating a fixed course.
	np.random.seed(_seed)
	
	# Define course parameters.
	velocity = 1
	if _straight:
		switch_probability = 0.00
	else:
		switch_probability = 0.01
	waypoint_curvature_sd = np.square(np.pi / 3)
	current_bearing = 2 * np.pi * np.random.rand()
	bearing_change_mean = 40
	starting_steps = 10
	ending_steps = 10

	# Define automated stuff.
	target_bearing = current_bearing
	middlet_max = t_max - starting_steps - ending_steps
	
	# Define course holders and counters.
	x_points = np.zeros((t_max, 2))
	b_points = np.zeros((t_max, 1))
	c_points = np.zeros((t_max, 1))
	v_points = np.zeros((t_max, 2))  # x, y.
	
	x_points[0, ] = [0.0, 0.0]
	c_points[0] = 0.0
	b_points[0] = current_bearing
	v_points[0, ] = np.asarray([velocity * np.cos(current_bearing), velocity * np.sin(current_bearing)])
	current_curvature = 0.0
	point_counter = 1

	# Create starting straight.
	for i in range(1, starting_steps):
		x_points[point_counter] = np.asarray([x_points[point_counter - 1, 0] + velocity * np.cos(current_bearing), x_points[point_counter - 1, 1] + velocity * np.sin(current_bearing)])
		c_points[point_counter] = current_curvature
		b_points[point_counter] = current_bearing
		v_points[point_counter, ] = np.asarray([velocity * np.cos(current_bearing), velocity * np.sin(current_bearing)])
		
		point_counter = point_counter + 1
	
	# Create intermediate path.
	for i in range(middlet_max):
		
		x_points[point_counter] = np.asarray([x_points[point_counter - 1, 0] + velocity * np.cos(current_bearing), x_points[point_counter - 1, 1] + velocity * np.sin(current_bearing)])
		c_points[point_counter] = current_curvature
		b_points[point_counter] = current_bearing
		v_points[point_counter, ] = np.asarray([velocity * np.cos(current_bearing), velocity * np.sin(current_bearing)])
		
		if switch_probability > np.random.rand():
			target_bearing = np.random.normal(current_bearing, waypoint_curvature_sd)
			change_steps = np.random.poisson(bearing_change_mean)
			next_bearings = np.linspace(current_bearing, target_bearing, change_steps)
		
		if np.abs(target_bearing - current_bearing) < 0.0001:
			current_bearing = target_bearing
			current_curvature = 0
		else:
			current_curvature = current_bearing - next_bearings[0]
			current_bearing = next_bearings[0]
			next_bearings = next_bearings[1:]
		
		point_counter = point_counter + 1
		
	# Create the final straight.
	for i in range(ending_steps):
		x_points[point_counter] = np.asarray([x_points[point_counter - 1, 0] + velocity * np.cos(current_bearing), x_points[point_counter - 1, 1] + velocity * np.sin(current_bearing)])
		c_points[point_counter] = 0
		b_points[point_counter] = current_bearing
		v_points[point_counter, ] = np.asarray([velocity * np.cos(current_bearing), velocity * np.sin(current_bearing)])
		
		point_counter = point_counter + 1
		
	np.random.seed()
		
	return {'x': x_points, 'c': c_points, 'b': b_points, 'v': v_points}
		
	
if __name__ == '__main__':
	course = generate_course()
	
	plt.figure()
	plt.scatter(course['x'][:, 0], course['x'][:, 1])
	plt.axis('equal')
	plt.pause(0.1)
	
	p = 0

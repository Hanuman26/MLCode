# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#This is a simple implementation of gradient descent for linera regression on a simle two dimensional data stored in data.csv
from numpy import *
from matplotlib import pyplot as plt
def run():
    points = genfromtxt("/home/mirk/ML/mycode/gradient-descent/data.csv", delimiter=",")
    learning_rate = 0.0001
# y = mx+b
    initial_b = 0
    initial_m = 0
    num_iterations = 2000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error(initial_b, initial_m, points))
    print "********************************************************"
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error(b, m, points))
    x = points[:,0]
    y = points[:,1]    
    plt.scatter(x,y)
    plt.plot(x,(m * x) + b)
    plt.show()
	

#function for calculating total error for candidate line 'y' value
def compute_error(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error/float(len(points))

#function for step gradient, finding minima for the b,m
def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y -((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)  
    return [new_b, new_m]

#function for running descent, iterating for finding best fit
def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    image = []
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m] 

if __name__ == '__main__':
    run()	

# Change the learning rate and check whether cost is
# decreasing. If yes you can increase the cost.
# Change the iteration depending on the efficiency you need.
# Once you reach Global Minima your cost will remain the same.

import numpy as np
import math

def gradient_descent(x,y, lr = 0.0001, iterate = 1000):
    m_curr = b_curr = 0     # Initializing the values.
    iteration = iterate
    learning_rate = lr
    n = len(x)
    cost_prev = 0

    for i in range(iteration):
        y_predicted = m_curr*x + b_curr     #y=mx+b
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        m_der = -(2/n) * sum(x*(y-y_predicted))
        b_der = -(2/n) * sum(y-y_predicted)
        m_curr = m_curr - (learning_rate * m_der)
        b_curr = b_curr - (learning_rate * b_der)
        if math.isclose(cost, cost_prev, rel_tol = 1e-20):
            break
        cost_prev = cost
        print("m= {}, b= {}, cost= {} iteration= {}".format(m_curr, b_curr, cost, i))

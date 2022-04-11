import numpy as np
from typing import *


def rk4(f, y0, t_eval, data):

    n_eq = len(y0)
    t0 = t_eval[0]
    dt = t_eval[1] - t_eval[0]

    f0 = np.zeros(n_eq)
    f(t0, y0, f0, data)
    
    t1 = t0 + dt / 2.0
    y1 = y0 + dt * f0 / 2.0
    f1 = np.zeros(n_eq)
    f(t1, y1, f1, data)
    
    t2 = t0 + dt / 2.0
    y2 = y0 + dt * f1 / 2.0
    f2 = np.zeros(n_eq)
    f(t2, y2, f2, data)
    
    t3 = t0 + dt	
    y3 = y0 + dt * f2
    f3 = np.zeros(n_eq)
    f(t3, y3, f3, data)
    
    y = y0 + dt * (f0 + 2.0 * f1 + 2.0 * f2 + f3) / 6.0

    return y


def dynamics(t, s, s_dot, data):

    """
    System dynamics: vertical landing on planetary body
        with constant gravity g and thrust T
    """
    #State
    h = s[0]
    v = s[1]
    m = s[2]

    #Data
    g = data[0]
    T = data[1]
    c = data[2] 

    #Equations of motion
    h_dot = v
    v_dot = - g + T/m
    m_dot = - T/c

    s_dot[0] = h_dot
    s_dot[1] = v_dot
    s_dot[2] = m_dot

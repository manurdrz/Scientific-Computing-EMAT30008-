# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:02:30 2023

@author: manue
"""

import numpy as np

def euler_step(f, x, t, dt):
    """
    Performs a single step Euler method to solve an ODE.

    Arguments:
    f -- function representing the derivative x' = f(x, t)
    x -- current value of x
    t -- current value of t
    dt -- time step

    Returns:
    The next value of x at time t+dt.
    """
    xnew = x + f(x, t) * dt
    tnew = t + dt
    return xnew, tnew

def rk4_step(f, x, t, dt):
            """
            Performs a single step Fourth Order Runge-Kutta  method to solve an ODE.
            """
            k1 = f(x, t)
            k2 = f(x + 0.5 * k1 * dt, t + 0.5 * dt)
            k3 = f(x + 0.5 * k2 * dt, t + 0.5 * dt)
            k4 = f(x + k3 * dt, t + dt)
            
            xnew= x + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt
            tnew= t + dt
            return xnew, tnew

def simpson38_step(f, x, t, dt):
            """
            Performs a single step simpson 3/8 method to solve an ODE.
            """
            k1 = f(x, t)
            k2 = f(x + dt*k1 /3, t + dt /3 )
            k3 = f(x - dt * (k1 /3 +k2),t + (2/3) * dt)
            k4 = f(x + dt*(+k1 -k2 +k3), t + dt)
            xnew= x + dt* (k1 + 3*k2 + 3*k3 + k4)/8
            tnew= t + dt
            return xnew, tnew
        
        
def midpoint_step(f, x, t, dt):
      """
      Performs a single step Midpoint method to solve an ODE.
      """
      k1 = f(x, t)
      k2 = f(x + dt/2, t + dt/2*k1)
      xnew= x + dt * k2
      tnew= t + dt
      return xnew, tnew
      
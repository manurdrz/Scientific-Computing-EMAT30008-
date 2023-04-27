import numpy as np
import sys
import matplotlib.pyplot as plt
import time
############### Equations ########################

def f(x, t, args):
    '''
    Function representing a first-order ODE: dx/dt = x
    Parameters:
    x : (float) - The value of the dependent variable x at time t.
    t : (float) - The value of the independent variable t.
    args : (dict) - Additional arguments required for the ODE.
    Returns:
    dxdt :  (float) - The value of the derivative dx/dt.
    '''
    return x

def exp(t):
    '''
    Function representing the analytical solution of the ODE: x(t) = exp(t)
    Returns:
    exp_t : (float) - The value of the dependent variable x(t) at time t.
    '''
    return np.exp(t)

def g(X, t, args):
    '''
    Function representing a second-order ODE in the form of a system of first-order ODEs.
    Parameters:
    X : (list) - A list containing the dependent variables [x, y] at time t.
    t : (float) - The value of the independent variable t.
    args : (dict) - Additional arguments required for the ODE.
    
    Returns:
    X : (list) - A list containing the derivatives [dx/dt, dy/dt].
    '''
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = -x
    X = [dxdt, dydt]
    return X

def g1(t):
    '''
    Function representing the analytical solution of the second-order ODE as a system of first-order ODEs.
    Parameters:
    t : (float) - The value of the independent variable t.
    Returns:
    sol : (list) - A list containing the dependent variables [sin(t), cos(t)] at time t.
    '''
    return [np.sin(t), np.cos(t)]


################### Steps ####################
def euler_step(f, x, t, dt, **params):
    '''
    One step of the Euler method.
    
    Parameters: 
    f : (function) - function containing ODE to be  solved.
    x : (np.array) - ODE's solution at time t.
    t : (float) - Time to evaluate .
    dt : (float) - Stepsize 
    **params: Any extra parameters  to evaluate  ODE.
    
    Returns :    
    Xnew : (np array) - ODE's solution at t + dt    
    '''
    dxdt = f(x,t, params)
    x_next = x + dt*np.array(dxdt)
    return x_next

def RK4_step(f, x, t, dt, **params):
    '''
    One step of the fourth-order Runge-Kutta method.
    
    Parameters: 
    f : (function) - function containing ODE to be solved.
    x : (np.array) - ODE's solution at time t.
    t : (float) - Time to evaluate.
    dt : (float) - Stepsize.
    **params: Any extra parameters to evaluate ODE.
    
    Returns :    
    Xnew : (np.array) - ODE's solution at time t + dt.
    '''
    k1 = np.array( f( x , t , params) )
    k2 = np.array( f( x+dt*(k1/2) , t+(dt/2) , params) )
    k3 = np.array( f( x+dt*(k2/2) , t+(dt/2) , params ) )
    k4 = np.array( f( x+dt*k3 , t+dt , params ) )
    x_next = x +  dt * (k1 + (2*k2) + (2*k3) + k4) / 6
    return x_next


def midpoint_step(f, x, t, dt, **params):
    '''
    One step of the midpoint method.
    
    Parameters: 
    f : (function) - function containing ODE to be solved.
    X : (np.array) - ODE's solution at time t.
    t : (float) - Time to evaluate.
    dt : (float) - step-size.
    **params: Any extra parameters to evaluate ODE.
    
    Returns :    
    Xnew : (np array) - ODE's solution   at t + dt       
    '''
    k1 = np.array(f(x, t, params))
    k2 = np.array(f(x + (dt/2)*k1,t + dt/2, params))
    x_next = x + dt*k2
    return x_next


def heun3_step(f,x,t,dt, **params):
    '''
    One step Heun 3rd order method.

   Parameters: 
   f : (function) - function containing ODE to be solved.
   X : (np.array) - ODE's solution at time t.
   t : (float) - Time to evaluate.
   dt : (float) - step-size.
   
   Returns :    
   Xnew : (np array) - ODE's solution   at t + dt  
    
    Example
    -------     
    Xnew = heun3_step(X=1, t=0, h=0.1, f)
    '''
    # calculate Xnew using formula
    k1 = dt*np.array(f(x, t, params))
    k2 = dt*np.array(f(x+(k1/3), t+(dt/3), params))
    k3 = dt*np.array(f(x+(2*(k2/3)), t+(2*(dt/3)), params))

    x_next = x + k1/4 + 3*k3/4
    
    return x_next

###################### Functions ##################################
def solve_to(f, method, t0, t1, X0,  **params):
    """
   Solves an ODE for an array of values at time t1 given X0 and t0 
   
   Parameters: 
     f : (function) -  ODE to be solved.
     method : Numerical method to use
     t0 : (float) - Initial Time 
     t1 : (float) - Final Time 
     X0 (np.array): the initial value of solution
     dt : (float) - Timestep.
     **params: Any extra parameters to evaluate ODE.
     
     Returns :    
     Xnew : (np array) - ODE's solution on the initial time t0 to the final 
                        time t1. The function returns the solution at time t1.
   """
    
    try:
        dt_max = params['dt_max']
    except KeyError:
        dt_max = 0.1
    # return X1
    while t0 < t1:
        if t0 + dt_max > t1:
            X = method(f, X0, t0, t1-t0, **params)
            t0 = t1
        else:
            X = method(f, X0, t0, dt_max,**params)
            X0 = X
            t0 += dt_max
    return X

def solve_ode(method, f, t, X0, use_method_dict=True, **params):
    """
   Solves an ODE from an initial value, t_0 to a final value t_end
   
   Parameters:
       method :(str or function) Numerical method to use
       f : (function)  The ODE to be solved
       t_values : (array) time values 
       X0 : (numpy array) The initial value of solution
       use_method_dict : (bool) if True, use a dictionary to map the method 
                                   name to the corresponding function
       **params: any extra parameters to be passed to the ODE function.
   Returns:
       X : (numpy array) The solution array
     """
    if use_method_dict:
        method_dict = {
            'euler': euler_step,
            'rk4': RK4_step,
            'midpoint': midpoint_step,
            'heun3': heun3_step 
        }

        if method in method_dict:
            method_func = method_dict[method]
        else:
            raise ValueError(f"Specified incorrect solver: {method}")
    else:
        if method == 'euler':
            method_func = euler_step
        elif method == 'rk4':
            method_func = RK4_step
        elif method == 'heun3':
            method_func = heun3_step
        elif method == 'midpoint':
            method_func = midpoint_step
        else:
            raise ValueError(f"Specified incorrect solver: {method}")
    X0 = np.array(X0)
    if len(X0) > 1:
        X = np.zeros((len(t), X0.shape[0]))
        X[0, :] = X0
    else:
        X = np.zeros(len(t))
        X[0] = X0

    for i in range(len(t) - 1):
        t0 = t[i]
        t1 = t[i + 1]
        X[i + 1] = solve_to(f, method_func, t0, t1, X[i], **params)
    return X

######################### Plots ###########################
def plot_solution(t, X, xlabel='t', ylabel='x', title='Solution', X_true=None):
    """
    Plots the solution to the ODE as a function of time with the specified 
    labels and title. If X_true is specified, the true solution is plotted.
    
    Parameters:
    t: (np.array) an array of time values.
    X: (np.array) an array of solved values.
    xlabel: (str)  the label for the x-axis.
    ylabel: (str)  the label for the y-axis.
    title: (str)  the title for the plot.
    X_true: (np.array)  an array of true solutions (optional).

    """
    if len(X.shape) <= 1:
        number_of_vars = 1
        if X_true is not None:
            X_true = np.array(X_true).transpose()
            plt.plot(t, X_true[:], label='True Solution')
            plt.plot(t, X[:], label='Approx Solution')
        else:
            plt.plot(t, X[:], label='Approx Solution')
    else:
        number_of_vars = X.shape[1]
        if X_true is not None:
            X_true = np.array(X_true).transpose()
            for i in range(number_of_vars):
                plt.plot(t, X_true[:,i], label='True Solution, x'+str(i))
                plt.plot(t, X[:,i], label='Approx Solution, x'+str(i))            
        else:
            for i in range(number_of_vars):
                plt.plot(t, X[:,i], label='Approx Solution, x'+str(i))
    plt.title(title), plt.xlabel(xlabel), plt.ylabel(ylabel), plt.legend()
    plt.show()
    return


def plot_error(methods, f, t0, t1, X0, X1_true, show_plot=True, **params):
    """
    The function plots the error vs. the time step for each numerical method 
    specified in methods.

    Parameters: 
    methods: (list of str or function): a list of numerical methods to be compared.
    f: (function) the ODE function to be solved.
    t0 :(float) the initial time.
    t1 : (float) the final time.
    X0: (np.array) the initial value of solution(s)
    X1_true: (np.array) an array of true dsolutions at the final time.
    show_plot: (bool) if True, show the plot; if False, only return the data.
    **params: any extra parameters to be passed to the ODE function.
    """

    method_errors = []
    X1_true = np.array(X1_true).transpose()

    for method in methods:
        dts = np.logspace(0, -5, 200)
        errors = []

        for dt in dts:
            t = np.linspace(t0, t1, 2)
            X = solve_ode(method, f, t, X0, dt_max=dt, **params)
            
            if len(X0) > 1:
                error = np.mean(np.abs(X[-1] - X1_true))
            else:
                error = np.abs(X[-1] - X1_true)
            errors.append(error)
        plt.loglog(dts, errors, label=str(method), linewidth=2)
        method_errors.append(errors)
    plt.xlabel('Delta t')
    plt.ylabel('Error')
    plt.title('Error Plot')
    plt.legend()

    if show_plot:
        plt.show()

    return method_errors, dts



def desired_tolerance(methods, f, t0, t1, X0, X_true, desired_tol, **params):
    """
    The function finds the minimum time step required to reach the desired 
    tolerance for each numerical method specified. 
    
    Parameters: 
    - methods (list of str or function): a list of numerical methods to be compared.
    - f (function): the ODE function to be solved.
    - t0 (float): the initial time.
    - t1 (float): the final time.
    - X0 (np.array): the initial value of the dependent variable(s).
    - X_true (np.array): an array of true dependent variable values at the final time.
    - desired_tol (float): the desired tolerance for the error.
    - **params: any extra parameters to be passed to the ODE function.

     """
    dt_line = np.full(200,[desired_tol])
    method_errors, hs = plot_error(methods, f, t0, t1, X0, X_true, show_plot=False, **params)
    
    for i, method in enumerate(methods):
        errs = np.array(method_errors[i])

        # Find the intersection between the error lines and the desired tolerance
        intersection = np.argwhere(np.diff(np.sign(dt_line - errs))).flatten()
        print('\nMethod:', method)
        # If method can reach the tolerance, print details - time and required h
        if intersection.size > 0:
            h_required = hs[intersection][0]
            print('dt to meet desired tol:', h_required)
            
            # Time and solve the ODE with the required h
            start_time = time.time()
            t = np.linspace(t0, t1, int((t1 - t0) / h_required) + 1)
            solve_ode(method, f, t, X0, dt_max=h_required, **params)
            end_time = time.time()
            print('Time taken to solve to desired tol: {:.4f}s'.format(end_time - start_time))
            
            # Plot the lines on the error vs h graph
            plt.axvline(h_required, c='r', linestyle='--')
        else:
            print('Method cannot reach desired tol')

    # Plot the desired tolerance line
    plt.axhline(desired_tol, linestyle='--', c='k', label='Desired Tol')
    plt.legend()
    plt.show()

########################### Checks 1st order and 2nd order  #######################
def main():
    
    
    t = np.linspace(0,1,100) 
    X0 = [1] 
    X = solve_ode('midpoint', f, t, X0, dt_max=0.1)
    
    plot_solution(t, X, 't', 'x(t)', 'First Order ODE Solution', X_true=exp(t))
    
    t = np.linspace(0,10,30)
    X0 = [0,1]
    X = solve_ode('midpoint', g, t, X0) 
    plot_solution(t, X[:,0], 'x', 'y', 'Second order ODE', g1(t))
    plot_solution(X[:,0], X[:,1], 'x', 'y', 'y against x')
    methods = ['euler', 'rk4', 'heun3', 'midpoint']
    method_errors, hs = plot_error(methods, g, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))
    desired_tol = 1e-3
    desired_tolerance(methods, f, 0, 1, [1], exp(1), desired_tol)
    
if __name__ == '__main__':
    main()

import numpy as np
import sys
import matplotlib.pyplot as plt
import time
############### Equations ########################

def f(x, t, args):
    return x

def exp(t):
    return np.exp(t)

def g(X, t, args):
    x = X[0]
    y = X[1]
    dxdt = y
    dydt = -x
    X = [dxdt, dydt]
    return X

def g1(t):
    return [np.sin(t), np.cos(t)]


################### Steps ####################
def euler_step(f, x, t, dt, **params):
    dxdt = f(x,t, params)
    x_next = x + dt*np.array(dxdt)
    return x_next

def RK4_step(f, x, t, dt, **params):

    k1 = np.array( f( x , t , params) )
    k2 = np.array( f( x+dt*(k1/2) , t+(dt/2) , params) )
    k3 = np.array( f( x+dt*(k2/2) , t+(dt/2) , params ) )
    k4 = np.array( f( x+dt*k3 , t+dt , params ) )
    x_next = x +  dt * (k1 + (2*k2) + (2*k3) + k4) / 6
    return x_next

def simpson38_step(f, x, t, dt, **params):
        
    k1 = np.array(f(x, dt, params))
    k2 = np.array(f(x + dt* k1/3, t  + dt/3, params ))
    k3 =  np.array(f(x + dt  * (k1 /3 +k2),t + (2/3) * dt, params))
    k4 = np.array(f(x + dt*(k1 -k2 +k3), t + dt,params))
    x_next= x + dt* (k1 + 3*k2 + 3*k3 + k4)/8     
    return x_next

def midpoint_step(f, x, t, dt, **params):
    
    k1 = np.array(f(x, t, params))
    k2 = np.array(f(x + (dt/2)*k1,t + dt/2, params))
    x_next = x + dt*k2
    return x_next

###################### Functions ##################################
def solve_to(f, method, t0, t1, X0,  **params):
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
    if use_method_dict:
        method_dict = {
            'euler': euler_step,
            'rk4': RK4_step,
            'simpson': simpson38_step,
            'midpoint': midpoint_step,
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
        elif method == 'simpson':
            method_func = simpson38_step
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
    methods = ['euler', 'rk4', 'simpson', 'midpoint']
    method_errors, hs = plot_error(methods, g, 0, 1, np.array([0,1]), np.array([np.sin(1), np.cos(1)]))
    desired_tol = 1e-3
    desired_tolerance(methods, f, 0, 1, [1], exp(1), desired_tol)
    
if __name__ == '__main__':
    main()

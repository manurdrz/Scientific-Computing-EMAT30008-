from solode import solve_ode
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

############### Equations ########################

def predator_prey(X, t, params):
    """
    Predator-prey ODE
    
    Parameters:
    X : (list)  initial conditions.
    t : (float) The time to evaluate
    params : (dict)  model parameters.
    
    Returns: 
    X: (list) Gradients computed for given values   
    """
    a = params['a']
    b = params['b']
    d = params['d']

    x = X[0]
    y = X[1]
      
    dxdt = x*(1-x) - (a*x*y) / (d+x)
    dydt = b*y*(1-(y/x))

    X =[dxdt, dydt]
    return X

def pc_predator_prey(X0, **params):
    """
    Predator-prey ODE phase condition. I.E : dx/dt = 0 at t=0
    
    Parameters:
    X0 : (list)  initial conditions.
    params : any parameters to be included in ODE
    
    Returns: 
    dxdt_0 : (list) Gradient at t = 0 
    """
    
    dxdt_0 = predator_prey(X0, 0, params)[0]
    
    return dxdt_0

################## Function ##########################

def root_finding_problem(X0, *data):
   """
   Returns the output of the root-finding problem that needs to be solved
   to obtain the initial conditions and period of a periodic orbit in an ODE 
   system.
   
   Parameters:
       X0 : (list)   initial conditions.
      data : (tuple) tuple of arguments containing  function, phase condition, 
                     and the model parameters.

   Returns: 
   output: (numpy.array)  The output of the root-finding problem.
    
    """
   
   T = X0[-1]    
   X0 = X0[:-1]    
   t = np.linspace(0, T, 3)    

   f, phase_condition, params = data if len(data) == 3 else data + (None,)
   if params is not None:
        solution = solve_ode('rk4', f, t, X0, **params)
   else:
        solution = solve_ode('rk4', f, t, X0)

   if params is not None:
        output = np.append(X0 - solution[-1, :], phase_condition(X0, **params))
   else:
        output = np.append(X0 - solution[-1, :], phase_condition(X0))

   return output



def numerical_shooting(f, phase_condition, X0, T_guess,  **params):
    """
   Returns the initial conditions and period of a periodic orbit in an ODE 
   system using the numerical shooting method.
   
   Parameters: 
   f : (function) The ODE system.
   phase_condition : (function) function that takes the initial conditions 
   and returns the values of some specific variable(s) at the end of the time 
   interval.
   X0 : (list) initial conditions.
   T_guess : (float)  initial guess for the period of the periodic orbit.
   params : any parameters.
       
   """
    X0_with_T = X0.copy()
    X0_with_T.append(T_guess)

    data = (f, phase_condition, params) if params else (f, phase_condition)

    sol = fsolve(root_finding_problem, X0_with_T, args=data)

    if sol[:-1].all() == np.array(X0).all() and sol[-1] == T_guess:
        print('Root Finder Failed, returning empty array...')
        return []

    X0 = sol[:-1]
    T = sol[-1]

    return X0, T


#################### PLOTS ############################

def compare_b_values(b1, b2):
    """
   Compare the solutions of the predator-prey model for different values of parameter b.

   Parameters:
   b1 (float): the lower bound of b
   b2 (float): the upper bound of b
   """

    t_eval = np.linspace(0, 100, 10000)
    deltat_max = 0.001
    b_vals = np.linspace(b1,b2,4)

    sol_b1 = solve_ode('rk4', predator_prey, t_eval, [0.5, 0.5],  a = 1, b =b_vals[0], d = 0.1)
    sol_b2 = solve_ode('rk4', predator_prey, t_eval, [0.5, 0.5],   a = 1, b =b_vals[1], d = 0.1)
    sol_b3 = solve_ode('rk4', predator_prey, t_eval, [0.5, 0.5],   a = 1, b =b_vals[2], d = 0.1)
    sol_b4 = solve_ode('rk4', predator_prey, t_eval, [0.5, 0.5],   a = 1, b =b_vals[3], d = 0.1)
    
    plt.subplot(2, 2, 1)
    plt.title('b = ' + str(b_vals[0]))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b1[:, 1], 'r', label='y'), plt.plot(t_eval, sol_b1[:, 0], 'g', label='x')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.title('b = '+str(b_vals[1]))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b2[:, 1], 'r', label='y'), plt.plot(t_eval, sol_b2[:, 0], 'g', label='x')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.title('b = ' + str(b_vals[2]))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b3[:, 1], 'r', label='y'), plt.plot(t_eval, sol_b3[:, 0], 'g', label='x')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title('b = '+str(b_vals[3]))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b4[:, 1], 'r', label='y'), plt.plot(t_eval, sol_b4[:, 0], 'g', label='x')
    plt.legend()
    
    plt.tight_layout(pad=1)
    plt.show()

    
def plot_orbit(X0, T, f, title, **params):
    """
   Plot the solution of an ODE for given initial conditions and parameters.

   Parameters:
   X0 :  the initial conditions of the ODE
   T : (float) the time duration of the simulation
   f : (function) the function defining the ODE
   title:  (str) the title of the plot
   **params: the parameters of the ODE function

   """
    t = np.linspace(0, T, 1000)
    X = solve_ode('rk4', f, t, X0, **params, h_max=0.001)
    x_data, y_data = X[:, 0], X[:, 1]

    plt.figure()
    plt.plot(t, x_data, label='x')
    plt.plot(t, y_data, label='y')
    plt.xlabel('t')
    plt.ylabel('x, y')
    plt.title(title)
    plt.legend()
    plt.show()


def find_time_period(func, a, b, d, X0, t_span=(0, 200), t_eval_resolution=20000001, slice_indices=(10000000, 14000001), round_precision=3):
    """
    Find the time period of a periodic solution of an ODE.

      Parameters:
      func (function): the function defining the ODE
      a (float): the value of parameter a
      b (float): the value of parameter b
      d (float): the value of parameter d
      X0 (array-like): the initial conditions of the ODE
      t_span (tuple): the time span of the simulation (default (0, 200))
      t_eval_resolution (int): the number of time steps used in the simulation (default 20000001)
      slice_indices (tuple): the indices of the time array used to slice the solution and find the period (default (10000000, 14000001))
      round_precision (int): the number of decimal places used to round the solution values (default 3)
    
      Returns:
      float: the time period of the periodic solution, rounded to 5 decimal places
      """
    t_eval = np.linspace(t_span[0], t_span[1], t_eval_resolution)
    
    sol = solve_ivp(lambda t, X: func(X, t, {'a': a, 'b': b, 'd': d}), t_span, X0, t_eval=t_eval)
    
    solt_slice = sol.t[slice_indices[0]:slice_indices[1]]
    solx_slice = sol.y[0][slice_indices[0]:slice_indices[1]]
    soly_slice = sol.y[1][slice_indices[0]:slice_indices[1]]

    rounded_soly_slice = np.around(soly_slice, round_precision)
    maxy = np.amax(rounded_soly_slice)

    i_long = list(np.asarray(rounded_soly_slice==maxy).nonzero()[0])
    i_long_1 = i_long[:len(i_long)//2]
    i_long_2 = i_long[len(i_long)//2:]

    start_i = i_long_1[(len(i_long_1) - 1)//2]
    end_i = i_long_2[(len(i_long_2) - 1)//2]
    
    plt.plot(solt_slice, soly_slice, 'tab:orange')
    plt.plot(solt_slice[start_i], soly_slice[start_i], 'ro')
    plt.plot(solt_slice[end_i], soly_slice[end_i], 'ro')
    
    plt.title('Finding the period')
    plt.xlabel('t')
    plt.ylabel('y')

    solt_period = solt_slice[start_i:end_i]
    solx_period = solx_slice[start_i:end_i]
    soly_period = soly_slice[start_i:end_i]

    plt.plot(solt_period, solx_period, label='x')
    plt.plot(solt_period, soly_period, label='y')
    plt.title('Plotting the period')
    plt.xlabel('t')
    plt.ylabel('x, y')
    plt.legend()
    plt.show()

    time_period = round(solt_slice[end_i] - solt_slice[start_i], 5)

    return time_period



def main():

    compare_b_values(0.1, 0.5)

    X0, T = numerical_shooting(predator_prey, pc_predator_prey,[1.6, 1.6], 10,  a=1, b=0.25, d=0.1)

    plot_orbit(X0, T, predator_prey, 'Periodic Orbit', a=1, b=0.2, d=0.1)
    

    a = 1
    d = 0.2
    b = 0.23
    X0 = [1, 1]
    time_period = find_time_period(predator_prey, a, b, d, X0)

    print("Isolating the time period of a solution to the predator-prey model\n")
    print(f"Time Period = {time_period} (5 d.p)")
    print("\nConditions:\n",
          f"a = {a}, d = {d}, b = {b}\n",
          f"X0 = {X0}")
    

if __name__ == '__main__':
    main()
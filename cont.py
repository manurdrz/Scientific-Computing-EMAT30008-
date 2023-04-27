import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from shootfinal import numerical_shooting, root_finding_problem


###################### Equations #########################

def cubic(x,params):
    c = params['c']
    return x**3 - x + c

def hopf(X, t, params):
    beta = params['beta']
    u1 = X[0]
    u2 = X[1]
    
    du1dt = beta*u1 - u2 - u1*(u1**2 + u2**2)
    du2dt = u1 + beta*u2 - u2*(u1**2 + u2**2)
    X = [du1dt, du2dt]
    return X
    
def pc_hopf(X0, **params):
    pc = hopf(X0, 0, params)[0]
    return pc
  
def modified_hopf(X, t, params):
    beta = params['beta']
    u1 = X[0]
    u2 = X[1]
    du1dt = beta*u1 - u2 + u1*(u1**2 + u2**2) - u1*(u1**2 + u2**2)**2
    du2dt = u1 + beta*u2 + u2*(u1**2 + u2**2) - u2*(u1**2 + u2**2)**2
    X = [du1dt, du2dt]

    return X

def pc_modified_hopf(X0, **params):
    pc = modified_hopf(X0, 0, params)[0]
    return pc


######################### Functions #######################

def natural_parameter(param_list, sols, param_to_vary, function, discretisation, phase_condition, **params):
    T_guess = 5
    Ts = []

    for i in range(len(param_list)):
        params[param_to_vary] = param_list[i]
        prev_sol = sols[i]

        if discretisation == numerical_shooting:
            try:
                X0, T = numerical_shooting(prev_sol.copy(), T_guess, function, phase_condition, **params)
                sols.append(list(X0))
                Ts.append(T)
            except ValueError:
                print('Root finder did not converge, try different T_guess')
                break
        else:
            root = fsolve(discretisation(function), sols[i], args=params)
            sols.append(root)
    return sols, param_list, Ts


def pseudo_arclength(param_list, param_range, sols, param_to_vary, function, discretisation, Ts, phase_condition, **params):
    param_list = param_list[:2].copy()

    def root_finding(x, *args):
        discretisation, function, u1, u2, p1, p2, param_to_vary, phase_condition, params = args
        u0 = x[:-1]
        p0 = x[-1]
        params[param_to_vary] = p0
        if discretisation == numerical_shooting:
            d = root_finding_problem(u0, function, phase_condition, params)
        else:
            d = discretisation(function(u0, params))
        root = np.append(d, np.dot(np.append(u0, p0) - np.append(u2 + (u2 - u1), p2 + (p2 - p1)), np.append((u2-u1), (p2-p1))))
        return root

    while np.min(param_range) <= param_list[-1] <= np.max(param_range):
        if len(sols) == 1:
            X0, NPCparams, T = natural_parameter(param_list, sols.copy(), param_to_vary, function, discretisation, phase_condition, **params)
            u1 = X0[-1].copy()
            Ts = T
            sols = X0[-2:].copy()

        u1 = np.array(sols[-2])
        u2 = np.array(sols[-1])
        param1 = param_list[-2]
        param2 = param_list[-1]

        if discretisation == numerical_shooting:
            u1 = np.append(u1, Ts[-2])
            u2 = np.append(u2, Ts[-1])
            pred = np.append(u2 + (u2 - u1), param2+(param2-param1))
        else:
            pred = np.append(u2 + (u2 - u1), param2+(param2-param1))

        x = fsolve(root_finding, np.array(pred),
                    args=(discretisation, function, u1, u2, param1, param2, param_to_vary, phase_condition, params))

        sols.append(x[:len(sols[-1])].tolist())
        Ts.append(x[-2])
        param_list = np.append(param_list, x[-1])


    return sols, param_list




def continuation(initial_u, param_to_vary, param_range, no_param_values, function, method='pseudo-arclength',
                      discretisation=None, phase_condition=None, T_guess=5, **params):
    def get_discretisation(disc):
        if disc is None:
            return lambda x: x
        elif disc == 'numerical-shooting':
            return numerical_shooting
        else:
            raise ValueError('Incorrect discretisation entered!')
    def get_solver(method_name):
        if method_name == 'pseudo-arclength':
            return pseudo_arclength
        elif method_name == 'natural-parameter':
            return natural_parameter
        else:
            raise ValueError('Incorrect method specified!')

    param_list = np.linspace(param_range[0], param_range[1], no_param_values)
    sols = [initial_u]
    Ts = [T_guess]
    discretisation = get_discretisation(discretisation)
    solver = get_solver(method)
    if method == 'natural-parameter':
        sols, param_list, T = solver(param_list, sols, param_to_vary, function, discretisation, phase_condition,
                                      **params)
        sols = np.array(sols[1:])
    else:
        sols, param_list = solver(param_list, param_range, sols, param_to_vary, function, discretisation, Ts,
                                  phase_condition, **params)
        sols = np.array(sols)
    return sols, param_list


##################### PLOTS ##########################################


def plot_continuation_excercise_results(continuation):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Natural Parameter Continuation - Cubic
    u, p = continuation([1], 'c', [-2, 2], 50, cubic, method='natural-parameter', c=-2)
    axes[0, 0].plot(p, u, label='Natural')
    axes[0, 0].set_xlabel('c')
    axes[0, 0].set_ylabel('Root Location')
    axes[0, 0].set_title('Natural Parameter Continuation')

     # Pseudo-Arclength Continuation - Cubic
    u, p = continuation([1], 'c', [-2, 2], 50, cubic, method='pseudo-arclength', c=-2)
    axes[0, 1].plot(p, u, label='PA')
    axes[0, 1].set_xlabel('c')
    axes[0, 1].set_ylabel('Root Location')
    axes[0, 1].set_title('Pseudo-Arclength Continuation')
     
     # Natural Parameter Continuation -  Hopf
    u, p = continuation([1.5, -1.5], 'beta', [2, 0], 80, hopf, 'natural-parameter', 'numerical-shooting', phase_condition=pc_hopf, T_guess=2, beta=1)
    axes[1, 0].plot(p, u[:, 0], label='Natural')
    axes[1, 0].set_xlabel('beta')
    axes[1, 0].set_ylabel('u1')
    axes[1, 0].set_title('Natural Parameter Continuation -  Hopf')
     
     # Pseudo-Arclength Continuation -  Hopf
    u, p = continuation([1.5, -1.5], 'beta', [2, 0], 80, hopf, 'pseudo-arclength', 'numerical-shooting', phase_condition=pc_hopf, T_guess=2, beta=2)
    axes[1, 1].plot(p, u[:, 0], label='PA')
    axes[1, 1].set_xlabel('beta')
    axes[1, 1].set_ylabel('u1')
    axes[1, 1].set_title('Pseudo-Arclength Continuation -  Hopf')
     
     # Natural Parameter Continuation - Modified Hopf
    u, p = continuation([1, 1], 'beta', [2, -1], 50, modified_hopf, 'natural-parameter', 'numerical-shooting', phase_condition=pc_modified_hopf, T_guess=6, beta=-1)
    axes[2, 0].plot(p, u[:,0], label='Natural')
    axes[2, 0].set_xlabel('beta')
    axes[2, 0].set_ylabel('u1')
    axes[2, 0].set_title('Natural Parameter Continuation - Modified Hopf')
     
     # Pseudo-Arclength Continuation - Modified Hopf
    u, p = continuation([1, 1], 'beta', [2, -1], 50, modified_hopf, 'pseudo-arclength', 'numerical-shooting', phase_condition=pc_modified_hopf, T_guess=6, beta=-1)
    axes[2, 1].plot(p, u[:, 0], label='PA')
    axes[2, 1].set_xlabel('beta')
    axes[2, 1].set_ylabel('u1')
    axes[2, 1].set_title('Pseudo-Arclength Continuation - Modified Hopf')
     
    plt.tight_layout()
    plt.show()


    plt.tight_layout()
    plt.show()

##################### EXAMPLES ##################################
def main():
    # # Example unique plot:
    # u, p = continuation([1], 'c', [-2, 2], 50, cubic, method='natural-parameter', c=-2)
    # plt.plot(p, u, label='Natural')
    plot_continuation_excercise_results(continuation)
        
      
if __name__ == '__main__':
    main()
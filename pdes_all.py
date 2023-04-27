import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D

###################### METHODS  ##################################


def forward_euler(pde, final_space_value, lmbda, num_of_x, num_t, bound_cond, p_func, q_func):
    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)
    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_t, num_of_x + 1))
    solution_matrix[0] = u_vect  
    # Check boundary conditions
    if bound_cond == 'dirichlet':
       main_diag = [1 - 2 * lmbda] * (num_of_x - 1)
       off_diag = [lmbda] * (num_of_x - 2)
       A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
       additive_vector = np.zeros(num_of_x - 1)
       for i in range(0,num_t -1):
           additive_vector[0] = p_func(i)
           additive_vector[-1] = q_func(i)
           solution_matrix[i+1][1:-1] = np.dot(A, solution_matrix[i][1:-1]) + lmbda*additive_vector
           solution_matrix[i+1][0] = additive_vector[0]
           solution_matrix[i+1][-1] = additive_vector[-1]

    elif bound_cond == 'neumann':
        main_diag = [1 - 2*lmbda] * (num_of_x + 1)
        off_diag = [lmbda] * num_of_x
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)
        for i in range(0,num_t -1):
            additive_vector[0] = -p_func(i)
            additive_vector[-1] = q_func(i)
            solution_matrix[i+1] = np.dot(A, solution_matrix[i]) + 2*deltax*lmbda*additive_vector
            
    elif bound_cond == 'periodic':
        main_diag = [1-2*lmbda] * num_of_x
        off_diag = [lmbda] * (num_of_x - 1)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        A[0, -1] = lmbda
        A[-1, 0] = lmbda
        solution_matrix = np.zeros((num_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]
        for i in range(0,num_t -1):
            solution_matrix[i+1] = np.dot(A, solution_matrix[i])

    else:# homgenous
        main_diag = [1-2*lmbda] * (num_of_x - 1)
        off_diag = [lmbda] * (num_of_x - 2)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        for i in range(0,num_t -1):
            solution_matrix[i+1][1: -1] = np.dot(A, solution_matrix[i][1: -1])
    return solution_matrix
   
def backward_euler(pde, final_space_value, lmbda, num_of_x, num_t, bound_cond, p_func, q_func):
    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)
    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_t, num_of_x + 1))
    solution_matrix[0] = u_vect  
    # Check boundary conditions
    if bound_cond == 'dirichlet':
       main_diag = [1 + 2 * lmbda] * (num_of_x - 1)
       off_diag = [-lmbda] * (num_of_x - 2)
       A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
       additive_vector = np.zeros(num_of_x - 1)
       for i in range(0,num_t -1):
           additive_vector[0] = p_func(i)
           additive_vector[-1] = q_func(i)
           solution_matrix[i+1][1:-1] = np.linalg.solve(A, solution_matrix[i][1:-1] + lmbda*additive_vector)
           solution_matrix[i+1][0] = additive_vector[0]
           solution_matrix[i+1][-1] = additive_vector[-1]

    elif bound_cond == 'neumann':
        main_diag = [1 + 2*lmbda] * (num_of_x + 1)
        off_diag = [-lmbda] * num_of_x
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)
        for i in range(0,num_t -1):
            additive_vector[0] = -p_func(i+1)
            additive_vector[-1] = q_func(i+1)
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i] + 2*deltax*lmbda*additive_vector)
            
    elif bound_cond == 'periodic':
        main_diag = [1+2*lmbda] * num_of_x
        off_diag = [-lmbda] * (num_of_x - 1)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        A[0, -1] = -lmbda
        A[-1, 0] = -lmbda
        solution_matrix = np.zeros((num_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]
        for i in range(0,num_t -1):
            solution_matrix[i+1] = np.linalg.solve(A, solution_matrix[i])
    else:# homogenous
        main_diag = [1-2*lmbda] * (num_of_x - 1)
        off_diag = [-lmbda] * (num_of_x - 2)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        for i in range(0,num_t -1):
            solution_matrix[i+1][1: -1] = np.linalg.solve(A, solution_matrix[i][1: -1])
    return solution_matrix
        
def crank_nicholson(pde, final_space_value, lmbda, num_of_x, num_t, bound_cond, p_func, q_func):
    # Evaluate initial solution values
    u_vect = np.linspace(0, final_space_value, num_of_x + 1)
    for i in range(num_of_x + 1):
        u_vect[i] = pde(u_vect[i], final_space_value)
    # Initialise the matrix of solutions with the initial solution values
    solution_matrix = np.zeros((num_t, num_of_x + 1))
    solution_matrix[0] = u_vect  
    # Check boundary conditions
    if bound_cond == 'dirichlet':
       main_diag = [1 + lmbda] * (num_of_x - 1)
       main_diag2 = [- lmbda] * (num_of_x - 1)
       off_diag = [-lmbda / 2] * (num_of_x - 2)
       off_diag2 = [lmbda / 2] * (num_of_x - 2)
       A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
       B = np.diag(main_diag2) + np.diag(off_diag2, -1) + np.diag(off_diag2, 1)
       additive_vector = np.zeros(num_of_x - 1)
       for i in range(0,num_t -1):
           additive_vector[0] = p_func(i)
           additive_vector[-1] = q_func(i)
           solution_matrix[i+1][1:-1] = np.linalg.solve(A, np.dot(B,solution_matrix[i][1:-1]) + lmbda*additive_vector)#original pone np.dot pero lo dejo en np.linal.solve
           solution_matrix[i+1][0] = additive_vector[0]
           solution_matrix[i+1][-1] = additive_vector[-1]

    elif bound_cond == 'neumann':
        main_diag = [1 + lmbda] * (num_of_x + 1)
        main_diag2 = [1 - lmbda] * (num_of_x + 1)
        off_diag = [-lmbda / 2] * (num_of_x )
        off_diag = [lmbda / 2] * (num_of_x )
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        B = np.diag(main_diag2) + np.diag(off_diag2, -1) + np.diag(off_diag2, 1)
        A[0, 1] = 2*A[0, 1]
        A[-1, -2] = 2*A[-1, -2]
        B[0, 1] = 2*A[0, 1]
        B[-1, -2] = 2*A[-1, -2]
        deltax = final_space_value / num_of_x
        additive_vector = np.zeros(num_of_x + 1)
        for i in range(0,num_t -1):
            additive_vector[0] = -p_func(i+1)
            additive_vector[-1] = q_func(i+1)
            solution_matrix[i+1] = np.linalg.solve(A, np.dot(B,solution_matrix[i]) + 2* deltax*lmbda*additive_vector)
            
    elif bound_cond == 'periodic':
        main_diag = [1 + lmbda] * (num_of_x )
        main_diag2 = [1 - lmbda] * (num_of_x)
        off_diag = [-lmbda / 2] * (num_of_x -1 )
        off_diag = [lmbda / 2] * (num_of_x - 1)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        B = np.diag(main_diag2) + np.diag(off_diag2, -1) + np.diag(off_diag2, 1)
        A[0, -1] = - lmbda/2
        A[-1, -0] = - lmbda/2
        B[0, -1] =  lmbda/2
        B[-1, 0] =  lmbda/2
        solution_matrix = np.zeros((num_t, num_of_x))
        solution_matrix[0] = u_vect[:-1]
        for i in range(0,num_t -1):
            solution_matrix[i+1] = np.linalg.solve(A,  np.dot(B,solution_matrix[i]))
    else:# homogenous
        main_diag = [1 + lmbda] * (num_of_x -1)
        main_diag2 = [1 - lmbda] * (num_of_x -1)
        off_diag = [-lmbda / 2] * (num_of_x -2 )
        off_diag = [lmbda / 2] * (num_of_x - 2)
        A = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        B = np.diag(main_diag2) + np.diag(off_diag2, -1) + np.diag(off_diag2, 1)
        for i in range(0,num_t -1):
            solution_matrix[i+1][1: -1] = np.linalg.solve(A, np.dot(B,solution_matrix[i][1:-1]))
    return solution_matrix

##################### SOLVER ########################################

def pde_solver(L, T, Nx, Nt, initial_condition, boundary_condition, diffusion_coefficient, rhs_function):
    # Discretize spatial and temporal domains
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, T, Nt + 1)
    dx = L / Nx
    dt = T / Nt

    u = np.zeros((Nx + 1, Nt + 1))
    u[:, 0] = initial_condition(x)

    for n in range(1, Nt + 1):
        u[0, n], u[-1, n] = boundary_condition(t[n])
        for i in range(1, Nx):
            D = diffusion_coefficient(x[i])
            R = rhs_function(x[i], t[n - 1])
            u[i, n] = u[i, n - 1] + dt * (D * (u[i + 1, n - 1] - 2 * u[i, n - 1] + u[i - 1, n - 1]) / dx**2 + R)
    
    # 2D plot of the solution at the final time step
    plt.plot(x, u[:, -1])
    plt.xlabel('x')
    plt.ylabel('u(x, T)')
    plt.title('Solution at final time step')
    plt.show()

    # 3D surface plot of the solution
    X, T = np.meshgrid(x, t, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, u, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    ax.set_title('Solution surface plot')
    plt.show()
    return u

def visualize_solution(solution_matrix, final_space_value, final_time_value):
    num_of_x, num_of_t = solution_matrix.shape
    x_values = np.linspace(0, final_space_value, num_of_x)
    t_values = np.linspace(0, final_time_value, num_of_t)

    X, T = np.meshgrid(x_values, t_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, solution_matrix.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    plt.show()

#########################################
def main():

    def initial_condition_1(x):
        return np.sin(np.pi * x)

    def initial_condition_2(x):
        L = 1.0
        return x * (L - x)
    
    def boundary_condition(t):
       return 0, 0
    
    # Define the diffusion coefficient function
    def diffusion_coefficient(x):
        return 1
    
    # Define the right-hand side (RHS) function
    def rhs_function(x, t):
        return 0

    # Example 1: Forward Euler with variable diffusion coefficient and non-homogeneous RHS function
    Nx1 = 100
    Nt1 = 100
    L1 = 1
    T1 = 0.1

    solution_matrix_1 = pde_solver(L1, T1, Nx1, Nt1, initial_condition_1, boundary_condition, diffusion_coefficient, rhs_function)
    visualize_solution(solution_matrix_1, L1, T1)

    plt.plot(np.linspace(0, L1, Nx1 + 1), solution_matrix_1[0])
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Line plot at final time (Example 1)')
    plt.show()

     # Example 2: Backward Euler with variable diffusion coefficient and non-homogeneous RHS function
    Nx2 = 100
    Nt2 = 100
    L2 = 1
    T2 = 0.1

    solution_matrix_2 = pde_solver(L2, T2, Nx2, Nt2, initial_condition_2, boundary_condition, diffusion_coefficient, rhs_function)
    visualize_solution(solution_matrix_2, L2, T2)

    plt.plot(np.linspace(0, L2, Nx2 + 1), solution_matrix_2[0])
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Line plot at final time (Example 2)')
    plt.show()
    
if __name__ == '__main__':
    main()
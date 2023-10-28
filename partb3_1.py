import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Define the Hopfield network dynamics
def hopfield_network(t,x):
    W = np.array([[0, 1], [1, 0]])
    b = np.array([0, 0])
    sigmoid = (1 - np.exp(-100 * x)) / (1 + np.exp(-100 * x))
    dxdt = -0.5 * x + 0.5 * np.dot(W.T, sigmoid) + 0.5 * b
    return dxdt


# Create a grid of initial conditions
x1 = np.linspace(-1, 1, 20)
x2 = np.linspace(-1, 1, 20)
X1, X2 = np.meshgrid(x1, x2)

# Numerically integrate the system for various initial conditions
for x1_0, x2_0 in zip(X1.flatten(), X2.flatten()):
    initial_conditions = [x1_0, x2_0]
    solution = solve_ivp(hopfield_network, [0, 10], initial_conditions, t_eval=np.linspace(0, 10, 100))

    # Plot the trajectory
    plt.plot(solution.y[0], solution.y[1])

###Second way:
# traj = []
# xt=0
# h=0.1
# W = np.array([[0, 1], [1, 0]])
# b = np.array([0, 0])
# sigmoid = (1 - np.exp(-100 * xt)) / (1 + np.exp(-100 * xt))
# dxdt = -0.5 * xt + 0.5 * np.dot(W.T, sigmoid) + 0.5 * b
# for x1_0, x2_0 in zip(X1.flatten(), X2.flatten()):
#     tmp = []
#     for t in range (200):
#         xt += h*dxdt
#         tmp.append(xt)
#     traj.append(tmp)
# traj = np.array(traj)


# Set axis limits and labels
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('x1')
plt.ylabel('x2')

# Show the plot
plt.show()

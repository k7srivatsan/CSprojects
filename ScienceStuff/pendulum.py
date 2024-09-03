import numpy as np
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def stepODE(t, y, g, L1, L2, m1, m2):
    [theta1, omega1, theta2, omega2] = y
    dydt = [omega1, 
            (-g * (2 * m1 + m2) * math.sin(theta1) - m2 * g * math.sin(theta1 - 2 * theta2) - 2 * math.sin(theta1 - theta2) * m2 * (omega2 ** 2 * L2 + omega1 ** 2 * L1 * math.cos(theta1 - theta2))) / (L1 * (2 * m1 + m2 - m2 * math.cos(2 * theta1 - 2 * theta2))),
            omega2, 
            (2 * math.sin(theta1 - theta2) * (omega1 ** 2 * L1 * (m1 + m2) + g * (m1 + m2) * math.cos(theta1) + omega2 ** 2 * L2 * m2 * math.cos(theta1 - theta2))) / (L2 * (2 * m1 + m2 - m2 * math.cos(2 * theta1 - 2 * theta2))),
            ]
    return dydt


g = 9.81 # m/s^2
L1 = 4 # m
L2 = 4
m1 = 5 # kg
m2 = 5
theta0m1 = math.pi / 2
theta0m2 = math.pi / 3
omega0m1  = -0.2
omega0m2 = -0.2

initCond = [theta0m1, omega0m1, theta0m2, omega0m2] # theta0, omega0
x1 = L1 * math.sin(theta0m1)
y1 = -(L1 * math.cos(theta0m1))
x2 = x1 + (L2 * math.sin(theta0m2))
y2 = x2 + -(L2 * math.cos(theta0m2))

t_eval = np.arange(0, 10, 0.01)
solve = solve_ivp(stepODE, t_span=[0, 10], y0=initCond, args=(g, L1, L2, m1, m2), rtol=1e-8, t_eval=t_eval)
T = solve.t
thetas = solve.y # first row has theta values corresponding to t-steps

X1 = L1 * np.sin(thetas[0, :])
Y1 = -(L1 * np.cos(thetas[0, :]))

X2 = X1 + (L2 * np.sin(thetas[2, :]))
Y2 = Y1 + -(L2 * np.cos(thetas[2, :]))
fig = plt.figure()

ax = fig.add_subplot(aspect='equal')
line1, = ax.plot([0, x1], [0, y1], lw=2, c='k')
# The pendulum bob: set zorder so that it is drawn over the pendulum rod.
bob_radius = 0.08
circle1 = ax.add_patch(plt.Circle((x1, y1), bob_radius,
                      fc='r', zorder=3))
line2, = ax.plot([x1, x2], [y1, y2], lw=2, c='k')
circle2 = ax.add_patch(plt.Circle((x2, y2), bob_radius,
                      fc='r', zorder=3))
# Set the plot limits so that the pendulum has room to swing!
ax.set_xlim(-(L1 + L2 + 1), L1 + L2 + 1)
ax.set_ylim(-(L1 + L2 + 1), L1 + L2 + 1)
global paths
paths, = ax.plot([], [], lw=1)
paths.set_data([], [])
global pathsx, pathsy
pathsx = []
pathsy = []

def animate(i):
    """Update the animation at frame i."""
    x1, y1 = X1[i], Y1[i]
    line1.set_data([0, x1], [0, y1])
    circle1.set_center((x1, y1))

    x2, y2 = X2[i], Y2[i]
    line2.set_data([x1, x2], [y1, y2])
    circle2.set_center((x2, y2))

    pathsx.append(X2[i])
    pathsy.append(Y2[i])
    paths.set_data(pathsx, pathsy)

nframes = len(X1)
diffT = [T[i + 1] - T[i] for (i, t) in enumerate(T[:-1])]
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=False,
                              interval=10)
plt.show()
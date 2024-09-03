import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import importlib
mpl_toolkits = importlib.import_module('mpl_toolkits')
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp

L = 30
n = 32
sample_spacing = L / n
alpha = 0.5

u0 = np.zeros((n, n))
u0[int(n/8) : int(3*n/8), int(5*n/8) : int(7*n/8)] = 100
x = np.arange(-L/2, L/2, L / n) # space
y = np.arange(-L/2, L/2, L / n)
X, Y = np.meshgrid(x, y)

dt = 0.1
t = np.arange(0, 20, dt) # time 

u0hatFFT = np.fft.fft2(u0)
kappas = 2 * math.pi * np.fft.fftfreq(n=n) # convert from linear frequencies to angular frequencies
kappasSquaredSum = np.array([[kappas[i] ** 2 + kappas[j] ** 2 for j in range(n)] for i in range(n)])

def stepTime(t, u, alpha):
    dudt = np.multiply(-(alpha ** 2) * kappasSquaredSum, np.reshape(u, (n, n)))
    return dudt.flatten()

uhatFFT = solve_ivp(stepTime, t_span=(0, 20), y0=u0hatFFT.flatten(), t_eval=t, args=(alpha,))
uhat = uhatFFT.y
uhat = uhat.reshape((n, n, -1))
print(uhat.shape)
for i in range(uhat.shape[-1]):
    uhat[:, :, i] = np.real(np.fft.ifft2(uhat[:, :, i]))
uhat = np.float16(uhat)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.view_init(20, -55, 0)
ax.set_xlim((-L/2, L/2))
ax.set_ylim((-L/2, L/2))
ax.set_zlim(-1, 100)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Heat Distribution")

ax.plot_surface(X, Y, uhat[:, :, 0], cmap=cm.coolwarm)

def animate(i):
    """Update the animation at frame i."""
    ax.cla()
    ax.set_xlim((-L/2, L/2))
    ax.set_ylim((-L/2, L/2))
    ax.set_zlim(-1, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Heat Distribution")

    ax.plot_surface(X, Y, uhat[:, :, i], cmap=cm.coolwarm)
    ax.text(s=f"{i * dt:.2f} seconds", x=0, y=10, z=4)

nframes = uhat.shape[-1]
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                              interval=100)


plt.show()
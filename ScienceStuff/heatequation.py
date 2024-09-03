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
n = 256
sample_spacing = L / n
alpha = 10

u0 = np.zeros(n)
u0[int(n/2) : int(15*n/16)] = 100
x = np.arange(-15, 15, L / n) # space
y = np.arange(0, 20, 0.01) # time 
X, Y = np.meshgrid(x, y)

u0hatFFT = np.fft.fft(u0, n=n)
kappas = 2 * math.pi * np.fft.fftfreq(n=n)

def stepTime(t, u, alpha):
    dydt = - (alpha ** 2) * (kappas ** 2) * u
    return dydt

uhatFFT = solve_ivp(stepTime, t_span=(0, 20), y0=u0hatFFT, t_eval=y, args=(alpha,))
uhat = uhatFFT.y

for i in range(uhat.shape[1]):
    uhat[:, i] = np.real(np.fft.ifft(uhat[:, i]))
uhat = np.float16(uhat)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.view_init(20, -55, 0)
ax.set_xlim((-L/2, L/2))
ax.set_ylim(0, 20)
ax.set_zlim(-1, 100)
ax.set_xlabel("Space")
ax.set_ylabel("Time (seconds)")
ax.set_zlabel("Heat Distribution")

cmap = matplotlib.colormaps.get_cmap('coolwarm')
color = cmap(uhat[:, 0] / 100)[:, :4]
ax.scatter(x, 0, uhat[:, 0], c=color)

def animate(i):
    """Update the animation at frame i."""
    ax.cla()
    ax.set_xlim((-L/2, L/2))
    ax.set_ylim(0, 20)
    ax.set_zlim(-1, 100)
    ax.set_xlabel("Space coordinates")
    ax.set_ylabel("Time")
    ax.set_zlabel("Heat Distribution")

    color = cmap(uhat[:, i] / 100)[:, :4]
    ax.scatter(x, i * 0.01, uhat[:, i], c=color)
    ax.text(s=f"{i * 0.01:.2f} seconds", x=0, y=10 + (i * 0.01), z=4)

nframes = uhat.shape[1]
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True,
                              interval=10)

# pcm = ax2.pcolormesh(np.float16(uhat), cmap=plt.cm.jet)
# plt.colorbar(pcm, ax=ax2)
# ax2.set_xlabel("Time (seconds)")
# ax2.set_ylabel("Space")
# fig.subplots_adjust(wspace=1)
plt.show()

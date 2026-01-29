import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.animation import FuncAnimation, PillowWriter


def update(frame):
    ax.view_init(elev=34, azim=frame)
    return fig,


print(os.getcwd())

df = pd.read_csv("grid_lhs_constrained_final_choice.csv")

print(df.head())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#######################  Plot Data  #######################

ax.scatter(df["Omega_m"], df["Omega_b"], df["Hubble_parameter"], c='blue', alpha=0.6, label='Samples')

#######################  Plot Real Universe Point  #######################

ax.scatter([0.3089], [0.0486], [0.6774], c='green', alpha=0.6, label='Real Universe')

####################### Plot Surface #######################
# Einschränkungen
a_min, a_max = 0.1, 0.4                 # A = Omega_cold-darkmatter
b_min, b_max = 0.001, 0.1               # B = Omega_baryon
c_min, c_max = 0.55, 0.85                # C = Omega_Lambda

# Achsen
ax.set_xlabel(r'$\Omega_m$')
ax.set_ylabel(r'$\Omega_b$')
ax.set_zlabel(r'$H_0$')
#ax.set_title(r'Latin Hypercube Sampling with constraint'
#             r' of $\Omega_m$+$\Omega_\Lambda$=1 $\wedge$ $\Omega_m \geq \Omega_b$' + '\n'
#             r'and $\Omega_m$ $\in$ [0.1, 0.5], $\Omega_b$ $\in$ [0.001, 0.1], $\Omega_\Lambda$ $\in$ [0.5, 0.9]')

# Achsengrenzen
ax.set_xlim(0.1, 0.5)
ax.set_ylim(0.001, 0.1)
ax.set_zlim(0.55, 0.85)

plt.legend()
plt.tight_layout()

# Setze Kameraperspektive
ax.view_init(elev=34, azim=17)

# Speichern
frames = np.linspace(0, 360, 360)  # smooth full rotation

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=40,   # ms between frames
    blit=False
)

writer = PillowWriter(fps=25)  # loop=0 → infinite loop
ani.save("plots/LHS_constrained_rotating_3d_plot.gif", writer=writer, dpi=200)
plt.show()

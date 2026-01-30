import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
print(os.getcwd())

df = pd.read_csv("grid_lhs_constrained_final_choice.csv")

print(df.head())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#######################  Plot Data  #######################

ax.scatter(df["Omega_m"], df["Omega_b"], df["Omega_Lambda"], c='blue', alpha=0.6, label='Samples')

#######################  Plot Real Universe Point  #######################

ax.scatter([0.3089], [0.0486], [0.6911], c='green', alpha=0.6, label='Real Universe')

####################### Plot Surface #######################
# Einschränkungen
a_min, a_max = 0.1, 0.4                 # A = Omega_cold-darkmatter
b_min, b_max = 0.001, 0.1               # B = Omega_baryon
c_min, c_max = 0.5, 0.9                 # C = Omega_Lambda

# Auflösung
n_grid = 100

# Gitter erzeugen
a_vals = np.linspace(a_min, a_max, n_grid)
b_vals = np.linspace(b_min, b_max, n_grid)
A, B = np.meshgrid(a_vals, b_vals)
C = 1 - A - B

A = A + B

# Ungültige Bereiche maskieren (wo c < 0 → np.nan)
C[(C > c_max) | (C < c_min)] = np.nan

# Glatte Fläche ohne Gitterstruktur
ax.plot_surface(A, B, C, alpha=0.4, color='gray', rstride=1, cstride=1, edgecolor='none')

####################### Set Options for Plot #######################

# Achsen
ax.set_xlabel(r'$\Omega_m$')
ax.set_ylabel(r'$\Omega_b$')
ax.set_zlabel(r'$\Omega_\Lambda$')
#ax.set_title(r'Latin Hypercube Sampling with constraint'
#             r' of $\Omega_m$+$\Omega_\Lambda$=1 $\wedge$ $\Omega_m \geq \Omega_b$' + '\n'
#             r'and $\Omega_m$ $\in$ [0.1, 0.5], $\Omega_b$ $\in$ [0.001, 0.2], $\Omega_\Lambda$ $\in$ [0.5, 0.9]')

# Achsengrenzen
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 0.15)
ax.set_zlim(0.5, 1)

plt.legend()
plt.tight_layout()

# Setze Kameraperspektive
ax.view_init(elev=34, azim=17)

# Speichern
plt.savefig("plots/LHS_constrained_sampling.pdf", format="PDF", dpi=300)
plt.show()

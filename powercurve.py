"""
This is the file for the power curve.
"""

__author__ = "Astrid Myren, UiB"
__email__ = "astrid.myren@student.uib.no"

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

a = np.linspace(0, 1, 100)
Cp = 4 * a * (1 - a) ** 2
plt.plot(a, Cp)
plt.title('Cp for different values of a')
plt.ylabel('Power coefficient: Cp')
plt.xlabel('Axial induction factor: a')
plt.show()

rho = 1.2
d = 90 #m
A = 1/4*np.pi*d**2
V = np.arange(0, 15, 0.1)
cut_in = 3 #m/s
cut_out = 25 #m/s
rated = 10 #m/s

for Cp in np.linspace(0, .59, 10):
    P = 0.5*A*rho*V**3*Cp/1000
    # cut-in speed
    P[V < cut_in] = 0
    # cut-out speed
    P[V >= cut_out] = 0
    # rated speed
    P[(V > rated) & (V < 25)] = 0.5 * Cp * A * rho * rated ** 3/1000

    plt.plot(V, P, label=round(Cp, 2))

plt.title('Power curve for different values of Cp')
plt.ylabel('Power (kW)')
plt.xlabel('Wind speed (m/s)')
plt.legend()
plt.show()

for d in np.arange(70, 110, 5):
    Cp = 0.35
    A = 1 / 4 * np.pi * d ** 2
    P = 0.5*A*rho*V**3*Cp/1000
    #cut in
    P[V < cut_in] = 0
    # cut-out speed
    P[V >= cut_out] = 0
    # rated speed
    P[(V > rated) & (V < 25)] = 0.5 * Cp * A * rho * rated ** 3/1000
    plt.plot(V, P, label=round(d, 2))

plt.title('Power curve for different rotor diameters. Cp=0.35')
plt.ylabel('Power (kW)')
plt.xlabel('Wind speed (m/s)')
plt.legend()
plt.show()
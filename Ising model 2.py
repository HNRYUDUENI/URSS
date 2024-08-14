#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, generate_binary_structure



N = 100
prob = 0.75


states_u = np.random.choice([1, -1], (N,N), p = [prob, 1 - prob]).astype(np.float64)
states_d = np.random.choice([-1, 1], (N,N), p = [prob, 1 - prob]).astype(np.float64)

def energy(lattice):
    rows, cols = lattice.shape
    E = 0
    neighbors = [(1,0), (0,1), (-1,0), (0,-1)]
    for x in range(rows):
        for y in range(cols):
            for a, b in neighbors:
                x_neighbors = (x + a) % rows
                y_neighbors = (y + b) % cols
                E += lattice[x, y] * lattice[x_neighbors, y_neighbors]
    E = -E
    return E

def energy_2 (lattice):
    kern = generate_binary_structure(2,1).astype(int)
    kern[1,1] = False
    arr = -lattice * convolve(lattice, kern,mode = "wrap", cval = 0)
    E = np.sum(arr)
    return E
    
Energy_u = energy_2(states_u)
Energy_d = energy_2(states_d)
print(f"energy up spin: {energy(states_u)}")
print(f"energy down spin: {energy(states_d)}")
plt.imshow(states_u)
plt.show()


@numba.jit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8)", nopython = True, parallel = True, nogil = True)
def metropolis(lattice, times, B, energy):
    lattice = lattice.copy()
    net_spin = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    for t in range(0, times - 1):
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = lattice[x, y]
        spin_f = spin_i * -1
        
        E_i = 0
        E_f = 0
        
        if x > 0 :
            E_i += -spin_i * lattice[x - 1, y]
            E_f += -spin_f * lattice[x - 1, y]
        if x < N - 1:
            E_i += -spin_i * lattice[x + 1, y]
            E_f += -spin_f * lattice[x + 1, y]
        if y > 0:
            E_i += -spin_i * lattice[x, y - 1]
            E_f += -spin_f * lattice[x, y - 1]
        if y < N - 1:
            E_i += -spin_i * lattice[x, y + 1]
            E_f += -spin_f * lattice[x, y + 1]
        dE = E_f - E_i
        if (dE > 0) * (np.random.random() < np.exp(-B * dE)):
            lattice[x, y] = spin_f
            energy += dE
        elif dE <= 0:
            lattice[x, y] = spin_f
            energy += dE
        net_spin[t] = lattice.sum()
        net_energy[t] = energy
        
        
    return net_spin, net_energy

spins, energies = metropolis(states_u, 1000000, 0.7, Energy_u)


fig, axes = plt.subplots(1, 2, figsize = (12, 5))
ax = axes[0]
ax.plot(spins / N**2)
ax.set_xlabel("average time step")
ax.set_ylabel("net spin")
ax.grid()
ax = axes[1]
ax.plot(energies)
ax.set_xlabel("average time step")
ax.set_ylabel("net energy")
ax.grid()
fig.tight_layout()
fig.suptitle("evelolution of spin and energ", y = 1.07, size = 18)
plt.show()


# In[3]:


import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt


def get_spin_energy(lattice, B):
    ms = np.zeros(len(B))
    E_mean = np.zeros(len(B))
    E_std = np.zeros(len(B))
    for i, b in enumerate(B):
        spins, energies = metropolis(lattice, 1000000, b, energy_2(lattice))
        ms[i] = spins[-100000] / N**2
        E_mean[i] = energies[-100000].mean()
        E_std[i] = energies[-100000].std()
    return ms, E_mean, E_std
Bs = np.arange(0.2, 2, 0.05)
ms_u, E_mean_u, E_std_u = get_spin_energy(states_u, Bs)
ms_d, E_mean_d, E_std_d = get_spin_energy(states_d, Bs)

plt.figure(figsize = (8,4))
plt.plot(1/Bs, ms_u)
plt.plot(1/Bs, ms_d)
plt.show()
        


# In[2]:


import numpy as np
import numba
from numba import jit
import matplotlib.pyplot as plt



plt.figure(figsize = (8,4))
plt.plot(1/Bs, E_mean_u)
plt.plot(1/Bs, E_mean_d)
plt.show()


# In[ ]:





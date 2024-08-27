#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import jit
from scipy.ndimage import convolve, generate_binary_structure


N = 100
prob_up = 0.75
prob_down = 0.75
states_u = np.random.choice([1,-1], (N,N), p = [prob_up, 1 - prob_up])
states_d = np.random.choice([-1,1], (N,N), p = [prob_down, 1 - prob_down])


def RDF (lattice,x , y ,r):
    size = lattice.shape[0]
    neighbors = []
    for dx in range(-r, r):
        for dy in range(-r, r):
            if dx == 0 and dy ==0 :
                continue
            distance = np.sqrt(dx**2 + dy**2)
            if distance <= r:
                nx, ny = (x + dx) % size, (y + dy)% size
                neighbors.append((lattice[nx, ny], distance))
    return neighbors
        
#print(RDF(states_d,3, 7, 5))

def GofR(lattice,x,y, r):
    distances = []              
    neighbors = RDF(lattice, x, y, r)
    spin = lattice[x, y]
    for neighbors_spin, distance in neighbors:
         if neighbors_spin == spin:
                
                distances.append(distance)
    return distances
        
def count_lattice_points_in_annulus(r_inner, r_outer):
    count = 0
    for dx in range(-int(r_outer), int(r_outer) + 1):
        for dy in range(-int(r_outer), int(r_outer) + 1):
            distance = np.sqrt(dx**2 + dy**2)
            if r_inner < distance <= r_outer:
                count += 1
    return count

def plot_GofR(lattice, x, y, r = 10, bins=20):
    distances = GofR(lattice, x, y, r)
    hist, bin_edges = np.histogram(distances, bins=bins, range=(0, r), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    
    rdf = np.zeros_like(hist, dtype=float)
    for i in range(len(hist)):
        r_inner = bin_edges[i]
        r_outer = bin_edges[i + 1]
        area = count_lattice_points_in_annulus(r_inner, r_outer)
        rdf[i] = hist[i] / area

    plt.plot(bin_centers, rdf, marker='o')
    plt.xlabel('Distance')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function for Square Lattice')
    plt.show()
    
plot_GofR(states_d, 2, 2, 15)


# In[3]:


N = 100
prob = 0.75

states_u = np.random.choice([1,-1], (N,N), p = [prob_up, 1 - prob_up])
states_d = np.random.choice([-1,1], (N,N), p = [prob_down, 1 - prob_down])

def energy(lattice):
    rows, cols = lattice.shape
    E = 0
    neighbors = [(1,0), (0,1), (-1,0), (0,-1)]
    for x in range(rows):
        for y in range(cols):
            if lattice[x,y] == 0:
                continue
                for a, b in neighbors:
                    x_neighbors = (x + a) % rows
                    y_neighbors = (y + b) % cols
                    E += lattice[x, y] * lattice[x_neighbors, y_neighbors]
    E = -E
    return E

def energy_2 (lattice):
    kern = generate_binary_structure(2,1).astype(int)
    kern[1][1] = False
    arr = -lattice * convolve(lattice, kern,mode = "wrap", cval = 0)
    E = arr.sum()
    return E
    
Energy_u = energy_2(states_u)
Energy_d = energy_2(states_d)
print(f"energy up spin: {energy_2(states_u)}")
print(f"energy down spin: {energy_2(states_d)}")
plt.imshow(states_u)
plt.show()


@jit(nopython = True)
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

spins, energies = metropolis(states_d, 1000000, 0.2, Energy_d)


fig, axes = plt.subplots(1, 2, figsize = (12, 4))
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
fig.suptitle("evelolution of spin and energy", y = 1.07, size = 18)
plt.show()


# In[4]:


def get_spin_energy(lattice, B):
    ms = np.zeros(len(B))
    E_mean = np.zeros(len(B))
    E_std = np.zeros(len(B))
    E2_mean = np.zeros(len(B))
    for i, b in enumerate(B):
        spins, energies = metropolis(lattice, 1000000, b, energy_2(lattice))
        ms[i] = spins[-100000].mean() / N**2
        E_mean[i] = energies[-100000].mean()
        E_std[i] = energies[-100000].std()
        E2_mean[i] = (energies[-100000]**2).mean()
    return ms, E_mean, E_std, E2_mean
Bs = np.arange(0.1, 2, 0.05)
ms_u, E_mean_u, E_std_u, E2_mean_u = get_spin_energy(states_u, Bs)
ms_d, E_mean_d, E_std_d, E2_mean_d = get_spin_energy(states_d, Bs)

plt.figure(figsize = (8,4))
plt.plot(1/Bs, ms_u, 'o--')
plt.plot(1/Bs, ms_d, 'o--')
plt.show()
        


# In[ ]:


C_V_u = (E2_mean_u - E_mean_u**2) #* Bs**2
C_V_d = (E2_mean_d - E_mean_d**2) #* Bs**2


plt.figure(figsize = (8,4))
plt.plot(1/Bs,E_std_u, label = 'specific heat capacity spin up')
plt.plot(1/Bs,C_V_d, label = 'specific heat capacity spin down')
plt.legend()
plt.show()


# In[10]:


plt.figure(figsize = (8, 5))
plt.plot(1/Bs, E_mean_u, label = 'average energy for spin state up')
plt.plot(1/Bs, E_mean_d, label = 'average energy for spin state down')
plt.legend()
plt.show()


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit



N = 10
@njit
def energy1(lattice, parms = 1):
    rows, cols = lattice.shape
    E = 0
    neighbors = [(1,0), (0,1), (-1,0), (0,-1)]
    for x in range(rows): 
        for y in range(cols):
            if lattice[x,y] == 0:
                continue
            for a, b in neighbors:
                x_neighbors = (x + a) % rows
                y_neighbors = (y + b) % cols
                E += lattice[x, y] * lattice[x_neighbors, y_neighbors]
    E = parms* E
    return E
@njit
def energy2(lattice, parms = 1):
    rows, cols = lattice.shape
    E = 0
    neighbors = [(1,0),(0,1),(-1,0),(0,-1),(2,0),(0,2),(0,2),(0,-2)]
    for x in range(rows):
        for y in range(cols):
            if lattice[x,y] == 0:
                continue
        for a, b in neighbors:
            x_neighbors = (x + a) % rows
            y_neighbors = (y + b) % cols
            E += lattice[x,y] * lattice[x_neighbors, y_neighbors]
    E = parms* E
    return E
@njit
def energy3(lattice, parms = 1):
    rows, cols = lattice.shape
    E = 0
    neighbors = [(1,0),(1,1),(-1,1),(-1,-1),(1,-1),(0,1),(-1,0),(0,-1)]
    for x in range(rows):
        for y in range(cols):
            if lattice[x,y] == 0:
                continue
        for a, b in neighbors:
            x_neighbors = (x + a) % rows
            y_neighbors = (y + b) % cols
            E += lattice[x,y] * lattice[x_neighbors, y_neighbors]
    E = parms* E
    return E

mat = np.zeros((N,N))
weights_d = [0.1, 0.8,0.1]
weights_u = [0.8, 0.1, 0.1]
def board(im, weights):
    im = im.copy()
    x,y = im.shape
    new_weight = np.cumsum(weights)
    for a in range(x):
        for b in range(y):
            if np.random.random() < new_weight[0]:
                im[a, b] = 1
            elif np.random.random() < new_weight[1]:
                im[a, b] = -1 
            else:
                im[a, b] = 0
    return im

rep_u = board(mat, weights_u)
rep_d = board(mat, weights_d)
energy_b = [(energy1(rep_u), energy2(rep_u),  energy3(rep_u),'spin up energy'),(energy1(rep_d), energy2(rep_d), energy3(rep_d),'spin down energy')]

plt.imshow(rep_u)
plt.show()
print(energy_b)


def metropolis_new(lattice, times, B):
    lattice = lattice.copy()
    net_spin = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    lowest_energy = energy1(lattice)
    lowest_energy_lattice = lattice.copy()
    
    
    
    for t in range(0, times - 1):    
        lattice_f = board(lattice, weights_d)
        E_f = energy1(lattice_f)
        
        dE = E_f - lowest_energy
        if (dE > 0) * (np.random.random() < np.exp(-B * dE)):
            lattice = lattice_f.copy()
           
        elif dE <= 0:
            lattice = lattice_f.copy()
            
        net_spin[t] = lattice.sum()
        current_energy = energy1(lattice)
        net_energy[t] = current_energy
        if current_energy < lowest_energy:
            lowest_energy = current_energy
            lowest_energy_lattice = lattice.copy()   
        
    return net_spin, net_energy,lowest_energy_lattice

spins_b, energies_b,lattice_opt = metropolis_new(rep_u, 100, 2)

plt.imshow(lattice_opt)

fig, axes = plt.subplots(1, 2, figsize = (12, 4))
ax = axes[0]
ax.plot(spins_b / N**2)
ax.set_xlabel("average time step")
ax.set_ylabel("net spin")
ax.grid()
ax = axes[1]
ax.plot(energies_b)
ax.set_xlabel("average time step")
ax.set_ylabel("net energy")
ax.grid()
fig.tight_layout()
fig.suptitle("evelolution of spin and energy", y = 1.07, size = 18)
plt.show()

# In[ ]:
import numpy as np
import scipy
from scipy.constants import pi, h, electron_mass
import matplotlib.pyplot as plt
import numba
from numba import njit




@njit  
def ins (lattice):
    x, y = lattice.shape 
    for a in range(x):
        for b in range(y):
            if lattice[a, b] == 0:
                if np.random.random() < 0.5:
                    lattice[a, b] = 1 
                else:
                    lattice[a, b] = -1 
                break
            else:
                continue
            
    return lattice
@njit                  
def rem (lattice):
    x,y = lattice.shape
    for a in range(x):
        for b in range(y):
            if lattice[a, b] != 0 :
                lattice[a, b] = 0
                break
            else:
                continue
    return lattice
def Lambda (B):    
    return np.sqrt(h**2 * B/(2*pi * electron_mass))
@njit  
def number (lattice):
    x, y = lattice.shape
    N = 0
    for a in range(x):
        for b in range(y):
            if lattice[a, b] != 0:
                N += 1
    return N

@jit  
def GCMC (lattice, times, B, mu):
    x, y = lattice.shape
    lattice = lattice.copy()
    net_spin = np.zeros(times - 1)
    net_energy = np.zeros(times - 1)
    lowest_energy = energy3(lattice)
    lowest_energy_lattice = lattice.copy()
    Vol = lattice.shape[0] * lattice.shape[1]
    N = number(lattice)

    
    for t in range(0, times -1):
        if np.random.random() < 0.5 :
            lattice_f = ins(lattice)
        else:
            lattice_f = rem(lattice)
        E_f = energy3(lattice_f)
        N = number(lattice_f)
        dE = E_f - lowest_energy
        #print(f"Step {t}: dE = {dE},Number = {N}, Current Lattice Sum = {lattice.sum()}, Proposed Lattice Sum = {lattice_f.sum()}")
        if N == 0:
            continue
        elif (dE > 0) * (np.random.random() < (np.exp(B * mu) *Vol *np.exp(-B * dE)/ N)):
            lattice = lattice_f.copy()
        elif dE <= 0:
            lattice = lattice_f.copy()
            
        net_spin[t] = lattice.sum()
        current_energy = energy3(lattice)
        net_energy[t] = current_energy
        if current_energy < lowest_energy:
            lowest_energy = current_energy
            lowest_energy_lattice = lattice.copy()   
        
    return net_spin, net_energy,lowest_energy_lattice, lattice

spins_b, energies_b,lattice_opt, lattice_inp = GCMC(rep_u, 100000, 0.7, 0.5)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(lattice_inp, cmap = 'viridis')
axs[0].set_title('input')
axs[1].imshow(lattice_opt, cmap = 'viridis')
axs[1].set_title('output')
plt.show()

fig, axes = plt.subplots(1, 2, figsize = (12, 4))
ax = axes[0]
ax.plot(spins_b / number(lattice_opt)**2)
ax.set_xlabel("average time step")
ax.set_ylabel("net spin")
ax.grid()
ax = axes[1]
ax.plot(energies_b)
ax.set_xlabel("average time step")
ax.set_ylabel("net energy")
ax.grid()
fig.tight_layout()
fig.suptitle("evelolution of spin and energy", y = 1.07, size = 18)
plt.show()


            
def get_spin_energy(lattice, B, mu):
    a = 1000
    ms = np.zeros(len(B))
    E_mean = np.zeros(len(B))
    E_std = np.zeros(len(B))
    E2_mean = np.zeros(len(B))
    for i, (b, m) in enumerate(zip(B,mu)):
        spins, energies,lowest_energy_lattice, lattice = GCMC(lattice, a, b, m)
        ms[i] = spins[-(a - 1)].mean() / number(lattice)**2
        E_mean[i] = energies[-(a - 1)].mean()
        E_std[i] = energies[-(a - 1)].std()
        E2_mean[i] = (energies[-(a - 1)]**2).mean()
    return ms, E_mean, E_std, E2_mean
Bs = np.arange(0.1, 10, 0.05)
mus = np.arange(0.1, 10, 0.05)
ms_u, E_mean_u, E_std_u, E2_mean_u = get_spin_energy(rep_u, Bs, mus)
ms_d, E_mean_d, E_std_d, E2_mean_d = get_spin_energy(rep_d, Bs, mus)



fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(mus, ms_u, 'o--')
axs[0].plot(mus, ms_d, 'o--')
axs[0].set_title('mean spin chem potential')
axs[1].plot(1/Bs, ms_u, 'o--')
axs[1].plot(1/Bs, ms_d, 'o--')
axs[1].set_title('mean spin temperature')
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
axs[0].plot(Bs, E_mean_u, 'o--')
axs[0].set_title('energy spin up chemical potential')
axs[1].plot(Bs, E_mean_d, 'o--')
axs[1].set_title('energy spin down chemical potential')
plt.show()

# In[ ]:
@njit
def Grand_partition(lattice, times, b, mu, energies):
    z_sum = 0
    E_set = set()
    
    for t in range(times):
        if np.random.random() < 0.5:
            lattice_f = ins(lattice)
        else:
            lattice_f = rem(lattice)

        total_E = energies(lattice_f)
        N_i = number(lattice_f)
        
        z = np.exp((N_i * mu - total_E) * 1/b)
        z_sum += z
        E_set.add(total_E)
        
    print(f"timestep = {t}, Partition Function Sum for b = {b}: {z_sum}")
    return z_sum, E_set


B = np.linspace(0.01,50 , 100) 
mu = 5e-21
times = 1000


Z_vals = []
E_vals = []

for b in B:
    z_sum, E_set = Grand_partition(states_u, times, b, mu, energy1)
    Z_vals.append(z_sum)  


Z_vals = np.array(Z_vals) 


plt.plot(B, Z_vals)
plt.title('Grand Partition Function vs Temperature')
plt.xlabel('Beta $\\mathcal{1/kbT}$')
plt.ylabel('Partition Function $\\mathcal{Z}$')
plt.grid(True)
plt.show()

Z_vals = Z_vals[np.isfinite(Z_vals)] 
plt.hist(E_set, bins=30, edgecolor='black')  
plt.title('Histogram of Partition Function Values')
plt.xlabel('Partition Function $\\mathcal{Z}$')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

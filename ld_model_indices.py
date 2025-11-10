###########################################################################
#
# Lorentz-Drude model of the gold metal's dielectric function
# source file: ld_model_indices.py
# author: Roman Malyshev
#
###########################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sqrt, cos, sin, exp, round


def lambdaToEv(lda):
    ''' Converts a wavelength value in nm to a value in electron volts '''
    #lda = lda*1e-9 # in m
    freq = c/lda # in s^-1
    eV = h*freq/e0 # in eV
    return np.round(eV,2)

def eps_r_f(w):
    ''' Calculating the intraband contribution
    to the dielectric function. '''
    return 1 - Omega_p**2/(w*(w-1j*G0))

def eps_r_b(w):
    ''' Calculating the interband contribution.
        Set - 1j*w*G_j[i] for positive cplx part.'''
    val = 0
    for i in range(len(f_j)):
        val += f_j[i]*w_p**2/(w_j[i]**2 - w**2 + 1j*w*G_j[i])
    return val

def eps_r(w):
    ''' Summing to get the dielectric (Lorentz-Drude) function. '''
    return np.array(eps_r_f(w) + eps_r_b(w))


def eps_r_lda(lda):
    ev = lambdaToEv(lda)
    return eps_r(ev)


# constants
c = 3e8        # m/s
e0 = 1.6e-19   # C
h = 6.63e-34   # J*s
hbar = h/(2*pi)

# Plasma frequency -- Rakic (1998) Gold
w_p = 9.03 # eV

# Rakic (1998) parameters (LD) -- all in eV
f0 = 0.760
G0 = 0.053
Omega_p = sqrt(f0)*w_p
f_j = [0.024, 0.010, 0.071, 0.601, 4.384]
G_j = [0.241, 0.345, 0.870, 2.494, 2.214]
w_j = [0.415, 0.830, 2.969, 4.304, 13.32]

# wavelengths in m
lda = np.arange(400,801,1)
# in eV
eV = lambdaToEv(lda*1e-9)
eps = eps_r(eV)
n = sqrt(eps).real
k = sqrt(eps).imag

# Write to file
pd.DataFrame({'n': n, 'k': k}).to_csv('Au_ld_400_800.csv')

# Read structure of Au_evap included with DDSCAT
lda = []
df = pd.read_csv('Au_evap', sep='\t', skiprows=2, header=None)[0]
keys = df[0].split()

for i in range(1, len(df)):
    lda.append(float(df[i].split()[0])*1000)

lda = np.array(lda)
eV = lambdaToEv(lda*1e-9)
eps = eps_r(eV)
n = sqrt(eps).real
k = sqrt(eps).imag

# Prepare file for DDSCAT
df = pd.DataFrame({'wave (um)': lda/1000,
                    'Re(n)': round(n, 2),
                    'Im(n)': round(abs(k), 2),
                    'eps1': round(eps.real, 2),
                    'eps2': round(eps.imag, 2)})
df.to_csv('Au_ld_2k', sep='\t', index=False)

f = open('Au_ld_2k')
lines = f.readlines()
f.close()

lines.insert(0, '1 2 3 0 0 = columns for wave, Re(n), Im(n), eps1, eps2\n')
lines.insert(0, 'Gold, evaporated (LD model) 200 - 2000 nm\n')
f = open('Au_ld_2k', 'w')
f.writelines(lines)
f.close()

# Plot
plt.plot(lda, n, label="n")
plt.plot(lda, abs(k), label="abs(k)")
plt.xlabel('Wavelength (nm)')
plt.legend()

###########################################################
# Generate a DDSCAT compatible input file for polystyrene #
###########################################################
df = pd.read_csv('polystyrene_n_real_sultanova2009.txt',
    sep=' ',
    skiprows=0,
    names=['wave (um)', 'Re(n)'])
lda = df['wave (um)'].values[::-1]*1e6
n = df['Re(n)'].values[::-1]
eps = n**2
psdf = pd.DataFrame({'wave (um)': round(lda, 4),
        'Re(n)': round(n, 2),
        'Im(n)': np.zeros(len(n)),
        'eps1': round(eps, 2),
        'eps2': np.zeros(len(n))})

# Plot
plt.plot(lda[::-1]*1000, n[::-1], label="n")
plt.xlabel('Wavelength (nm)')
plt.legend()
plt.title('Polystyrene, Sultanova (2009), $n$')

# Plot dielectric function
plt.plot(lda[::-1]*1000, eps[::-1], label="$\epsilon$")
plt.xlabel('Wavelength (nm)')
plt.legend()
plt.title('Polystyrene, Sultanova (2009), $\epsilon_1$')

# Write dataframe to CSV file
psdf.to_csv('PS_sultanova', sep='\t', index=False)

f = open('PS_sultanova')
lines = f.readlines()
f.close()

lines.insert(0, '1 2 3 0 0 = columns for wave, Re(n), Im(n), eps1, eps2\n')
lines.insert(0, 'Polystyrene, Sultanova (2009) 436 - 1052 nm\n')
f = open('PS_sultanova', 'w')
f.writelines(lines)
f.close()

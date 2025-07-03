'''FITTING SUR DES DATA BRUITEES SIMULEES'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Lorentzian function
def lorentzian(x, amp, center, width):
    return amp * (0.5 * width)**2 / ((x - center)**2 + (0.5 * width)**2)

# Total simulated spectrum: Rayleigh + Stokes + Anti-Stokes + background
def brillouin_spectrum(x, amp_r, width_r, amp_b, center_b, width_b, offset):
    rayleigh = lorentzian(x, amp_r, 0, width_r)
    stokes = lorentzian(x, amp_b, center_b, width_b)
    antistokes = lorentzian(x, amp_b, -center_b, width_b)
    return rayleigh + stokes + antistokes + offset

# Frequency axis
x = np.linspace(-10, 10, 500)

# Ground truth parameters
params_true = {
    'amp_r': 5.0,      # Rayleigh amplitude (strong)
    'width_r': 0.5,    # Rayleigh width (narrow)
    'amp_b': 1.0,      # Brillouin amplitude (weaker)
    'center_b': 5.0,   # Brillouin shift (GHz)
    'width_b': 1.2,    # Brillouin width
    'offset': 0.05     # Background offset
}

# Simulate spectrum
y = brillouin_spectrum(x, **params_true)
y_noisy = y + np.random.normal(0, 0.05, size=x.shape)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, y_noisy, label="Noisy simulated data")
plt.plot(x, y, '--', label="True model", alpha=0.7)
plt.title("Simulated Brillouin spectrum with Rayleigh")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Intensity (a.u.)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Initial guess
p0 = [4.0, 0.8, 0.8, 4.8, 1.0, 0.0]

# Fit
popt, _ = curve_fit(brillouin_spectrum, x, y_noisy, p0=p0)

# Plot fitted result
fitted = brillouin_spectrum(x, *popt)
plt.plot(x, y_noisy, label='Noisy data')
plt.plot(x, fitted, label='Fitted spectrum')
plt.plot(x, y, '--', label='True model')
plt.legend()
plt.title("Fit of Brillouin spectrum with Rayleigh")
plt.grid(True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Intensity")
plt.tight_layout()
plt.show()

# Print fitted params
param_names = ['amp_r', 'width_r', 'amp_b', 'center_b', 'width_b', 'offset']
for name, val in zip(param_names, popt):
    print(f"{name}: {val:.3f}")


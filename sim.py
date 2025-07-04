import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def lorentzian(frq, amplitude, center, gamma):
    return amplitude * gamma**2 / (gamma**2 + (frq - center)**2)

def fit_lorentzian(x, y):
    try:
        amplitude_guess = np.max(y)
        center_guess = x[np.argmax(y)]
        gamma_guess = (x[-1] - x[0]) / 10
        popt, _ = curve_fit(lorentzian, x, y, p0=[amplitude_guess, center_guess, gamma_guess])
        return popt  # amplitude, center, gamma
    except (RuntimeError, ValueError):
        return None
def fwhm_lorentzian(gamma):
    return 2 * gamma

def root(x, a, b, c):
    delta = b * b - 4 * a * (c - x)
    return (-b + np.sqrt(delta)) / (2 * a)

def synthetic_signal(width, height, order, rayleigh, stokes, antistokes):
    border = 8
    x_border = width * 0.12
    starting_frq = -7.5
    ending_frq = 7.5

    y_c = 0.5
    A = 500
    b = (height / 2) ** 2 / (A - order**2) + 1

    A = 240000
    a = 1
    b = 0.4
    x_c = -300

    min_x = np.sqrt(A - (order + 1)**2 - 0.5**2 / (b**2))
    max_x = np.sqrt(A - 4)
    x_c = x_border / 2
    a = (width - x_border - x_c) / (max_x - min_x)
    y_scale = 1 / height

    frq_lut = np.zeros((height, width), dtype=np.float32)
    signal = np.zeros((height, width), dtype=np.float32)

    rayleigh_pos = np.zeros((height - 2 * border, order), dtype=np.float32)
    rayleigh_pos_remapped = np.zeros_like(rayleigh_pos)
    rayleigh_pos_remapped_postfit = np.zeros_like(rayleigh_pos)

    def fit_poly2(x, y):
        return np.polyfit(x, y, 2)

    first_line = border

    for y in range(border, height - border):
        y_idx = y - first_line
        y_scaled = y * y_scale - y_c

        # Rayleigh peaks
        for n in range(1, order + 1):
            x = abs(a) * (np.sqrt(A - (n + 1)**2 - y_scaled**2 / (b * b)) - min_x) + x_c
            rayleigh_pos[y_idx, order - n] = x

        step = (rayleigh_pos[y_idx, -1] - rayleigh_pos[y_idx, 0]) / (order - 1)
        for n in range(order):
            rayleigh_pos_remapped[y_idx, n] = rayleigh_pos[y_idx, 0] + n * step

        # Polynomial fit
        p = fit_poly2(rayleigh_pos[y_idx, :], rayleigh_pos_remapped[y_idx, :])
        a_p, b_p, c_p = p
        remapped_fit = root(rayleigh_pos_remapped[y_idx, :], a_p, b_p, c_p)
        rayleigh_pos_remapped_postfit[y_idx, :] = remapped_fit

        # frq LUT
        start_column = int(np.floor(root(rayleigh_pos_remapped[y_idx, 1] - step / 2, a_p, b_p, c_p)))
        end_column = int(np.ceil(root(rayleigh_pos_remapped[y_idx, -1] + step / 2, a_p, b_p, c_p)))

        for x in range(start_column, end_column):
            x_remapped = np.polyval(p, x)
            dist = x_remapped - np.polyval(p, rayleigh_pos_remapped_postfit[y_idx, 0])
            pos = np.mod(dist, step)

            frq = pos / step
            if frq > 0.5:
                frq -= 1
            if frq < -0.5:
                frq += 1

            frq += 0.5
            frq = frq * (ending_frq - starting_frq) + starting_frq
            if 0 <= x < width:
                frq_lut[y, x] = frq

        # Add peaks to signal
        for x in range(start_column, end_column):
            if 0 <= x < width:
                frq = frq_lut[y, x]
                signal[y, x] += lorentzian(frq, *rayleigh)
                signal[y, x] += lorentzian(frq, *stokes)
                signal[y, x] += lorentzian(frq, *antistokes)

    # Normalize and convert to 16-bit image
    signal_u16 = np.uint16((signal / np.max(signal)) * 65535)

    y_pos = np.arange(border, height - border)
    rayleigh_pos_fit = rayleigh_pos_remapped_postfit[:, 1:]  # Skip the first column
    # Décalage moyen en pixels entre les ordres
    step = (rayleigh_pos[0, -1] - rayleigh_pos[0, 0]) / (order - 1)

    # Décalage en pixels basé sur la différence de fréquence
    stokes_shift = (stokes[1] - rayleigh[1]) / (ending_frq - starting_frq) * step
    antistokes_shift = (antistokes[1] - rayleigh[1]) / (ending_frq - starting_frq) * step

    # Positions Stokes et Anti-Stokes décalées
    stokes_pos_fit = rayleigh_pos_fit + stokes_shift
    antistokes_pos_fit = rayleigh_pos_fit + antistokes_shift

    # Concatène Rayleigh, Stokes, Anti-Stokes dans cet ordre
    multiorder_fit = np.concatenate([rayleigh_pos_fit, stokes_pos_fit, antistokes_pos_fit], axis=1)

    # Retourne tout
    #print(signal_u16.shape,y_pos,multiorder_fit,frq_lut)
    return signal_u16, y_pos, multiorder_fit, frq_lut


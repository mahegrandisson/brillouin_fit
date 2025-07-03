import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ----------------- LORENTZIAN FIT FUNCTIONS -----------------
def lorentzian(frq, amplitude, center, gamma):
    return amplitude * gamma**2 / (gamma**2 + (frq - center)**2)
def fit_lorentzian(x, y, kind="stokes"):
    try:
        amplitude_guess = np.max(y)
        center_guess = x[np.argmax(y)]
        gamma_guess = (x[-1] - x[0]) / 5  # Bonne largeur moyenne

        # Ajustement en fonction du type de pic
        if kind == "rayleigh":
            amplitude_guess *= 1.2   # Plus intense
            gamma_guess *= 0.8      # Un peu plus étroit
        elif kind in ("stokes", "antistokes"):
            amplitude_guess *= 0.8   # Moins intense
            gamma_guess *= 1.2       # Un peu plus large

        popt, _ = curve_fit(lorentzian, x, y, p0=[amplitude_guess, center_guess, gamma_guess])
        return popt
    except (RuntimeError, ValueError):
        return None


def fwhm_lorentzian(gamma):
    return 2 * gamma

def root(x, a, b, c):
    delta = b * b - 4 * a * (c - x)
    return (-b + np.sqrt(delta)) / (2 * a)

# ----------------- SYNTHETIC SIGNAL -----------------
def synthetic_signal(width, height, order, rayleigh, stokes, antistokes):
    border = 8
    x_border = width * 0.1
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

        for n in range(1, order + 1):
            x = abs(a) * (np.sqrt(A - (n + 1)**2 - y_scaled**2 / (b * b)) - min_x) + x_c
            rayleigh_pos[y_idx, order - n] = x

        step = (rayleigh_pos[y_idx, -1] - rayleigh_pos[y_idx, 0]) / (order - 1)
        for n in range(order):
            rayleigh_pos_remapped[y_idx, n] = rayleigh_pos[y_idx, 0] + n * step

        p = fit_poly2(rayleigh_pos[y_idx, :], rayleigh_pos_remapped[y_idx, :])
        a_p, b_p, c_p = p
        remapped_fit = root(rayleigh_pos_remapped[y_idx, :], a_p, b_p, c_p)
        rayleigh_pos_remapped_postfit[y_idx, :] = remapped_fit

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

        for x in range(start_column, end_column):
            if 0 <= x < width:
                frq = frq_lut[y, x]
                signal[y, x] += lorentzian(frq, *rayleigh)
                signal[y, x] += lorentzian(frq, *stokes)
                signal[y, x] += lorentzian(frq, *antistokes)

    signal_u16 = np.uint16((signal / np.max(signal)) * 65535)
    y_pos = np.arange(border, height - border)
    rayleigh_pos_fit = rayleigh_pos_remapped_postfit[:, 1:]

    return signal_u16, y_pos, rayleigh_pos_fit, frq_lut, starting_frq, ending_frq

# ----------------- ANALYZE -----------------
def analyze_signal_multiorders(signal, frq_lut, y_pos, rayleigh_fit, show_every=50):
    n_orders = rayleigh_fit.shape[1] // 3
    results = []
    stokes_fit_positions = np.full((len(y_pos), n_orders), np.nan)
    
    for i, y in enumerate(y_pos):
        # print(i,y)
        row = signal[y, :]
        frq_row = frq_lut[y, :]

        for order in range(n_orders):
            # --- Fit du pic Stokes ---
            stokes_x = int(rayleigh_fit[i, order * 3 + 1])
            window = 20
            x_start = max(0, stokes_x - window)
            x_end = min(len(row), stokes_x + window)
            intensity = row[x_start:x_end].astype(np.float32)
            frq_window = frq_row[x_start:x_end]

            if intensity.max() == 0:
                continue

            intensity /= intensity.max()
            fit_params = fit_lorentzian(frq_window, intensity, kind="stokes")

            if fit_params is None:
                continue

            amp, center, gamma = fit_params
            fwhm = fwhm_lorentzian(gamma)
            x_interp = np.interp(center, frq_window, np.arange(x_start, x_end))
            stokes_fit_positions[i, order] = x_interp

            rayleigh_x = int(rayleigh_fit[i, order * 3 + 0])
            antistokes_x = int(rayleigh_fit[i, order * 3 + 2])
            rayleigh_frq = frq_lut[y, rayleigh_x]
            antistokes_frq = frq_lut[y, antistokes_x]

            delta_stokes = center - rayleigh_frq
            delta_antistokes = rayleigh_frq - antistokes_frq

            results.append({
                'y': y,
                'order': order + 1,
                'stokes_center': center,
                'fwhm': fwhm,
                'delta_stokes': delta_stokes,
                'delta_antistokes': delta_antistokes
            })

            # if i % show_every == 0 and order == 0:
            #     frq_dense = np.linspace(frq_window.min(), frq_window.max(), 500)
            #     fit_curve = lorentzian(frq_dense, *fit_params)

            #     plt.figure(figsize=(6, 4))
            #     plt.plot(frq_window, intensity, 'bo', label='Données')
            #     plt.plot(frq_dense, fit_curve, 'r-', label='Fit Lorentzien')
            #     plt.axvline(center, color='r', linestyle='--', label='Stokes')
            #     plt.axvline(rayleigh_frq, color='g', linestyle='--', label='Rayleigh')
            #     plt.axvline(antistokes_frq, color='m', linestyle='--', label='Anti-Stokes')
            #     plt.title(f"Fit (ordre {order+1}) à y={y} | FWHM={fwhm:.3f}")
            #     plt.xlabel("Fréquence")
            #     plt.ylabel("Intensité")
            #     plt.legend()
            #     plt.grid(True)
            #     plt.tight_layout()
            #     plt.show()

    plot_signal_with_fit_multiorders(signal, rayleigh_fit, y_pos, stokes_fit_positions)
    return results

def plot_signal_with_fit_multiorders(signal_2d, rayleigh_fit, y_pos, stokes_fit=None, title="Signal avec fit des pics multiples"):
    plt.figure(figsize=(12, 6))
    plt.imshow(signal_2d, cmap='gray', origin='lower', aspect='auto')

    n_orders = rayleigh_fit.shape[1] // 3
    colors = ['g', 'r', 'm']
    labels = ['Rayleigh', 'Stokes', 'Anti-Stokes']

    for order in range(n_orders):
        for j in range(3):
            x_curve = rayleigh_fit[:, order * 3 + j]
            plt.plot(x_curve, y_pos, color=colors[j], linewidth=1.5, alpha=0.7)

        #if stokes_fit is not None:
        #    plt.plot(stokes_fit[:, order], y_pos, 'c--', linewidth=1.5, label=f"Stokes fit {order+1}" if order == 0 else "")

    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.legend()
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# ----------------- MAIN -----------------
if __name__ == "__main__":
    width = 512
    height = 1024
    order = 4
    rayleigh    = (1.0,  0.0, 0.1)   # Le pic central
    stokes      = (0.6,  5.0, 0.3)   # Pic Stokes (à droite du Rayleigh)
    antistokes  = (0.5, -5.0, 0.3)   # Pic Anti-Stokes (à gauche du Rayleigh)

    signal_img, y_pos, rayleigh_fit, frq_lut, start_frq, end_frq = synthetic_signal(width, height, order, rayleigh, stokes, antistokes)

    step = np.mean(rayleigh_fit[:, 1:] - rayleigh_fit[:, :-1])
    stokes_shift = (stokes[1] - rayleigh[1]) / (end_frq - start_frq) * step
    antistokes_shift = (antistokes[1] - rayleigh[1]) / (end_frq - start_frq) * step
    stokes_pos_fit = rayleigh_fit + stokes_shift
    antistokes_pos_fit = rayleigh_fit + antistokes_shift

    multiorder_fit = np.zeros((rayleigh_fit.shape[0], rayleigh_fit.shape[1] * 3))
    for i in range(rayleigh_fit.shape[1]):
        multiorder_fit[:, i * 3 + 0] = rayleigh_fit[:, i]
        multiorder_fit[:, i * 3 + 1] = stokes_pos_fit[:, i]
        multiorder_fit[:, i * 3 + 2] = antistokes_pos_fit[:, i]

    results = analyze_signal_multiorders(signal_img, frq_lut, y_pos, multiorder_fit, show_every=50)

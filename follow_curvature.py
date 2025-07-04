'''BALAYAGE PAR FENETRE (EN X ET Y), MOYENNAGE SUR CES FENTRES'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Lorentzienne
def lorentzian(x, A, x0, gamma, y0):
    return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2) + y0

def detect_triplet(profile, prominence=0.05, distance=5):
    peaks, props = find_peaks(profile, prominence=prominence, distance=distance)
    if len(peaks) < 3:
        return None  # Pas de triplet
    center = len(profile) // 2
    triplet = sorted(peaks, key=lambda p: abs(p - center))[:3]
    return sorted(triplet)  # (anti, rayleigh, stokes)

def fit_brillouin_peak(x, y, p0):
    popt, pcov = curve_fit(lorentzian, x, y, p0=p0)
    A, x0, gamma, y0 = popt
    perr = np.sqrt(np.diag(pcov))
    return {
        "shift": x0,
        "fwhm": gamma,
        "A": A,
        "offset": y0,
        "shift_err": perr[1],
        "fwhm_err": perr[2]
    }

def extract_triplet_positions(signal, prominence=0.05):
    H, W = signal.shape
    triplets = []

    for y in range(H):
        profile = signal[y, :]
        norm_profile = profile / np.max(profile) if np.max(profile) > 0 else profile
        peaks, _ = find_peaks(norm_profile, prominence=prominence)

        if len(peaks) >= 3:
            center = W // 2
            sorted_peaks = sorted(peaks, key=lambda p: abs(p - center))[:3]
            sorted_peaks = sorted(sorted_peaks)
            triplets.append(tuple(sorted_peaks))
        else:
            triplets.append((np.nan, np.nan, np.nan))

    return triplets

def compute_average_shifts(triplets):
    anti_ray_shifts = []
    stokes_ray_shifts = []

    for anti, ray, stokes in triplets:
        if not np.isnan(anti) and not np.isnan(ray) and not np.isnan(stokes):
            anti_ray_shifts.append(ray - anti)
            stokes_ray_shifts.append(stokes - ray)

    avg_anti_ray = np.nanmean(anti_ray_shifts)
    avg_stokes_ray = np.nanmean(stokes_ray_shifts)

    print(f"Décalage moyen Anti-Stokes ↔ Rayleigh : {avg_anti_ray:.2f} px")
    print(f"Décalage moyen Stokes ↔ Rayleigh     : {avg_stokes_ray:.2f} px")

    return avg_anti_ray, avg_stokes_ray

def fit_triplets_by_window(signal, triplets, window_height=10, window_width=6):
    H, W = signal.shape
    fwhms_stokes, fwhms_anti = [], []

    for y0 in range(0, H - window_height + 1, window_height):
        stokes_stack = []
        anti_stack = []

        for dy in range(window_height):
            y = y0 + dy
            if y >= H:
                continue
            anti, ray, stokes = triplets[y]
            x = np.arange(W)

            if not np.isnan(anti) and not np.isnan(stokes):
                x_anti = x[int(anti - window_width):int(anti + window_width + 1)]
                y_anti = signal[y, int(anti - window_width):int(anti + window_width + 1)]
                if len(y_anti) == 2 * window_width + 1:
                    anti_stack.append(y_anti)

                x_stokes = x[int(stokes - window_width):int(stokes + window_width + 1)]
                y_stokes = signal[y, int(stokes - window_width):int(stokes + window_width + 1)]
                if len(y_stokes) == 2 * window_width + 1:
                    stokes_stack.append(y_stokes)

        if anti_stack:
            profile_anti = np.mean(anti_stack, axis=0)
            try:
                A_guess = profile_anti.max() - profile_anti.min()
                y0_guess = profile_anti.min()
                res = fit_brillouin_peak(np.arange(len(profile_anti)), profile_anti, p0=(A_guess, window_width, 4, y0_guess))
                fwhms_anti.append(res["fwhm"])
            except:
                fwhms_anti.append(np.nan)

        if stokes_stack:
            profile_stokes = np.mean(stokes_stack, axis=0)
            try:
                A_guess = profile_stokes.max() - profile_stokes.min()
                y0_guess = profile_stokes.min()
                res = fit_brillouin_peak(np.arange(len(profile_stokes)), profile_stokes, p0=(A_guess, window_width, 4, y0_guess))
                fwhms_stokes.append(res["fwhm"])
            except:
                fwhms_stokes.append(np.nan)

    print(f"FWHM moyen Anti-Stokes (fenêtres) : {np.nanmean(fwhms_anti):.2f} px")
    print(f"FWHM moyen Stokes (fenêtres)      : {np.nanmean(fwhms_stokes):.2f} px")

    return fwhms_anti, fwhms_stokes

def simulate_brillouin_image(width=150, height=100, noise_level=0.02, curvature=10):
    image = np.zeros((height, width), dtype=np.float32)
    x = np.arange(width)
    delta = 10
    gamma = 3
    for y in range(height):
        A = 1.0
        y0 = 0.05 + 0.01 * np.random.randn()
        offset = curvature * ((y - height / 2) ** 2) / height**2  # courbure quadratique
        rayleigh_pos = width // 2 + offset
        signal = (
            lorentzian(x, A, rayleigh_pos - delta, gamma, 0) +
            lorentzian(x, A, rayleigh_pos, gamma, 0) +
            lorentzian(x, A, rayleigh_pos + delta, gamma, 0)
        )
        noise = noise_level * np.random.randn(width)
        image[y] = signal + y0 + noise
    image /= image.max()
    return image

def show_image(image):
    plt.figure(figsize=(8, 5))
    plt.imshow(image, aspect='auto', cmap='viridis')
    plt.title("Image Brillouin simulée")
    plt.xlabel("Spectre")
    plt.ylabel("Ligne")
    plt.colorbar(label="Intensité")
    plt.show()

# ----------- MAIN --------------
if __name__ == "__main__":
    image = simulate_brillouin_image(width=150, height=100, noise_level=0.03)
    show_image(image)

    triplets = extract_triplet_positions(image, prominence=0.05)
    compute_average_shifts(triplets)

    # Fit par fenêtre (méthode robuste)
    fwhm_anti, fwhm_stokes = fit_triplets_by_window(image, triplets, window_height=10, window_width=6)

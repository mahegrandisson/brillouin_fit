'''ON CONSIDERE UNE GROSSE LIGNE ET ON MOYENNE DESSUS ON A DONC PLUSIEURS MOYENNES LOCALES AU LIEU D'UNE SEULE GROSSE MOYENNE'''

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

# Fit simple d'un pic Brillouin
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

# Détection des triplets sur chaque ligne (anti-Stokes, Rayleigh, Stokes)
def extract_triplet_positions(signal, prominence=0.05):
    H, W = signal.shape
    triplets = []
 
    for y in range(H):
        profile = signal[y, :]
        norm_profile = profile / np.max(profile) if np.max(profile) > 0 else profile
        peaks, _ = find_peaks(norm_profile, prominence=prominence)

        if len(peaks) >= 3:
            # on prend les 3 pics les plus proches du centre
            center = W // 2
            sorted_peaks = sorted(peaks, key=lambda p: abs(p - center))[:3]
            sorted_peaks = sorted(sorted_peaks)
            triplets.append(tuple(sorted_peaks))
        else:
            triplets.append((np.nan, np.nan, np.nan))

    return triplets

# Moyennes des décalages anti-stokes ↔ rayleigh et stokes ↔ rayleigh
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

#obsolete
def analyze_image_shifts(signal, window=6, plot=True):
    H, W = signal.shape
    fwhms_stokes = []
    fwhms_anti = []

    for y in range(H):
        profile = signal[y, :]
        x = np.arange(W)

        triplet = detect_triplet(profile)
        if triplet is None:
            fwhms_stokes.append(np.nan)
            fwhms_anti.append(np.nan)
            continue
        anti, rayleigh, stokes = triplet

        x_stokes = x[stokes - window:stokes + window + 1]
        y_stokes = profile[stokes - window:stokes + window + 1]

        x_anti = x[anti - window:anti + window + 1]
        y_anti = profile[anti - window:anti + window + 1]

        try:
            # Stokes Fit
            A_guess = y_stokes.max() - y_stokes.min()
            y0_guess = 0
            res_s = fit_brillouin_peak(x_stokes, y_stokes, p0=(A_guess, stokes, 1, y0_guess))
            fwhms_stokes.append(res_s["fwhm"])
        except:
            fwhms_stokes.append(np.nan)

        try:
            # Anti-Stokes Fit
            A_guess = y_anti.max() - y_anti.min()
            y0_guess = 0
            res_a = fit_brillouin_peak(x_anti, y_anti, p0=(A_guess, anti, 1, y0_guess))
            fwhms_anti.append(res_a["fwhm"])
        except:
            fwhms_anti.append(np.nan)

        
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(x, profile, label='Profil complet')
            plt.plot(x_stokes, y_stokes, 'r.', label='Stokes (fit)')
            plt.plot(x_anti, y_anti, 'b.', label='Anti-Stokes (fit)')
            plt.title(f"Ligne {y} — FWHM Stokes: {fwhms_stokes[-1]:.2f}, Anti-Stokes: {fwhms_anti[-1]:.2f}")
            plt.xlabel("Pixel")
            plt.ylabel("Intensité")
            plt.legend()
            plt.tight_layout()
            plt.show()

    return np.array(fwhms_stokes), np.array(fwhms_anti)

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

# Fit par fenêtres verticales (10% hauteur)
def fit_each_brillouin_peak_by_window(signal, triplets, window=7, window_height_ratio=0.5):
    H, W = signal.shape
    window_height = int(window_height_ratio * H)
    fwhms_stokes = []
    fwhms_anti = []

    for y0 in range(0, H - window_height + 1, window_height):
        anti_stack = []
        stokes_stack = []

        for dy in range(window_height):
            y = y0 + dy
            if y >= H:
                continue
            anti, ray, stokes = triplets[y]
            profile = signal[y, :]
            x = np.arange(W)

            if not np.isnan(anti):
                x_anti = x[int(anti - window):int(anti + window + 1)]
                y_anti = profile[int(anti - window):int(anti + window + 1)]
                if len(x_anti) == 2 * window + 1:
                    anti_stack.append(y_anti)

            if not np.isnan(stokes):
                x_stokes = x[int(stokes - window):int(stokes + window + 1)]
                y_stokes = profile[int(stokes - window):int(stokes + window + 1)]
                if len(x_stokes) == 2 * window + 1:
                    stokes_stack.append(y_stokes)

        if anti_stack:
            profile_anti = np.mean(anti_stack, axis=0)
            x_fit = np.arange(len(profile_anti))
            try:
                A_guess = profile_anti.max() - profile_anti.min()
                y0_guess = profile_anti.min()
                res = fit_brillouin_peak(x_fit, profile_anti, p0=(A_guess, window, 5, y0_guess))
                fwhms_anti.append(res["fwhm"])
            except:
                fwhms_anti.append(np.nan)

        if stokes_stack:
            profile_stokes = np.mean(stokes_stack, axis=0)
            x_fit = np.arange(len(profile_stokes))
            try:
                A_guess = profile_stokes.max() - profile_stokes.min()
                y0_guess = profile_stokes.min()
                res = fit_brillouin_peak(x_fit, profile_stokes, p0=(A_guess, window, 4, y0_guess))
                fwhms_stokes.append(res["fwhm"])
            except:
                fwhms_stokes.append(np.nan)

    avg_anti_fwhm = np.nanmean(fwhms_anti)
    avg_stokes_fwhm = np.nanmean(fwhms_stokes)

    print(f"FWHM moyen Anti-Stokes (fenêtres) : {avg_anti_fwhm:.2f} px")
    print(f"FWHM moyen Stokes (fenêtres)      : {avg_stokes_fwhm:.2f} px")

    return fwhms_anti, fwhms_stokes

def show_image(image):
    plt.figure(figsize=(8, 5))
    plt.imshow(image, aspect='auto', cmap='viridis')
   

# ----------- MAIN --------------
if __name__ == "__main__":
    image = simulate_brillouin_image(width=150, height=100, noise_level=0.03)
    show_image(image)

    #---------------------------------
    # shifts, fwhms = analyze_image_shifts(image,plot=False)
    #---------------------------------

    triplets = extract_triplet_positions(image, prominence=0.05)
    compute_average_shifts(triplets)

    s,f = fit_each_brillouin_peak_by_window(image, triplets)
    

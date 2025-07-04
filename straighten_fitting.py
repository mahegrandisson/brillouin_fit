'''ON ANNLUE LA COURBURE EN REDRESSANT LES DROITES ET ON FAIT NOS MESURES DE MOYENNES DESSUS'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ------------------ Modèle de pic : Lorentzienne ------------------
def lorentzian(x, A, x0, gamma, y0):
    return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2) + y0

# ------------------ Détection simple de triplet ------------------
def detect_triplet(profile, prominence=0.05, distance=5):
    peaks, _ = find_peaks(profile, prominence=prominence, distance=distance)
    if len(peaks) < 3:
        return None
    center = len(profile) // 2
    triplet = sorted(peaks, key=lambda p: abs(p - center))[:3]
    return sorted(triplet)

# ------------------ Fit d’un seul pic ------------------
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

# ------------------ Extraire tous les triplets ------------------
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
            triplets.append(tuple(sorted(sorted_peaks)))
        else:
            triplets.append((np.nan, np.nan, np.nan))
    return triplets

# ------------------ Moyennes des décalages ------------------
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

# ------------------ Simulation d'image Brillouin ------------------
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

# ------------------ Fit sur chaque pic du triplet ------------------
def fit_each_brillouin_peak(signal, triplets, window=7):
    fwhms_stokes = []
    fwhms_anti = []
    for y, (anti, ray, stokes) in enumerate(triplets):
        profile = signal[y, :]
        x = np.arange(len(profile))
        if not np.isnan(anti) and not np.isnan(stokes):
            # Anti-Stokes
            x_anti = x[int(anti - window):int(anti + window + 1)]
            y_anti = profile[int(anti - window):int(anti + window + 1)]
            if len(x_anti) == 2 * window + 1:
                try:
                    A_guess = y_anti.max() - y_anti.min()
                    y0_guess = y_anti.min()
                    res = fit_brillouin_peak(x_anti, y_anti, p0=(A_guess, anti, 5, y0_guess))
                    fwhms_anti.append(res["fwhm"])
                except:
                    pass
            # Stokes
            x_stokes = x[int(stokes - window):int(stokes + window + 1)]
            y_stokes = profile[int(stokes - window):int(stokes + window + 1)]
            if len(x_stokes) == 2 * window + 1:
                try:
                    A_guess = y_stokes.max() - y_stokes.min()
                    y0_guess = y_stokes.min()
                    res = fit_brillouin_peak(x_stokes, y_stokes, p0=(A_guess, stokes, 4, y0_guess))
                    fwhms_stokes.append(res["fwhm"])
                except:
                    pass
    print(f"FWHM moyen Anti-Stokes : {np.nanmean(fwhms_anti):.2f} px")
    print(f"FWHM moyen Stokes      : {np.nanmean(fwhms_stokes):.2f} px")
    return fwhms_anti, fwhms_stokes

# ------------------ Correction de la courbure (quadratique) ------------------
def straighten_image_quadratic(image, triplets):
    H, W = image.shape
    rayleigh_positions = [ray for (_, ray, _) in triplets if not np.isnan(ray)]
    y_indices = [y for y, (_, ray, _) in enumerate(triplets) if not np.isnan(ray)]

    # Ajustement quadratique des positions Rayleigh
    coeffs = np.polyfit(y_indices, rayleigh_positions, deg=2)
    poly = np.poly1d(coeffs)
    center_ref = int(W // 2)

    straightened = np.zeros_like(image)
    for y in range(H):
        shift = int(round(center_ref - poly(y)))
        straightened[y] = np.roll(image[y], shift)
    return straightened

# ------------------ Affichage ------------------
def show_image(image, title="Image Brillouin simulée"):
    plt.figure(figsize=(8, 5))
    plt.imshow(image, aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Spectre (px)")
    plt.ylabel("Ligne")
    plt.colorbar(label="Intensité")
    plt.tight_layout()
    plt.show()

# ------------------ MAIN ------------------
if __name__ == "__main__":
    image = simulate_brillouin_image(width=150, height=100, noise_level=0.03)
    show_image(image, title="Image avant redressement")

    triplets = extract_triplet_positions(image, prominence=0.05)
    compute_average_shifts(triplets)

    image_flat = straighten_image_quadratic(image, triplets)
    show_image(image_flat, title="Image redressée (corrigée par polynôme degré 2)")

    triplets_flat = extract_triplet_positions(image_flat, prominence=0.05)
    fit_each_brillouin_peak(image_flat, triplets_flat)



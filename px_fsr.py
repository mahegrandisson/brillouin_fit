'''TEST MESURE LA DISTANCE ENTRE STOKES D'UN TRIPLET ET ANTI STOKES DU PROCHAIN. PERMET DE DEDUIRE le frequency shift du brillouin (vB)
CODE EN DUR POUR AVOIR SEULEMENT DEUX TRIPLETS'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Lorentzienne
def lorentzian(x, A, x0, gamma, y0):
    return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2) + y0

# Détection des pics
def detect_peaks(profile, prominence=0.05, distance=5):
    peaks, _ = find_peaks(profile, prominence=prominence, distance=distance)
    return peaks

# Fit simple d'un pic Brillouin
def fit_brillouin_peak(x, y, p0):
    popt, pcov = curve_fit(lorentzian, x, y, p0=p0)
    A, x0, gamma, y0 = popt
    return {
        "shift": x0,
        "fwhm": gamma,
        "A": A,
        "offset": y0
    }

# Simulation d'une image avec 2 triplets par ligne
def simulate_double_triplet_image_curved(width=200, height=100, noise_level=0.02):
    image = np.zeros((height, width), dtype=np.float32)
    x = np.arange(width)

    rayleigh_pos_1_base = width // 4
    rayleigh_pos_2_base = 3 * width // 4
    delta = 10
    gamma = 3

    # Paramètre de courbure (positive => arc vers le haut)
    curvature = 0.004  # ajuste cette valeur pour la force de la courbure

    for y in range(height):
        A = 1.0
        y0 = 0.05 + 0.01 * np.random.randn()

        offset = curvature * (y - height / 2)**2
        rayleigh_pos_1 = rayleigh_pos_1_base + offset
        rayleigh_pos_2 = rayleigh_pos_2_base + offset

        signal = (
            lorentzian(x, A, rayleigh_pos_1 - delta, gamma, 0) +
            lorentzian(x, A, rayleigh_pos_1, gamma, 0) +
            lorentzian(x, A, rayleigh_pos_1 + delta, gamma, 0) +

            lorentzian(x, A, rayleigh_pos_2 - delta, gamma, 0) +
            lorentzian(x, A, rayleigh_pos_2, gamma, 0) +
            lorentzian(x, A, rayleigh_pos_2 + delta, gamma, 0)
        )
        noise = noise_level * np.random.randn(width)
        image[y] = signal + y0 + noise

    image /= image.max()
    return image



# Extraire positions des pics par ligne en trouvant 6 pics les plus proches du centre des deux triplets
def extract_six_peak_positions(signal, prominence=0.05):
    H, W = signal.shape
    peak_positions = []

    for y in range(H):
        profile = signal[y, :]
        norm_profile = profile / np.max(profile) if np.max(profile) > 0 else profile
        peaks, _ = find_peaks(norm_profile, prominence=prominence, distance=5)

        if len(peaks) >= 6:
            # On veut 3 pics autour du premier triplet et 3 autour du second
            center1 = W // 4
            center2 = 3 * W // 4
            # Tri par distance à chaque centre
            peaks1 = sorted(peaks, key=lambda p: abs(p - center1))[:3]
            peaks2 = sorted(peaks, key=lambda p: abs(p - center2))[:3]
            peaks_combined = sorted(peaks1) + sorted(peaks2)
            peak_positions.append(tuple(peaks_combined))
        else:
            peak_positions.append((np.nan,) * 6)

    return peak_positions

# Fit des pics (anti1, ray1, stokes1, anti2, ray2, stokes2)
def fit_double_triplets(signal, peak_positions, window=7):
    fwhms = []
    shifts = []

    for y, peaks in enumerate(peak_positions):
        if any(np.isnan(peaks)):
            fwhms.append([np.nan]*6)
            shifts.append([np.nan]*6)
            continue

        profile = signal[y, :]
        x = np.arange(len(profile))
        line_fwhm = []
        line_shifts = []

        for peak in peaks:
            x_peak = x[int(peak - window):int(peak + window + 1)]
            y_peak = profile[int(peak - window):int(peak + window + 1)]
            if len(x_peak) != 2 * window + 1:
                line_fwhm.append(np.nan)
                line_shifts.append(np.nan)
                continue
            try:
                A_guess = y_peak.max() - y_peak.min()
                y0_guess = y_peak.min()
                res = fit_brillouin_peak(x_peak, y_peak, p0=(A_guess, peak, 4, y0_guess))
                line_fwhm.append(res["fwhm"])
                line_shifts.append(res["shift"])
            except:
                line_fwhm.append(np.nan)
                line_shifts.append(np.nan)

        fwhms.append(line_fwhm)
        shifts.append(line_shifts)

    return np.array(fwhms), np.array(shifts)

def show_image(image):
    plt.figure(figsize=(8, 5))
    plt.imshow(image, aspect='auto', cmap='viridis')
    plt.title("Image Brillouin simulée (Double triplet)")
    plt.xlabel("Pixel")
    plt.ylabel("Ligne")
    plt.colorbar(label="Intensité")
    plt.show()

# ----------- MAIN --------------
if __name__ == "__main__":
    image = simulate_double_triplet_image_curved(width=200, height=100, noise_level=0.03)
    show_image(image)

    peaks = extract_six_peak_positions(image, prominence=0.05)

    fwhms, shifts = fit_double_triplets(image, peaks)

    # Calcul de la distance entre Stokes du premier triplet et Anti-Stokes du second triplet (en pixels)
    # Indices : triplet1 = (anti1=0, ray1=1, stokes1=2), triplet2 = (anti2=3, ray2=4, stokes2=5)
    distances = []
    for line_shifts in shifts:
        stokes1 = line_shifts[2]
        anti2 = line_shifts[3]
        if not np.isnan(stokes1) and not np.isnan(anti2):
            distances.append(anti2 - stokes1)
        else:
            distances.append(np.nan)

    avg_distance = np.nanmean(distances)
    print(f"Distance moyenne entre Stokes du premier triplet et Anti-Stokes du deuxième triplet : {avg_distance:.2f} px")
    #TO DO : convertir le décalage en GHz et applique la formule vB = 1/2 * (FSR-PX) avec PX le décalage en GHz et FSR le FSR du VIPA en GHz (30 GHz pour nous)

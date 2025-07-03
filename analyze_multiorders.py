'''FICHIER POUR TESTER L'ANALYSE AVEC PLUSIEURS ORDRES --> J'ai PRIS LA MEILLEURE IMAGE CENTREE SUR UN ROI INTERESSANT POUR
LIMITER LA COURBURE ET AVOIR DE MEILLEURS RESULTATS'''

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

def synthetic_signal(width, height, n_triplets, rayleigh, stokes, antistokes, curvature_radius=80000):
    """
    Crée une image avec plusieurs triplets (Rayleigh, Stokes, Anti-Stokes),
    espacés pour éviter chevauchement.
    Chaque ligne a une légère courbure en arc de cercle.
    """

    shifts = np.array([rayleigh[1], stokes[1], antistokes[1]])
    largeur_triplet = shifts.max() - shifts.min()
    marge = 1.0
    facteur = 5
    espacement = (largeur_triplet + marge) * facteur
    centres_base = np.linspace(-espacement*(n_triplets-1)/2, espacement*(n_triplets-1)/2, n_triplets)

    x_min = centres_base.min() + shifts.min() - 5 * max(rayleigh[2], stokes[2], antistokes[2])
    x_max = centres_base.max() + shifts.max() + 5 * max(rayleigh[2], stokes[2], antistokes[2])
    x = np.linspace(x_min, x_max, width)

    signal = np.zeros((height, width), dtype=np.float32)
    frq_lut = np.zeros((height, width), dtype=np.float32)

    y_c = height / 2

    for y in range(height):
        # Calcul du décalage horizontal pour la courbure en arc
        dy = y - y_c
        if abs(dy) > curvature_radius:
            delta_x = 0  # hors du cercle, pas de décalage (ou on peut clipper)
        else:
            delta_x = curvature_radius - np.sqrt(curvature_radius**2 - dy**2)

        # Centres décalés sur cette ligne
        centres = centres_base + delta_x

        row = np.zeros_like(x)
        for center in centres:
            row += lorentzian(x, rayleigh[0], center + rayleigh[1], rayleigh[2])
            row += lorentzian(x, stokes[0], center + stokes[1], stokes[2])
            row += lorentzian(x, antistokes[0], center + antistokes[1], antistokes[2])
        signal[y, :] = row
        frq_lut[y, :] = x

    return signal, frq_lut, centres_base


def group_triplets_by_spacing(peak_frqs, stokes_shift=2, tolerance=1.5):
    """
    Regroupe les pics en triplets basés sur l’espacement attendu Rayleigh ± shift.

    Args:
        peak_frqs (list or ndarray): fréquences triées de pics détectés
        stokes_shift (float): décalage attendu entre Rayleigh et Stokes/Anti-Stokes
        tolerance (float): tolérance sur l’espacement en GHz

    Returns:
        list of tuples: chaque tuple contient (anti_stokes, rayleigh, stokes)
    """
    triplets = []

    for i in range(len(peak_frqs)):
        for j in range(len(peak_frqs)):
            for k in range(len(peak_frqs)):
                if i == j or j == k or i == k:
                    continue
                f1, f2, f3 = sorted([peak_frqs[i], peak_frqs[j], peak_frqs[k]])
                # Cherche une structure (Anti, Rayleigh, Stokes)
                d1 = abs(f2 - f1 - stokes_shift)
                d2 = abs(f3 - f2 - stokes_shift)
                if d1 < tolerance and d2 < tolerance:
                    triplets.append((f1, f2, f3))
                # print(triplets)     
                    

    # Optionnel : filtrer les doublons
    triplets_unique = []
    seen = set()
    for t in triplets:
        t_key = tuple(np.round(t, 3))  # pour éviter flottants trop proches
        if t_key not in seen:
            seen.add(t_key)
            triplets_unique.append(t)

    return triplets_unique

# analyze.py
# def analyze_triplets(signal, frq_lut, y_range, centers, stokes_shift=5.0, antistokes_shift=-5.0):
#     results = []

#     for y in y_range:
#         row = signal[y, :]
#         frq_row = frq_lut[y, :]

#         for idx, ray_center in enumerate(centers):
#             for kind, shift in zip(['Rayleigh', 'Stokes', 'Anti-Stokes'], [0.0, stokes_shift, antistokes_shift]):
#                 guess = ray_center + shift
#                 mask = (frq_row > guess - 0.5) & (frq_row < guess + 0.5)
#                 if np.count_nonzero(mask) < 5:
#                     continue

#                 x_fit = frq_row[mask]
#                 y_fit = row[mask]
#                 y_fit /= np.max(y_fit)

#                 fit = fit_lorentzian(x_fit, y_fit)
#                 if fit is not None:
#                     amp, ctr, gamma = fit
#                     results.append((y, idx, kind, amp, ctr, gamma))

#     return results

from scipy.signal import find_peaks
import numpy as np

def analyze_triplets_in_patches(signal, frq_lut, 
                                h_patch=10, w_patch=50, 
                                dy=5, dx=10,
                                min_prominence=0.065, min_height=0.8):
    """
    Analyse par patchs et trace les lignes reliant les Anti-Stokes (rouge), Rayleigh (bleu), et Stokes (vert)
    détectés dans chaque colonne de patchs.
    """
    H, W = signal.shape
    detections = []

    # Dictionnaires pour stocker les coordonnées des pics par colonne de patch
    lines_by_column = {}  # clé = x0 ; valeur = dict avec x/y pour anti, ray, stokes

    for y0 in range(0, H - h_patch + 1, dy):
        for x0 in range(0, W - w_patch + 1, dx):
            patch = signal[y0:y0+h_patch, x0:x0+w_patch]
            patch_frq = frq_lut[y0:y0+h_patch, x0:x0+w_patch]

            spectrum = np.mean(patch, axis=0)
            frqs = np.mean(patch_frq, axis=0)

            sort_idx = np.argsort(frqs)
            frqs_sorted = frqs[sort_idx]
            spec_sorted = spectrum[sort_idx]

            if np.max(spec_sorted) == 0:
                continue

            spec_sorted /= np.max(spec_sorted)

            peaks, props = find_peaks(spec_sorted,
                                       prominence=(min_prominence * np.max(spec_sorted)),
                                       height=min_height)
            peak_frqs = frqs_sorted[peaks]
            triplets = group_triplets_by_spacing(peak_frqs)
            # print("TRIPLETS",triplets)

            if len(triplets) > 0:
                frq_row = np.mean(patch_frq, axis=0)
                try:
                    anti, ray, st = triplets[0]  # ordre supposé
                    anti_idx = np.argmin(np.abs(frq_row - anti))
                    ray_idx  = np.argmin(np.abs(frq_row - ray))
                    st_idx   = np.argmin(np.abs(frq_row - st))

                    x_anti, y_anti = x0 + anti_idx, y0 + h_patch // 2
                    x_ray,  y_ray  = x0 + ray_idx,  y0 + h_patch // 2
                    x_st,   y_st   = x0 + st_idx,   y0 + h_patch // 2

                    if x0 not in lines_by_column:
                        lines_by_column[x0] = {
                            "anti_x": [], "anti_y": [],
                            "ray_x": [],  "ray_y": [],
                            "st_x": [],   "st_y": []
                        }

                    lines_by_column[x0]["anti_x"].append(x_anti)
                    lines_by_column[x0]["anti_y"].append(y_anti)
                    lines_by_column[x0]["ray_x"].append(x_ray)
                    lines_by_column[x0]["ray_y"].append(y_ray)
                    lines_by_column[x0]["st_x"].append(x_st)
                    lines_by_column[x0]["st_y"].append(y_st)

                except Exception as e:
                    print(f"Erreur conversion fréquence → pixel : {e}")

            detections.append({
                "y": y0,
                "x": x0,
                "triplets": triplets,
                "num_peaks": len(peaks),
                "raw_peaks": peak_frqs
            })
    
    # print(lines_by_column.items())
    for x0, coords in lines_by_column.items():
        if len(coords["anti_x"]) >= 2:
            plt.plot(coords["anti_x"], coords["anti_y"], 'ro', linewidth=2, label='Anti-Stokes' if x0 == 0 else "")
        if len(coords["ray_x"]) >= 2:
            plt.plot(coords["ray_x"], coords["ray_y"], 'bo', linewidth=2, label='Rayleigh' if x0 == 0 else "")
        if len(coords["st_x"]) >= 2:
            plt.plot(coords["st_x"], coords["st_y"], 'go', linewidth=2, label='Stokes' if x0 == 0 else "")

    # Une seule légende pour la première ligne
    plt.legend()
    return detections





def analyze_patches_auto(signal, frq_lut,
                         h_patch=8, w_patch=8,
                         dy=2, dx=2,
                         min_prominence=0.02,
                         min_height=0.1):     # 10 % de l’amplitude max
    H, W = signal.shape
    detections = []

    for y0 in range(0, H - h_patch + 1, dy):
        for x0 in range(0, W - w_patch + 1, dx):
            patch = signal[y0:y0+h_patch, x0:x0+w_patch]
            frq_patch = frq_lut[y0:y0+h_patch, x0:x0+w_patch]

            # Projection en spectre 1D
            spectrum = patch.mean(axis=0)
            raw_max = spectrum.max()
            if raw_max == 0:
                continue

            # Détection avec seuils en intensité brute
            peaks, props = find_peaks(
                spectrum,
                prominence=min_prominence * raw_max,
                height=min_height * raw_max
            )
            if not len(peaks):
                continue

            frqs = frq_patch[h_patch//2]
            peak_frqs   = frqs[peaks]
            peak_pixels = [(x0 + int(p), y0 + h_patch//2) for p in peaks]

            detections.append({
                'origin': (x0, y0),
                'peak_frqs': peak_frqs,
                'peak_pixels': peak_pixels,
                'peak_props': props
            })

    return detections



def plot_detected_peaks(signal, results, frq_lut):
    plt.figure(figsize=(12, 6))
    plt.imshow(signal, cmap='gray', origin='lower', aspect='auto')
    
    color_map = {'Rayleigh': 'g', 'Stokes': 'r', 'Anti-Stokes': 'b'}
    marker_map = {'Rayleigh': 'o', 'Stokes': 's', 'Anti-Stokes': 'x'}

    for y, idx, kind, amp, ctr, gamma in results:
        # Interpolation position X à partir de la fréquence
        frq_row = frq_lut[y, :]
        x_pixel = np.interp(ctr, frq_row, np.arange(len(frq_row)))
        plt.plot(x_pixel, y, marker_map[kind], color=color_map[kind], label=kind if y == results[0][0] else "", markersize=6, alpha=0.7)

    plt.title("Pics détectés : Rayleigh (vert), Stokes (rouge), Anti-Stokes (bleu)")
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def freq_to_pixel(frq_lut_row, target_frq):
    # Cherche l'indice du pixel où frq_lut_row est le plus proche de target_frq
    idx = np.argmin(np.abs(frq_lut_row - target_frq))
    return idx

if __name__ == "__main__":
    width, height = 512, 1024
    rayleigh = (1.0, 0.0, 0.1)
    stokes = (0.6, 3, 0.3)
    antistokes = (0.5, -3, 0.3)
    n_triplets =3

    signal, frq_lut, centers = synthetic_signal(width, height, n_triplets, rayleigh, stokes, antistokes)
    
    from tifffile import imread

    sig = imread("brillouin_images/Quest_quartz_chamber_in_water_tube_razor_edge_100ms_no_slit.tif").astype(np.float32)
    sig /= np.max(sig)
    
        # Définir la région d’intérêt
    y_start, y_end = 100, 300
    x_start, x_end = 450, 600

    signal = sig[y_start:y_end, x_start:x_end]
    # print(np.max(signal))
    # signal = sig
    shift_brillouin = 11.0
    margin = 1.5      # couvre ±1.5×shift
    height,width = signal.shape
    freq_min = - shift_brillouin * margin   # → -16.5 GHz
    freq_max = + shift_brillouin * margin   # → +16.5 GHz
    freq_axis = np.linspace(freq_min, freq_max, width)  # en GHz
    frq_lut = np.tile(freq_axis, (height, 1))
    plt.figure(figsize=(10, 5))
    plt.imshow(signal, cmap='gray', origin='lower', aspect='auto')
    plt.title("Image synthétique (triplets Rayleigh / Stokes / Anti-Stokes)")
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")
    plt.colorbar(label="Intensité")
    plt.tight_layout()
    plt.show()
    # detections = analyze_patches_auto(
    # signal, frq_lut,
    # h_patch=10, w_patch=50,
    # dy=1, dx=2,
    # min_prominence=0.03,
    # min_height=0.8
    # )
    # Affichage
    # plt.figure(figsize=(10, 6))
    # plt.imshow(signal, cmap='gray', origin='lower', aspect='auto')
    # for det in detections:
    #     for x_pix, y_pix in det['peak_pixels']:
    #         plt.plot(x_pix, y_pix, 'ro', markersize=3)
    # plt.title("Detections par patchs glissants")
    # plt.show()
    # results = analyze_triplets(signal, frq_lut, y_range=range(0, height, 10), centers=centers)
    results = analyze_triplets_in_patches(
    signal, frq_lut,
    h_patch=10, w_patch=50,
    dy=5, dx=10,
    min_prominence=0.03,
    min_height=0.2
)

    # Avant de lancer l'analyse
    plt.figure(figsize=(10, 6))
    plt.imshow(signal, cmap='gray', origin='lower', aspect='auto')
    plt.title("Détection des Anti-Stokes sur la première ligne de patchs")
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")

    # Appel avec tracé direct dans la figure active
    detections = analyze_triplets_in_patches(signal, frq_lut,
                                            h_patch=10, w_patch=50,
                                            dy=5, dx=10,
                                            min_prominence=0.03, min_height=0.2)

    plt.tight_layout()
    plt.legend()
    plt.show()

    # results = analyze_triplets_auto(signal, frq_lut, y_range=range(0, height, 10))

    rayleigh_x, rayleigh_y = [],[]
    stokes_x,   stokes_y = [],[]
    antistokes_x, antistokes_y = [],[]
    
    plt.figure(figsize=(10, 6))
    plt.imshow(signal, cmap='gray', origin='lower', aspect='auto')
    plt.title("Détections de pics superposées")
    plt.xlabel("X (colonnes)")
    plt.ylabel("Y (lignes)")
    iter = 0 
    
    for res in results:
        seen=[]
        y_val = res['y']
        frq_row = frq_lut[y_val, :]  # tableau fréquence sur la ligne y
        iter=0
        triplets = {}
        mod = 0
        index=0
        for triplet in res['triplets']:
            key = str(index)
            if key not in triplets:
                triplets[key] = {}

            pixel_centers = [freq_to_pixel(frq_row, c) for c in triplet]  # convertit fréquence → pixel
            a, b, c = [pixel_centers[0], y_val], [pixel_centers[1], y_val], [pixel_centers[2], y_val]

            if a not in seen:
                plt.plot(pixel_centers[0], y_val, 'ro')
                triplets[key].setdefault("antistokes_x", []).append(pixel_centers[0])
                triplets[key].setdefault("antistokes_y", []).append(y_val)

            if b not in seen:
                plt.plot(pixel_centers[1], y_val, 'bo')
                triplets[key].setdefault("rayleigh_x", []).append(pixel_centers[1])
                triplets[key].setdefault("rayleigh_y", []).append(y_val)

            if c not in seen:
                plt.plot(pixel_centers[2], y_val, 'go')
                triplets[key].setdefault("stokes_x", []).append(pixel_centers[2])
                triplets[key].setdefault("stokes_y", []).append(y_val)

            seen.extend([a, b, c])

            mod += 1
            if mod == 3:
                mod = 0
                index += 1

            
            
        

            
            

    center_x = width / 2



    # 1) On a déjà rempli ces listes au-dessus :
    # 2) Calcul des moyennes
    ray_x_mean = np.mean(rayleigh_x) if rayleigh_x else None
    ray_y_mean = np.mean(rayleigh_y) if rayleigh_y else None

    st_x_mean = np.mean(stokes_x) if stokes_x else None
    st_y_mean = np.mean(stokes_y) if stokes_y else None

    anti_x_mean = np.mean(antistokes_x) if antistokes_x else None
    anti_y_mean = np.mean(antistokes_y) if antistokes_y else None

    # 3) Tracé des points moyens
    if ray_x_mean is not None:
        plt.scatter(ray_x_mean, ray_y_mean, c='r', s=100, marker='X', label='Rayleigh mean')
    if st_x_mean is not None:
        plt.scatter(st_x_mean, st_y_mean, c='b', s=100, marker='X', label='Stokes mean')
    if anti_x_mean is not None:
        plt.scatter(anti_x_mean, anti_y_mean, c='g', s=100, marker='X', label='Anti-Stokes mean')

    # 4) Mettre à jour la légende pour ces nouveaux points
    plt.legend()

    plt.tight_layout()
    plt.show()
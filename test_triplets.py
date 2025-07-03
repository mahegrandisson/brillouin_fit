'''TEST FIT EN SEPARANT LES TRIPLETS DANS LE STOCKAGE'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline

# Fonction de groupement des pics par espacement
def group_triplets_by_spacing(peaks, spacing_tol=0.05):
    triplets = []
    peaks = np.sort(peaks)
    for i in range(len(peaks) - 2):
        a, b, c = peaks[i], peaks[i+1], peaks[i+2]
        if np.all(np.diff([a, b, c]) > 0):
            d1 = b - a
            d2 = c - b
            if abs(d1 - d2) / max(d1, d2) < spacing_tol:
                triplets.append((a, b, c))
    return triplets

# Étape de détection et regroupement
def detect_and_group_triplets(signal, window_height=10, step=5):
    H, W = signal.shape
    all_triplet_points = []

    for y0 in range(0, H - window_height + 1, step):
        strip = signal[y0:y0 + window_height, :]
        profile = np.mean(strip, axis=0)
        norm_profile = profile / np.max(profile) if np.max(profile) != 0 else profile

        peaks, props = find_peaks(norm_profile, prominence=0.04, height=0.0)
        triplets = group_triplets_by_spacing(peaks)

        for triplet in triplets:
            anti, ray, stokes = triplet
            all_triplet_points.append({
                "anti": (anti, y0 + window_height // 2),
                "ray": (ray, y0 + window_height // 2),
                "stokes": (stokes, y0 + window_height // 2),
            })
    return all_triplet_points

# Regrouper les triplets détectés par ordre avec clustering spatial (DBSCAN)
def cluster_triplets(triplet_points, eps=20, min_samples=3):
    coords = [(t["ray"][0], t["ray"][1]) for t in triplet_points]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    labels = clustering.labels_
    clusters = {}

    for label, triplet in zip(labels, triplet_points):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = {"anti": [], "ray": [], "stokes": []}
        clusters[label]["anti"].append(triplet["anti"])
        clusters[label]["ray"].append(triplet["ray"])
        clusters[label]["stokes"].append(triplet["stokes"])
    return clusters

# Tracer les lignes moyennes pour chaque ordre
def plot_clusters_with_splines(signal, clusters):
    plt.figure(figsize=(12, 6))
    plt.imshow(signal, cmap='gray')

    for label, pts in clusters.items():
        for key, color in zip(["anti", "ray", "stokes"], ["r", "b", "g"]):
            pts_sorted = sorted(pts[key], key=lambda p: p[1])  # tri par y
            if len(pts_sorted) < 4:
                continue
            x_vals = [p[0] for p in pts_sorted]
            y_vals = [p[1] for p in pts_sorted]
            spline = UnivariateSpline(y_vals, x_vals, s=5)
            y_new = np.linspace(min(y_vals), max(y_vals), 200)
            x_new = spline(y_new)
            plt.plot(x_new, y_new, color=color, linewidth=2)

    plt.title("Triplets groupés par ordres avec lignes moyennes")
    plt.axis('off')
    plt.show()




if __name__ == "__main__":
    width, height = 512, 1024
    rayleigh = (1.0, 0.0, 0.1)
    stokes = (0.6, 3, 0.3)
    antistokes = (0.5, -3, 0.3)
    n_triplets =3

    # signal, frq_lut, centers = synthetic_signal(width, height, n_triplets, rayleigh, stokes, antistokes)
    
    from tifffile import imread

    sig = imread("brillouin_images/Quest_quartz_chamber_in_water_tube_razor_edge_100ms_no_slit.tif").astype(np.float32)
    sig /= np.max(sig)
    
        # Définir la région d’intérêt
    y_start, y_end = 100, 300
    x_start, x_end = 450, 600

    signal = sig[y_start:y_end, x_start:x_end]
    # Pipeline complet
    triplets = detect_and_group_triplets(signal, window_height=10, step=4)
    clusters = cluster_triplets(triplets, eps=30, min_samples=4)
    plot_clusters_with_splines(signal, clusters)
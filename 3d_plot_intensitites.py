'''PERMET DE PLOT EN 3D LES INTENSITES'''


import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

img_name = "Quest_quartz_chamber_in_water_tube_razor_edge_100ms_no_slit.tif"
image = tiff.imread("brillouin_images/"+img_name)
pixels = np.array(image)
plt.imshow(pixels)
plt.show()
threshold = 3000
pixels_thresh = np.where(pixels > threshold, pixels, 0)
height, width = pixels.shape
X, Y = np.meshgrid(np.arange(width), np.arange(height))
Z = pixels_thresh
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='hot', edgecolor='none')
fig.colorbar(surf, ax=ax, label="Intensité")
ax.set_title("Carte 3D de l'image (intensité des pixels)")
ax.set_xlabel("Colonne (X)")
ax.set_ylabel("Ligne (Y)")
ax.set_zlabel("Intensité (Z)")

plt.tight_layout()
plt.show()
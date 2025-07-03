'''Génerer de jolies images pour voir les intensités'''

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import os


for file in os.listdir("brillouin_images"):
    
    image = tiff.imread("brillouin_images/"+file)
    print("Forme de l'image :", image.shape)  

    pixels = np.array(image)


    print(pixels.shape)
    max = -1
    for i in range(pixels.shape[1]):
        summing = sum(pixels[:,i])
        if summing>max:
            max=summing
            index_max=i

    import numpy as np
    import matplotlib.pyplot as plt

    col_range = 5  # moyenne sur 5 colonnes
    start_col = index_max - col_range // 2
    end_col = index_max + col_range // 2 + 1


    selected_cols = pixels[:, start_col:end_col]

    sum_intensities = np.sum(pixels, axis=0)

    cols = np.arange(pixels.shape[1])

    # plt.plot(cols, sum_intensities)
    # plt.xlabel("Colonne")
    # plt.ylabel("Somme des intensités")
    # plt.title("Somme des intensités par colonne pour toute l'image")
    # plt.show()

    plt.imshow(pixels, cmap='hot')  # ou 'gray', 'viridis', 'hot', etc.
    plt.colorbar(label="Intensité")     # Ajoute une échelle d'intensité
    plt.title("Image avec intensité des pixels")
    plt.xlabel("Colonnes")
    plt.ylabel("Lignes")
    plt.show()
    plt.savefig("heatmap/" + file[:-4] + "_intensity_model.png")
    #plt.savefig("results/" + file[:-4] + "_intensity_model.png")
    
    
    
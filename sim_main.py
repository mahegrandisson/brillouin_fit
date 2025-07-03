
"SIMULER UN JOLI SIGNAL"
import matplotlib.pyplot as plt
from sim import synthetic_signal


if __name__ == "__main__":

    width = 1024
    height = 576
    order = 5
    rayleigh = [1, 0.0, 0.1]
    stokes = [0.8, 1.0, 0.15]
    antistokes = [0.6, -1.0, 0.2]

    synthetic_signal_img, y_pos, rayleigh_fit, frq_lut = synthetic_signal(width, height, order, rayleigh, stokes, antistokes)

    # Pour visualiser le signal
    plt.imshow(synthetic_signal_img, cmap='gray', aspect='auto')
    plt.title("Synthetic Signal")
    plt.colorbar()
    plt.show()
    

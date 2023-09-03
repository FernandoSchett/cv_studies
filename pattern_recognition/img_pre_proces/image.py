import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
from skimage.exposure import histogram

class Imagem:
    def __init__(self, img_path, image, name="imagem.png"):
        self.img_path = img_path
        self.orignal = image
        self.image = image
        self.name = name

    def reset(self):
        self.image = self.orignal

    def gen_hist_rgb(self, path):
        hist_red, bins = histogram(self.image[:, :, 0])
        hist_green, _ = histogram(self.image[:, :, 1])
        hist_blue, _ = histogram(self.image[:, :, 2])

        plt.figure(figsize=(10, 6))
        plt.title('RGB Histogram - ' + self.name)
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.plot(hist_red, color='red', label='Red Channel')
        plt.plot(hist_green, color='green', label='Green Channel')
        plt.plot(hist_blue, color='blue', label='Blue Channel')
        plt.legend()
        plt.grid()

        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'rgb_histogram.png')
        plt.savefig(path)
        plt.close()


    def gen_hist_opp(self, path):

        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        O1 = (r - g) / np.sqrt(2)
        O2 = (r + g - 2 * b) / np.sqrt(6)
        O3 = (r + g + b) / np.sqrt(3)
        hist_O1, bins_O1 = histogram(O1)
        hist_O2, bins_O2 = histogram(O2)
        hist_O3, bins_O3 = histogram(O3)

        plt.figure(figsize=(10, 6))
        plt.title("Histogramas de Oponentes - " + self.name)
        plt.plot(bins_O1, hist_O1, label="O1 (R-G)")
        plt.plot(bins_O2, hist_O2, label="O2 (Y-B)")
        plt.plot(bins_O3, hist_O3, label="O3 (Intensity)")
        plt.xlabel("Valor do Canal")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()
        
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'opp_histogram.png')
        plt.savefig(path)
        plt.close()

    
    def gen_tcolor_dist(self, path):
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        
        mean_r, std_r = np.mean(r), np.std(r)
        mean_g, std_g = np.mean(g), np.std(g)
        mean_b, std_b = np.mean(b), np.std(b)
        
        hist_r, bins_r = histogram((r - mean_r)/std_r)
        hist_g, bins_g = histogram((g - mean_g)/std_g)
        hist_b, bins_b = histogram((b - mean_b)/std_b)

        plt.figure(figsize=(10, 6))
        plt.title("Histogramas de Cores Transformadas - " + self.name)
        plt.plot(bins_r, hist_r, label="CTR Red")
        plt.plot(bins_g, hist_g, label="CTR Green")
        plt.plot(bins_b, hist_b, label="CTR Blue")

        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()
        plt.tight_layout()

        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'ctr_histogram.png')
        plt.savefig(path)
        plt.close()

    def gen_all(self, path):
        self.gen_hist_rgb(path)
        self.gen_hist_opp(path)
        self.gen_tcolor_dist(path)

    def save(self, path):
        io.imsave(path, skimage.img_as_ubyte(self.image))

    def int_change(self, value):
        self.image = np.clip(self.image + value, 0, 255).astype(np.uint8)
    
    def int_shift(self, value):
        self.image = np.clip(self.image * value, 0, 255).astype(np.uint8)
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
from skimage.exposure import histogram

class Histogram:
    def __init__(self, x=1, y=1, name = 'histogram.png'):
        self.name = name + '.png'
        self.x = x
        self.y = y
        self.num_imagens = 0
        plt.figure(figsize=(self.x*4, self.y*4))
        
    def plus_image(self):
        self.num_imagens += 1
    
    def change_pic(self, num):
        self.num_imagens = num
        plt.close()
        plt.figure(figsize=(self.x*4, self.y*4))

    def reset(self):
        plt.close()
        
    def save(self, path=""):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, self.name)
        plt.savefig(path)
        
    def gen_hist_rgb(self, imagem):
        r, g, b = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]

        O1 = (r - g) / np.sqrt(2)
        O2 = (r + g - 2 * b) / np.sqrt(6)
        O3 = (r + g + b) / np.sqrt(3)
        hist_O1, bins_O1 = histogram(O1)
        hist_O2, bins_O2 = histogram(O2)
        hist_O3, bins_O3 = histogram(O3)

        self.plus_image()
        plt.subplot(self.x, self.y, self.num_imagens)
        plt.title("Histogramas de Oponentes")
        plt.plot(bins_O1, hist_O1, label="O1 (R-G)")
        plt.plot(bins_O2, hist_O2, label="O2 (Y-B)")
        plt.plot(bins_O3, hist_O3, label="O3 (Intensity)")
        plt.xlabel("Valor do Canal")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()


    def gen_tcolor_dist(self, path):
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        
        mean_r, std_r = np.mean(r), np.std(r)
        mean_g, std_g = np.mean(g), np.std(g)
        mean_b, std_b = np.mean(b), np.std(b)
        
        hist_r, bins_r = histogram((r - mean_r)/std_r)
        hist_g, bins_g = histogram((g - mean_g)/std_g)
        hist_b, bins_b = histogram((b - mean_b)/std_b)
        
        self.plus_image()
        plt.subplot(self.x, self.y, self.num_imagens)
        plt.title("Histogramas de Cores Transformadas")
        plt.plot(bins_r, hist_r, label="CTR Red")
        plt.plot(bins_g, hist_g, label="CTR Green")
        plt.plot(bins_b, hist_b, label="CTR Blue")

        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()
        plt.tight_layout()
    
    def gen_hist_rgb(self, path):
        hist_red, bins = histogram(self.image[:, :, 0])
        hist_green, _ = histogram(self.image[:, :, 1])
        hist_blue, _ = histogram(self.image[:, :, 2])

        plt.figure(figsize=(10, 6))
        plt.title('RGB Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.plot(hist_red, color='red', label='Red Channel')
        plt.plot(hist_green, color='green', label='Green Channel')
        plt.plot(hist_blue, color='blue', label='Blue Channel')
        plt.legend()
        plt.grid()
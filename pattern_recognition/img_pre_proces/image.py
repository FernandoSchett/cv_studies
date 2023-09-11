import os
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import io
from skimage.exposure import histogram
from skimage import exposure

class Imagem:
    def __init__(self, img_path, image, name="imagem.png"):
        self.img_path = img_path
        self.orignal = image
        self.image = image.copy()
        self.name = name
    
    def reset(self):
        self.image = self.orignal
    
    def reset_hist():
        plt.close()

    def save(self, path):
        io.imsave(path, skimage.img_as_ubyte(self.image))

    def int_change(self, value):
        self.image = self.image * value
        #self.image = exposure.rescale_intensity(self.image, in_range=(0, 255)).astype('uint16')
        
    def int_shift(self, value):
        self.image = self.image + value
        #self.image = exposure.rescale_intensity(self.image, in_range=(0, 255)).astype('uint16')

    def normalize(self):
        self.image = np.round(self.image).astype(int)
        #self.image = exposure.rescale_intensity(self.image, in_range=(0, 255), out_range='uint8').astype('uint16')

        #for channel in range(3):
        #    min_arr = np.min(self.image[:, :, channel])
        #    max_arr = np.max(self.image[:, :, channel])
        #    self.image[:, :, channel] = ((self.image[:, :, channel] - min_arr) / (max_arr - min_arr)) * (255 - 0) + 0
        #    self.image[:, :, channel] = np.round(self.image[:, :, channel]).astype(int)

    def save_hist(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, 'histogram.png')
        plt.savefig(path)


    def gen_hist_rgb(self):
        hist_red, bins_red = histogram(self.image[:, :, 0])
        hist_green, bins_green = histogram(self.image[:, :, 1])
        hist_blue, bins_blue = histogram(self.image[:, :, 2])
        #print(self.image[:, :, 0])
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        plt.title(self.get_name())
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.plot(bins_red, hist_red, color='red', label='Red Channel')
        plt.plot(bins_green, hist_green, color='green', label='Green Channel')
        plt.plot(bins_blue, hist_blue, color='blue', label='Blue Channel')
        plt.legend()
        plt.grid()

    def gen_hist_opp(self):

        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        O1 = (r - g) / np.sqrt(2)
        O2 = (r + g - 2 * b) / np.sqrt(6)
        O3 = (r + g + b) / np.sqrt(3)
        hist_O1, bins_O1 = histogram(O1)
        hist_O2, bins_O2 = histogram(O2)
        hist_O3, bins_O3 = histogram(O3)
        #print(O1)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        plt.title(self.get_name())
        plt.plot(bins_O1, hist_O1, label="O1 (R-G)")
        plt.plot(bins_O2, hist_O2, label="O2 (Y-B)")
        plt.plot(bins_O3, hist_O3, label="O3 (Intensity)")
        plt.xlabel("Valor do Canal")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()

    
    def gen_tcolor_dist(self):
        r, g, b = self.image[:, :, 0], self.image[:, :, 1], self.image[:, :, 2]
        #print(self.image[:, :, 0])
        mean_r, std_r = np.mean(r), np.std(r)
        mean_g, std_g = np.mean(g), np.std(g)
        mean_b, std_b = np.mean(b), np.std(b)
        #print((r - mean_r)/std_r)
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        hist_r, bins_r = histogram((r - mean_r)/std_r)
        hist_g, bins_g = histogram((g - mean_g)/std_g)
        hist_b, bins_b = histogram((b - mean_b)/std_b)

        plt.title(self.get_name())
        plt.plot(bins_r, hist_r, label="CTR Red")
        plt.plot(bins_g, hist_g, label="CTR Green")
        plt.plot(bins_b, hist_b, label="CTR Blue")

        plt.xlabel("Valor")
        plt.ylabel("Frequência")
        plt.legend()
        plt.grid()

    # Getter for name
    def get_name(self):
        return self.name
    
    # Setter for name
    def set_name(self, name):
        self.name = name
    
    # Getter for image
    def get_image(self):
        return self.image
    
    # Setter for image
    def set_image(self, image):
        self.image = image
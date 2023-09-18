"""
File:           transformer.py
Last changed:   17/09/2023 18:45
Purpose:        Pre-Proccessing image fuctions         
Authors:        Fernando Antonio Marques Schettini   
Usage: 
    HowToExecute:   python3 transformer.py       
Dependecies:
    skimage
    matplotlib
    numpy
"""
from skimage import exposure, filters, transform, color
import skimage

class Transformer:

    @staticmethod
    def equalize_hist(image):
        equalized_image = exposure.equalize_hist(image)  
        return equalized_image

    @staticmethod
    def to_gray(image):
        gray_image = color.rgb2gray(image)
        gray_image = skimage.img_as_ubyte(gray_image)
        return gray_image

    @staticmethod
    def gamma_adjust(image, gamma, gain):
        gamma_image = exposure.adjust_gamma(image, gamma, gain)
        return gamma_image

    @staticmethod
    def gauss_filter(image, sigma):
        gauss_image = filters.gaussian(image, sigma,channel_axis=None)
        return gauss_image

    @staticmethod
    def laplace_filter(image):
        laplace_image = filters.laplace(image)
        return laplace_image
    
    @staticmethod
    def alterar_tam(image, tam):
        imagem_tam = transform.resize(image, tam)
        return imagem_tam
    
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    shape = (256, 256)
    random_gray_image = np.random.rand(*shape)

    equalized_image = Transformer.equalize_hist(random_gray_image)
    gamma_image = Transformer.gamma_adjust(random_gray_image, gamma=1.5, gain=1)
    gauss_image = Transformer.gauss_filter(random_gray_image, sigma=2)
    laplace_image = Transformer.laplace_filter(random_gray_image)
    resized_image = Transformer.alterar_tam(random_gray_image, (128, 128))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(random_gray_image, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("Eq Hist")
    plt.imshow(equalized_image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Gamma")
    plt.imshow(gamma_image, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("Gauss")
    plt.imshow(gauss_image, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Laplace")
    plt.imshow(laplace_image, cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("Resize")
    plt.imshow(resized_image, cmap='gray')

    plt.tight_layout()
    plt.show()
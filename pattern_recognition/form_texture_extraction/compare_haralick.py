from skimage import io, feature
import matplotlib.pyplot as plt
import numpy as np

# Função para calcular e plotar o histograma da GLCM
def plot_glcm_histogram(glcm, title):
    histograma = glcm.flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(histograma)), histograma, color='gray')
    plt.title(title)
    plt.xlabel('Nível de Cinza')
    plt.ylabel('Frequência')
    plt.savefig(title +'.png')
    plt.show()

# Carregue as três imagens
image1 = io.imread("treino_he (66)_gaussian_4.jpg", as_gray=True)
image2 = io.imread("treino_pas (20)_gamma_0.5.jpg", as_gray=True)
image3 = io.imread("treino_picro (5)_equalized.jpg", as_gray=True)

# Defina os parâmetros para calcular a GLCM
distances = [1]  
angles = [0]  
levels = 256  
symmetric = True  
normalized = True  

# Calcule a matriz GLCM e plote o histograma para cada imagem
glcm1 = feature.graycomatrix(image1, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normalized)
plot_glcm_histogram(glcm1, 'Histograma da GLCM - Imagem 1')

glcm2 = feature.graycomatrix(image2, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normalized)
plot_glcm_histogram(glcm2, 'Histograma da GLCM - Imagem 2')

glcm3 = feature.graycomatrix(image3, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normalized)
plot_glcm_histogram(glcm3, 'Histograma da GLCM - Imagem 3')

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, feature
from skimage.color import rgb2gray
from skimage.transform import resize
import os

# Função para calcular e mostrar o LBP de uma imagem
def calculate_and_display_lbp(image_path, n_points, radius, method):
    # Carregar a imagem
    image = io.imread(image_path)
    
    # Converter a imagem para tons de cinza se não estiver
    if image.shape[-1] == 3:
        image = rgb2gray(image)

    
    # Calcular o LBP
    lbp_image = feature.local_binary_pattern(image, n_points, radius, method)
    
    # Plotar a imagem original e o LBP
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Imagem Original')
    
    axes[1].imshow(lbp_image, cmap='gray')
    axes[1].set_title('LBP (n_points={}, radius={})'.format(n_points, radius))
    
    plt.show()

    print(lbp_image)
    
    # Extrair o nome do arquivo da imagem
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Salvar a imagem LBP com o nome adequado
    lbp_image_name = f"{image_name}_lbp.png"
    #plt.close()
    #plt.imshow(lbp_image, cmap='gray')
    #plt.savefig(lbp_image_name)
    #plt.close()
    
    io.imsave(lbp_image_name, ((lbp_image * 255)/8).astype('uint8'))

# Definir parâmetros
n_points = 8  # Número de pontos a serem considerados
radius = 1    # Raio da vizinhança

# Lista de imagens
imagens = [
    "treino_he (66)_gaussian_4.jpg",
    "treino_pas (20)_gamma_0.5.jpg",
    "treino_picro (5)_equalized.jpg",

]

# Loop sobre as imagens e calcular o LBP para cada uma
for imagem in imagens:
    image_path = imagem
    calculate_and_display_lbp(image_path, n_points, radius, method='uniform')

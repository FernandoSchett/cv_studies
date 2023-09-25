from skimage import io, filters
import os
from skimage import img_as_ubyte

# Lista dos nomes dos arquivos das imagens
imagens = [
    "treino_he (66)_gaussian_4.jpg",
    "treino_pas (20)_gamma_0.5.jpg",
    "treino_picro (5)_equalized.jpg"
]

# Iterando sobre cada imagem
for imagem in imagens:
    # Lendo a imagem em escala de cinza
    img = io.imread(imagem, as_gray=True)
    
    # Aplicando o operador Sobel
    edge_sobel = filters.sobel(img)
    edge_sobel = img_as_ubyte(edge_sobel)

    print(edge_sobel)
    # Construindo o nome do arquivo de saída
    base, ext = os.path.splitext(imagem)
    output_filename = f"{base}_sobel{ext}"
    
    # Salvando a imagem resultante
    io.imsave(output_filename, edge_sobel)

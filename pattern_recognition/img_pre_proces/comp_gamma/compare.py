import skimage.io as io
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma

# Carregando as imagens
image1 = io.imread("./treino_azan (9).jpg")
image2 = io.imread("./treino_pams (1).jpg")

# Convertendo as imagens para cinza
gray_image1 = rgb2gray(image1)
gray_image2 = rgb2gray(image2)

# Aplicando a correção gama para diferentes valores
gammas = [0.5, 1.5]  # Você pode alterar esses valores conforme necessário
gamma_adjusted_images1 = [adjust_gamma(gray_image1, gamma=g) for g in gammas]
gamma_adjusted_images2 = [adjust_gamma(gray_image2, gamma=g) for g in gammas]

# Salvando as imagens originais em cinza
io.imsave("./treino_azan (9)_gray.jpg", (gray_image1 * 255).astype('uint8'))
io.imsave("./treino_pams (1)_gray.jpg", (gray_image2 * 255).astype('uint8'))

# Salvando as imagens ajustadas com gamma
for g, g_adj_image1, g_adj_image2 in zip(gammas, gamma_adjusted_images1, gamma_adjusted_images2):
    io.imsave(f"./treino_azan (9)_gamma_{g}.jpg", (g_adj_image1 * 255).astype('uint8'))
    io.imsave(f"./treino_pams (1)_gamma_{g}.jpg", (g_adj_image2 * 255).astype('uint8'))

import skimage.io as io
from skimage.color import rgb2gray
from skimage.exposure import adjust_gamma

# Carregando as imagens
image1 = io.imread("./treino_he (66).jpg")
image2 = io.imread("./treino_pas (20).jpg")
image3 = io.imread("./treino_picro (5).jpg")

# Convertendo as imagens para cinza
gray_image1 = rgb2gray(image1)
gray_image2 = rgb2gray(image2)
gray_image3 = rgb2gray(image3)

# Aplicando a correção gama para diferentes valores
gammas = [0.5, 1.5]  # Você pode alterar esses valores conforme necessário
gamma_adjusted_images1 = [adjust_gamma(gray_image1, gamma=g) for g in gammas]
gamma_adjusted_images2 = [adjust_gamma(gray_image2, gamma=g) for g in gammas]
gamma_adjusted_images3 = [adjust_gamma(gray_image3, gamma=g) for g in gammas]

# Salvando as imagens ajustadas com gamma
for g, g_adj_image1, g_adj_image2, g_adj_image3 in zip(gammas, gamma_adjusted_images1, gamma_adjusted_images2, gamma_adjusted_images3):
    io.imsave(f"./treino_he (66)_gamma_{g}.jpg", (g_adj_image1 * 255).astype('uint8'))
    io.imsave(f"./treino_pas (20)_gamma_{g}.jpg", (g_adj_image2 * 255).astype('uint8'))
    io.imsave(f"./treino_picro (5)_gamma_{g}.jpg", (g_adj_image3 * 255).astype('uint8'))

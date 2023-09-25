import skimage.io as io
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import laplace, gaussian

# Carregando as imagens
image1 = io.imread("./treino_he (66).jpg")
image2 = io.imread("./treino_pas (20).jpg")
image3 = io.imread("./treino_picro (5).jpg")

# Convertendo as imagens para cinza
gray_image1 = rgb2gray(image1)
gray_image2 = rgb2gray(image2)
gray_image3 = rgb2gray(image3)

# Aplicando o filtro gaussiano com diferentes sigmas
sigmas = [4, 16, 32]
gaussian_images1 = [gaussian(gray_image1, sigma=s) for s in sigmas]
gaussian_images2 = [gaussian(gray_image2, sigma=s) for s in sigmas]
gaussian_images3 = [gaussian(gray_image3, sigma=s) for s in sigmas]


# Redimensionando as imagens
resized_gray_image1 = resize(gray_image1, (1024, 1024))
resized_gray_image2 = resize(gray_image2, (1024, 1024))
resized_gray_image3 = resize(gray_image3, (1024, 1024))

for s, g_image1, g_image2, g_image3 in zip(sigmas, gaussian_images1, gaussian_images2, gaussian_images3):
    io.imsave(f"./treino_he (66)_gaussian_{s}.jpg", (g_image1 * 255).astype('uint8'))
    io.imsave(f"./treino_pas (20)_gaussian_{s}.jpg", (g_image2 * 255).astype('uint8'))
    io.imsave(f"./treino_picro (5)_gaussian_{s}.jpg", (g_image3 * 255).astype('uint8'))

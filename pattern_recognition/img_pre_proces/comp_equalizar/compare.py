import skimage.io as io
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.transform import resize


# Carregando as imagens
image1 = io.imread("./treino_azan (9).jpg")
image2 = io.imread("./treino_pams (1).jpg")

# Convertendo as imagens para cinza
gray_image1 = rgb2gray(image1)
gray_image2 = rgb2gray(image2)

equalized_image1 = equalize_hist(gray_image1)
equalized_image2 = equalize_hist(gray_image2)

resized_gray_image1 = resize(gray_image1, (1024, 1024))
resized_gray_image2 = resize(gray_image2, (1024, 1024))
resized_equalized_image1 = resize(equalized_image1, (1024, 1024))
resized_equalized_image2 = resize(equalized_image2, (1024, 1024))

# Salvando as imagens em cinza
io.imsave("./treino_azan (9)_gray.jpg", (gray_image1 * 255).astype('uint8'))
io.imsave("./treino_pams (1)_gray.jpg", (gray_image2 * 255).astype('uint8'))

# Salvando as imagens com histograma equalizado
io.imsave("./treino_azan (9)_equalized.jpg", (equalized_image1 * 255).astype('uint8'))
io.imsave("./treino_pams (1)_equalized.jpg", (equalized_image2 * 255).astype('uint8'))

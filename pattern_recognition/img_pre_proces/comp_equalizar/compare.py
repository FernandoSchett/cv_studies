import skimage.io as io
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.transform import resize


# Carregando as imagens
image1 = io.imread("./treino_he (66).jpg")
image2 = io.imread("./treino_pas (20).jpg")
image3 = io.imread("./treino_picro (5).jpg")

# Convertendo as imagens para cinza
gray_image1 = rgb2gray(image1)
gray_image2 = rgb2gray(image2)
gray_image3 = rgb2gray(image3)

equalized_image1 = equalize_hist(gray_image1)
equalized_image2 = equalize_hist(gray_image2)
equalized_image3 = equalize_hist(gray_image3)

resized_gray_image1 = resize(gray_image1, (1024, 1024))
resized_gray_image2 = resize(gray_image2, (1024, 1024))
resized_gray_image3 = resize(gray_image3, (1024, 1024))

resized_equalized_image1 = resize(equalized_image1, (1024, 1024))
resized_equalized_image2 = resize(equalized_image2, (1024, 1024))
resized_equalized_image3 = resize(equalized_image3, (1024, 1024))

# Salvando as imagens com histograma equalizado
io.imsave("././treino_he (66)_equalized.jpg", (equalized_image1 * 255).astype('uint8'))
io.imsave("./treino_pas (20)_equalized.jpg", (equalized_image2 * 255).astype('uint8'))
io.imsave("./treino_picro (5)_equalized.jpg", (equalized_image3 * 255).astype('uint8'))


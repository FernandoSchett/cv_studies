from skimage import exposure, filters, transform, color

class Transformer:

    @staticmethod
    def equalize_hist(image):
        equalized_image = exposure.equalize_hist(image.get_image())  
        return equalized_image

    @staticmethod
    def to_gray(image):
        gray_image = color.rgb2gray(image.get_image())
        return gray_image

    @staticmethod
    def gamma_adjust(image, gamma, gain):
        gamma_image = exposure.adjust_gamma(image.get_image(), gamma, gain)
        return gamma_image

    @staticmethod
    def gauss_filter(image, sigma):
        gauss_image = filters.gaussian(image.get_image(), sigma, multichannel=True)
        return gauss_image

    @staticmethod
    def laplace_filter(image):
        laplace_image = filters.laplace(image.get_image())
        return laplace_image
    
    @staticmethod
    def alterar_tam(image, tam):
        imagem_tam = transform.resize(image.get_image(), tam)
        return imagem_tam
    
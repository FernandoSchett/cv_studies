from skimage import io

class Imagem:
    def __init__(self, img_path, image, name="imagem"):
        self.img_path = img_path
        self.orignal = image
        self.image = image
        self.name = name
    
    def reset(self):
        self.image = self.orignal

    def save(self, path):
        io.imsave(path, self.image)

    # Gets e Sets
    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
    
    def get_image(self):
        return self.image
    
    def set_image(self, image):
        self.image = image
    
    def get_path(self):
        return self.img_path
    
    def set_path(self, image):
        self.img_path = image
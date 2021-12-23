import numpy as np
from PIL import Image, ImageFilter

#Loading and resizinf an image using PIL, and return its pixel data 

def pil_resize_image(image_path, resize_shape, image_mode = 'RGBA', image_filter = None):
    im = Image.open(image_path)
    converted_im = im.convert(image_mode)
    resized_im = converted_im.resize(resize_shape, Image.LANCZOS)
    if image_filter is not None:
        resized_im = resized_im.filter(image_filter)
    im_data = resized_im.getdata()

    return np.asarray(im_data)

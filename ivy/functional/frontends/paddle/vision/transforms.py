#global

import numpy as np
from PIL import Image
from . import transforms

def adjust_brightness(img, brightness_factor):
    
    #   0 gives a black image, 1 gives the original image while 2 doubles the brightness

    fake_img_np = np.array(img)
    fake_img = Image.fromarray(fake_img_np)
    converted_img = transforms.adjust_brightness(fake_img, brightness_factor)
    return converted_img

def adjust_contrast(img, contrast_factor):

    #   0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2

    fake_img_np = np.array(img)
    fake_img = Image.fromarray(fake_img_np)
    converted_img = transforms.adjust_contrast(fake_img, contrast_factor)
    return converted_img

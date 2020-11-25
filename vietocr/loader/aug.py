from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.aug = iaa.SomeOf(1, 
        [sometimes(iaa.GaussianBlur(sigma=(0, 1.0))),
        sometimes(iaa.MotionBlur(k=3)),
        sometimes(iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)),
        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))),
        sometimes(iaa.Invert(0.25, per_channel=0.5)),
        sometimes(iaa.Rotate((-20, 20)))
    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img


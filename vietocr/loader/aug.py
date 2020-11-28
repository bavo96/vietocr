from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia

class ImgAugTransform:
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.aug = iaa.SomeOf(3, 
        [
        sometimes(iaa.AddToHueAndSaturation((-50, 50), per_channel=True) ),
        sometimes(iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True)),
        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.02))),
        sometimes(iaa.Invert(0.7, per_channel=0.8)),
        sometimes(iaa.Rotate((-20, 20)))
    ])
      
  def __call__(self, img):
    img = np.array(img)
    img = self.aug.augment_image(img)
    img = Image.fromarray(img)
    return img

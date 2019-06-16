# -*- coding: utf-8 -*-

"""
Post processing for OCR
"""

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import disk
from skimage import exposure
from skimage.filters import (threshold_niblack,  threshold_sauvola)
from skimage.filters.rank import median


class Preprocessing:
    def __init__(self):
        None
        
    def contrast_stretching(img):
        """
        Contrast stretching using scikit-image
        """
        p2,  p98 = np.percentile(img,  (2,  98))
        img_filtered = exposure.rescale_intensity(img,  in_range=(p2,  p98))
        img_filtered = (img_filtered*255.0/np.max(img_filtered)).astype(np.uint8)
        
        return img_filtered
        
    def general_enhancement(img,  method_type):
        """
        General function for enhancing the coin
        """
        if method_type[0] == "guidedFilter":
            # Guided Filter : Edge preserving filtering
            if len(method_type) == 3:
                img_filtered = cv2.guidedFilter(img,  method_type[1], method_type[2])
            else:
                radius = max(5, 0.3*int(len(img)))
                # eps**2 is similar to sigmaColor in bilateralFilter
                eps = 10
                img_filtered = cv2.guidedFilter(img,  radius,  eps)
        elif method_type[0] == "bilateralFilter":
            # bilateralFilter : Edge preserving filtering
            if len(method_type) == 4:
                img_filtered = cv2.guidedFilter(img,  method_type[1], method_type[2], method_type[3])
            else:
                """                
                Filter size: Large filters (d > 5) are very slow,  so it is recommended to use d = 5 for real-time applications, 
                and perhaps d = 9 for offline applications that need heavy noise filtering.
                
                Sigma values: For simplicity,  you can set the 2 sigma values to be the same. 
                If they are small (< 10), the filter will not have much effect,  
                whereas if they are large (> 150), they will have a very strong effect,  making the image look “cartoonish”.
                """
                # The kernel size. This is the neighborhood where the local variance will be calculated,
                # and where pixels will contribute (in a weighted manner).
                d = 30
                # Filter sigma in the color space. A larger value of the parameter means that farther colors within
                # the pixel neighborhood (see sigmaSpace ) will be mixed together,  resulting in larger
                # areas of semi-equal color
                sigmaColor = 50
                # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels
                # will influence each other as long as their colors are close enough (see sigmaColor ).
                # When d>0 , it specifies the neighborhood size regardless of sigmaSpace .
                # Otherwise,  d is proportional to sigmaSpace .
                sigmaSpace = 0
                
                img_filtered = cv2.bilateralFilter(img,  d,  sigmaColor,  sigmaSpace)

    def hue_enhance(img):
        """
        Convert dark or light areas to black
        """
        # Equalize Hue part
        __h,  __s,  __v = Preprocessing.rgb2hsv(img[:, :, 0].astype('double')/255.0,  img[:, :, 1].astype('double')/255.0,  img[:, :, 2].astype('double')/255.0)
        __h[__h > 0.1] = np.median(__h[__h > 0.1])
                
        __r,  __g,  __b = Preprocessing.hsv2rgb( __h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        
        return (np.concatenate((__r[..., np.newaxis], __g[..., np.newaxis], __b[..., np.newaxis]),
                               axis=len(np.shape(__r)))). astype(np.uint8)

    def thresholding(im,  type='Sauvola', window_ratio=0.15, k=0.8):
        """
        Thresholding
        """
        if type == 'Otsu':
            # Otsu thresholding
            _,  im_thr = cv2.threshold((im*255.0/np.max(im)).astype(np.uint8), 0,  255,
                                       cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        elif type == 'Niblack':
            thresh_niblack = threshold_niblack(im,  window_size=int(2*np.floor(np.shape(im)[0]*window_ratio/2)+1), k=k)
            im_thr = 255*((im < thresh_niblack).astype(np.uint8))
        elif type == 'Sauvola':
            thresh_sauvola = threshold_sauvola(im,  window_size=int(2*np.floor(np.shape(im)[0]*window_ratio/2)+1), k=k)
            im_thr = 255*((im < thresh_sauvola).astype(np.uint8))
                    
        return im_thr
    
    def median_filtering(img, size=2, repeat=2):
        for n in range(0,  repeat):
            img = median(img,  disk(size))
        
        return img
    
    def hue_enhance(self):
        """
        Convert shaded/light parts to black (background filtering)
        """
        # The kernel size. This is the neighborhood where the local variance will be calculated,
        # and where pixels will contribute (in a weighted manner).
        d = int(10.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        # Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel
        # neighborhood (see sigma_space ) will be mixed together,  resulting in larger areas of semi-equal color
        sigma_color = int(80.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will
        # influence each other as long as their colors are close enough (see sigma_color ).
        # When d>0 , it specifies the neighborhood size regardless of sigma_space .
        # Otherwise,  d is proportional to sigma_space .
        sigma_space = int(80.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        self.stripe = cv2.bilateralFilter(self.stripe.astype(np.uint8), d,  sigma_color,  sigma_space)
        
        # Equalize Hue part
        __h,  __s,  __v = self.rgb2hsv(self.stripe[:, :, 0].astype('double')/255.0,
                                       self.stripe[:, :, 1].astype('double')/255.0,
                                       self.stripe[:, :, 2].astype('double')/255.0)
        __h[__h > 0.1]  = np.median(__h[__h > 0.1])
        __r,  __g,  __b = self.hsv2rgb(__h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        self.stripe = (np.concatenate((__r[..., np.newaxis], __g[..., np.newaxis], __b[..., np.newaxis]),
                                      axis=len(np.shape(__r)))).astype(np.uint8)
        
        # Thresholding
        __h,  __s,  __v = self.rgb2hsv(self.stripe[:, :, 0].astype('double')/255.0,
                                       self.stripe[:, :, 1].astype('double')/255.0,
                                       self.stripe[:, :, 2].astype('double')/255.0)
        __ms = np.median(1.0*__s[np.isfinite(__s)])
        __mv = np.median(1.0*__v[np.isfinite(__v)])
        
        # Threshold abnormal points
        """
        seuil = 0.15 
        ind_cont = np.logical_and(np.logical_or(__s<(__ms-seuil),(__ms+seuil)<__s),np.logical_or(__v<(__mv-seuil),(0.95)<__v))        
        __s[ind_cont]  = 1.0
        __v[ind_cont]  = 0.0
        """
        # Equalize normal points
        seuil = 0.05
        ind_cont = np.logical_and(np.logical_and((__ms-seuil) < __s,  __s < (__ms+seuil)),
                                  np.logical_and((__mv-seuil) < __v,  __v < (__mv+seuil)))
        __s[ind_cont] = __ms
        __v[ind_cont] = __mv
        
        __r,  __g,  __b = self.hsv2rgb(__h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        self.stripe = (np.concatenate((__r[..., np.newaxis],
                                       __g[..., np.newaxis],
                                       __b[..., np.newaxis]), axis=len(np.shape(__r)))).astype(np.uint8)
              
        # Enhance contrast : image = Image.open('downloads/jcfeb2011.jpg') http://pillow.readthedocs.io/en/3.3.x/reference/ImageEnhance.html
        # Sharpen
        #ImageEnhance.Sharpness(image).enhance(2)
        # Smooth 
        #ImageEnhance.Sharpness(image).enhance(0.5)
        # Enhance Brightness
        #enhancerBrightness = ImageEnhance.Sharpness(Image.fromarray(self.stripe.astype(np.uint8)))
        self.stripe = np.asarray(ImageEnhance.Brightness(Image.fromarray(self.stripe.astype(np.uint8))).enhance(1.1))

    """
    COLOR FRAME CONVERSION
    """
    @staticmethod
    def rgb2hsv(r,  g,  b):
        """
        Conversion functions between RGB (Red,  Green,  Blue components) and
        HSV (Hue:position in the spectrum,  Saturation:color saturation ("purity"), Value:color brightness)
        """
        __rgb = np.concatenate((r[..., np.newaxis], g[..., np.newaxis], b[..., np.newaxis]), axis=len(np.shape(r)))
        maxc = np.max(__rgb,  axis=len(np.shape(r))).astype('float')
        minc = np.min(__rgb,  axis=len(np.shape(r))).astype('float')
        
        s = (maxc-minc) / maxc
        
        rc = (maxc-r) / (maxc-minc+0.00000000000001)
        gc = (maxc-g) / (maxc-minc+0.00000000000001)
        bc = (maxc-b) / (maxc-minc+0.00000000000001)
        
        v = maxc
        
        h = (minc == maxc)*np.zeros(np.shape(r)) + (minc != maxc)*((r == maxc)*(bc-gc) +
                                                                   (r != maxc)*(g == maxc)*(2.0+rc-bc) +
                                                                   (r != maxc)*(g != maxc)*(b == maxc)*(4.0+gc-rc))
        h = (h/6.0) % 1.0
        
        h = np.asarray(h)
        s = np.asarray(s)
        v = np.asarray(v)
        
        return h,  s,  v

    @staticmethod
    def hsv2rgb(h,  s,  v):
        """
        Conversion functions between RGB (Red,  Green,  Blue components) and
        HSV (Hue:position in the spectrum,  Saturation:color saturation ("purity"), Value:color brightness)
        """
        i = (h*6.0).astype(np.uint8) # assumes int truncates!
        f = (h*6.0) - i
        p = v*(1.0 - s)
        q = v*(1.0 - s*f)
        t = v*(1.0 - s*(1.0-f))
        i = i % 6.0
        
        r = (s == 0.0)*v+(s != 0.0)*((i == 0)*v+(i == 1)*q+(i == 2)*p+(i == 3)*p+(i == 4)*t+(i == 5)*v)
        g = (s == 0.0)*v+(s != 0.0)*((i == 0)*t+(i == 1)*v+(i == 2)*v+(i == 3)*q+(i == 4)*p+(i == 5)*p)
        b = (s == 0.0)*v+(s != 0.0)*((i == 0)*p+(i == 1)*p+(i == 2)*t+(i == 3)*v+(i == 4)*v+(i == 5)*q)
        
        return r,  g,  b

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 21:49:29 2016

@author: victor
"""
import sys
import cv2
import numpy as np
from PIL import Image,  ImageEnhance
from skimage.feature import (match_descriptors,  ORB,  match_template)
from skimage.filters import threshold_niblack
from skimage.color import rgb2gray
from skimage.restoration import denoise_nl_means

from Algorithm import Template, Preprocessing


class AversRevers:
    
    def __init__(self,  quelcote):
        self.side_type = quelcote
        # image extraite du cote extrait
        self.im_BRG = None
        # coordonnfe du cercle dans l'image originale [x,  y,  rayon]
        self.circleCoordinates = None
        # coordonnee du rectangle pour extractio dans l'image originale [xdebut,  xfin,  ydebut,  yfin]
        self.boundaries = None
        self.radius = None
        self.stripe = None
        self.marge_hor = np.array([0,  0])
        self.marge_vert = None
        # Criteres pour l'extraction des blocs de lettres en pourcentage de rayon : 
        # position verticale en partant du haut# taille horizontale# taille verticale # deviation autour de la mediane
        self.crit_shape = 0.05,  0.3,  0.02,  0.3,  0.1,  0.2,  0.05 # en pourcentage
        # Initialization matching,  number of descriptor
        self.Ndescr = 300
        # OCR text results
        self.OCR_results = None

    """
    IMAGE IMPROVEMENT
    """
    def enhance_image(self,  method_type):
        """
        Image enhancing (removing details while preserving edges) using opencv
        """
        if method_type[0] == "guidedFilter":
            if len(method_type) == 3:
                self.im_BRG = cv2.guidedFilter(self.im_BRG,  method_type[1], method_type[2])
            else:
                radius = 10
                # eps**2 is similar to sigma_color in bilateralFilter
                eps = 10
                self.im_BRG = cv2.guidedFilter(self.im_BRG,  radius,  eps)

        elif method_type[0] == "bilateralFilter":
            if len(method_type) == 4:
                self.im_BRG = cv2.guidedFilter(self.im_BRG,  method_type[1], method_type[2], method_type[3])
            else:
                # filter size (bwr)
                d = 9
                # sigma_color small~10 and big~150
                sigma_color = 50
                # sigma_space small~10 and big~150
                sigma_space = 50
                self.im_BRG = cv2.bilateralFilter(self.im_BRG,  d,  sigma_color,  sigma_space)

        elif method_type[0] == "fastNlMeansDenoisingColored":
            self.coin_brg = cv2.fastNlMeansDenoisingColored(self.coin_brg, templateWindowSize=7, searchWindowSize=21)
            # Local threshold
            self.coin_gray = cv2.adaptiveThreshold(self.coin_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 4)

        elif method_type[0] == "nonLocalMean":
            # nonLocalMean
            # http://docs.opencv.org/2.4/modules/photo/doc/denoising.html#fastnlmeansdenoisingcolored
            self.coin_brg = cv2.fastNlMeansDenoisingColored(self.coin_brg, templateWindowSize= 7, searchWindowSize=21)
            # Non local mean
            self.coin_gray = denoise_nl_means(self.coin_brg, 7, 9, 0.08, multichannel = True)

    def hue_enhance(self):
        """
        Convert dark or light pixels to black
        """
        # Bilateral filtering
        # The kernel size. This is the neighborhood where the local variance will be calculated,
        # and where pixels will contribute (in a weighted manner).
        d = int(10.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        # Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel
        # neighborhood (see sigmaSpace ) will be mixed together,  resulting in larger areas of semi-equal color
        sigma_color = int(80.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will
        # influence each other as long as their colors are close enough (see sigma_color ). When d>0 ,
        # it specifies the neighborhood size regardless of sigmaSpace . Otherwise,  d is proportional to sigmaSpace .
        sigmaSpace = int(80.0*np.min(self.stripe.shape[0], self.stripe.shape[1])/500.0)
        self.stripe = cv2.bilateralFilter(self.stripe.astype(np.uint8), d,  sigma_color,  sigmaSpace)

        # Equalize Hue part
        __h,  __s,  __v = self.rgb2hsv(self.stripe[:, :, 0].astype('double')/255.0,
                                       self.stripe[:, :, 1].astype('double')/255.0,
                                       self.stripe[:, :, 2].astype('double')/255.0)
        __h[__h > 0.1] = np.median(__h[__h>0.1])
        __r,  __g,  __b = self.hsv2rgb(__h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        self.stripe = (np.concatenate((__r[..., np.newaxis], __g[..., np.newaxis], __b[..., np.newaxis]),
                                      axis=len(np.shape(__r)))).astype(np.uint8)

        # Threshold
        __h,  __s,  __v = self.rgb2hsv(self.stripe[:, :, 0].astype('double')/255.0,
                                       self.stripe[:, :, 1].astype('double')/255.0,
                                       self.stripe[:, :, 2].astype('double')/255.0)
        __ms = np.median(1.0*__s[np.isfinite(__s)])
        __mv = np.median(1.0*__v[np.isfinite(__v)])

        # Equalize normal points
        seuil = 0.05
        ind_cont = np.logical_and(np.logical_and((__ms-seuil) < __s, __s < (__ms+seuil)),
                                  np.logical_and((__mv-seuil) < __v, __v < (__mv+seuil)))
        __s[ind_cont] = __ms
        __v[ind_cont] = __mv
        
        __r,  __g,  __b = self.hsv2rgb(__h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        self.stripe = (np.concatenate((__r[..., np.newaxis], __g[..., np.newaxis], __b[..., np.newaxis]),
                                      axis=len(np.shape(__r)))).astype(np.uint8)
        # Enhance Brightness
        self.stripe = np.asarray(ImageEnhance.Brightness(Image.fromarray(self.stripe.astype(np.uint8))).enhance(1.1))

    """
    STRIPE EXTRACTION
    """
    def coin2stripe(self):
        """
        Transformation the coin from a polar (circle) from to cartesian frame (stripe)
        """

        # Init stripe
        self.stripe = np.zeros((int(self.radius), int(2*np.pi*self.radius), 3))
        # Mapping for new coordinates (stripe)
        x_lin,  y_lin = np.meshgrid(range(0,  int(self.radius), 1), range(0,  int(2*np.pi*self.radius), 1))

        x_rad = np.minimum((self.im_BRG.shape[0]-1)*np.ones(x_lin.shape),
                           np.maximum(0*np.ones(x_lin.shape),
                           self.xc+(self.radius-x_lin)*np.sin(np.pi/2+y_lin/self.radius)))
        y_rad = np.minimum((self.im_BRG.shape[1]-1)*np.ones(y_lin.shape),
                           np.maximum(0*np.ones(y_lin.shape),
                           self.yc+(self.radius-x_lin)*np.cos(np.pi/2+y_lin/self.radius)))
        # Transposition        
        x_lin = x_lin.T
        y_lin = y_lin.T
        x_rad = x_rad.T
        y_rad = y_rad.T

        # Polar coordinate (circle)
        xrad_r = np.ravel(x_rad,  order='C')
        yrad_r = np.ravel(y_rad,  order='C')

        # For each color
        for color in 0,  1,  2:
            """
            Transformation to linear indices
            """
            index_stripe = np.ravel_multi_index(np.concatenate(([np.ravel(x_lin,  order='C')],
                                                                [np.ravel(y_lin,  order='C')],
                                                                [np.ravel(y_lin,  order='C')*0+color])),
                                                dims=self.stripe.shape,  order='C')
            # Linear interpolation around the index_stripe
            index_nw = np.ravel_multi_index(np.concatenate(([np.ceil(xrad_r).astype(np.int32)],
                                                            [np.ceil(yrad_r).astype(np.int32)],
                                                            [(yrad_r*0+color).astype(np.int32)])),
                                            dims=self.im_BRG.shape,  order='C')
            index_ne = np.ravel_multi_index(np.concatenate(([np.ceil(xrad_r).astype(np.int32)],
                                                            [np.floor(yrad_r).astype(np.int32)],
                                                            [(yrad_r*0+color).astype(np.int32)])),
                                            dims=self.im_BRG.shape,  order='C')
            index_sw = np.ravel_multi_index(np.concatenate(([np.floor(xrad_r).astype(np.int32)],
                                                            [np.ceil(yrad_r).astype(np.int32)],
                                                            [(yrad_r*0+color).astype(np.int32)])),
                                            dims=self.im_BRG.shape,  order='C')
            index_se = np.ravel_multi_index(np.concatenate(([np.floor(xrad_r).astype(np.int32)],
                                                            [np.floor(yrad_r).astype(np.int32)],
                                                            [(yrad_r*0+color).astype(np.int32)])),
                                            dims=self.im_BRG.shape,  order='C')
           
            # Frame transform
            pixel_nw = (xrad_r-np.floor(xrad_r))*(yrad_r-np.floor(yrad_r))*np.ravel(self.im_BRG,  order='C')[index_nw]
            pixel_ne = (xrad_r-np.floor(xrad_r))*(np.ceil(yrad_r)-yrad_r)*np.ravel(self.im_BRG,  order='C')[index_ne]
            pixel_sw = (np.ceil(xrad_r)-xrad_r) * (yrad_r-np.floor(yrad_r))*np.ravel(self.im_BRG,  order='C')[index_sw]
            pixel_se = (np.ceil(xrad_r)-xrad_r) * (np.ceil(yrad_r)-yrad_r)*np.ravel(self.im_BRG,  order='C')[index_se]

            # Linear interpolation
            np.ravel(self.stripe,  order='C')[index_stripe] = pixel_nw+pixel_ne+pixel_sw+pixel_se

    def show_stripe(self):
        Image.fromarray(self.stripe.astype(np.uint8)[:, :, [2,  1,  0]]).show()

    def save_stripe(self,  pathfile,  mode):
        """
        Save stripe for OCR recognition
        """
        if mode == 1:
            # Get band all around the circle circumference (for full legend)
            if self.side_type=="avers":
                result = Image.fromarray(np.concatenate(
                            (self.stripe[int(self.stripe.shape[0]*0.5/10):int(self.stripe.shape[0]*3.5/10):1,
                                         int(self.stripe.shape[1]*1.0/10):int(self.stripe.shape[1]*4.0/10):1,
                                         :].astype(np.uint8),
                             self.stripe[int(self.stripe.shape[0]*0.5/10):int(self.stripe.shape[0]*3.5/10):1,
                                         int(self.stripe.shape[1]*6.0/10):int(self.stripe.shape[1]*9.0/10):1,
                                         :].astype(np.uint8))
                            , axis=1))
            elif self.side_type == "revers":
                result = Image.fromarray(self.stripe[
                                         int(self.stripe.shape[0]*0.5/10):int(self.stripe.shape[0]*3.5/10):1,
                                         int(self.stripe.shape[1]*2.0/10):int(self.stripe.shape[1]*8.0/10):1,
                                         :].astype(np.uint8))
                                                 
            result.save(pathfile,  'jpeg')
            
        elif mode == 2:
            # Get band on the lower part of the coin (year)
            result = Image.fromarray(np.concatenate(
                                    (self.stripe[int(self.stripe.shape[0]/3):0:-1,
                                                 int(self.stripe.shape[1]*1.0/10):int(self.stripe.shape[1]*0/10):-1,
                                                 :].astype(np.uint8),
                                     self.stripe[int(self.stripe.shape[0]/3):0:-1,
                                                 int(self.stripe.shape[1]*10/10):int(self.stripe.shape[1]*9.0/10):-1,
                                                 :].astype(np.uint8)), axis=1)
                                    )
            result.save(pathfile,  'jpeg')
            
        elif mode == 3:
            # Get band on the lower right (workshop letter)
            result = Image.fromarray(np.concatenate((
                        self.stripe[int(self.stripe.shape[0]/3):0:-1,
                                    int(self.stripe.shape[1]*0.2/10):int(self.stripe.shape[1]*0/10):-1,
                                    :].astype(np.uint8),
                        self.stripe[int(self.stripe.shape[0]/3):0:-1,
                                    int(self.stripe.shape[1]*10/10):int(self.stripe.shape[1]*7.8/10):-1,
                                    :].astype(np.uint8)), axis=1))
            result.save(pathfile, 'jpeg')

        elif mode == 4:
            # Get band all around the circle circumference minus the part where the head overlaps (for full legend)
            result = Image.fromarray(np.concatenate(
                        (self.stripe[int(self.stripe.shape[0]*0.5/10):int(self.stripe.shape[0]*3.5/10):1,
                                     int(self.stripe.shape[1]*2.0/10):int(self.stripe.shape[1]*5.0/10):1,
                                     :].astype(np.uint8),
                         self.stripe[int(self.stripe.shape[0]*0.5/10):int(self.stripe.shape[0]*3.5/10):1,
                                     int(self.stripe.shape[1]*5.0/10):int(self.stripe.shape[1]*8.0/10):1,
                                     :].astype(np.uint8)), axis=1))
        return result

    def save_coin(self,  pathfile,  mode):
        """
        Save extract from coin for OCR recognition
        """
        if mode == 1:
            # front
            result = Image.fromarray(self.im_BRG[int(self.im_BRG.shape[0]*1/6.0):int(self.im_BRG.shape[0]*3.1/6.0):1,
                                                 int(self.im_BRG.shape[1]*1.5/6.0):int(self.im_BRG.shape[1]*4.5/6.0):1
                                                 , :].astype(np.uint8))
            result.save(pathfile, 'jpeg')
        elif mode == 2:
            # back
            result = Image.fromarray(self.im_BRG[int(self.im_BRG.shape[0]*3.2/6.0):int(self.im_BRG.shape[0]*4.4/6.0):1,
                                                 int(self.im_BRG.shape[1]*1.7/6.0):int(self.im_BRG.shape[1]*4.3/6.0):1
                                                 , :].astype(np.uint8))
            result.save(pathfile, 'jpeg')
        return result

    """
    WORD EXTRACTION
    """
    def extract_words(self):
        """
        Extracting block of word
        """

        # 0. Preprocessing : denoising and  et renforcement des contours
        im = rgb2gray(self.stripe.copy())

        # 1. Thresholding image to binary image (Otsu thresholding will work well)
        # Seuillage Otsu
        _,  im_thr = cv2.threshold((im*255.0/np.max(im)).astype(np.uint8), 0,  255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Niblack Thresholding
        print(2*np.floor(np.shape(im)[0]*0.15/2) + 1)
        
        thresh_niblack = threshold_niblack(im,  window_size=int(2*np.floor(np.shape(im)[0]*0.15/2) + 1), k=0.8)
        im_thr = 255*((im < thresh_niblack).astype(np.uint8))
        
        # 2. Add some Dilation,  Erosion to remove noise
        # Horizontal dilatation
        #im_thr = cv2.morphologyEx(im_thr,  cv2.MORPH_CLOSE,  np.ones((1,  5),np.uint8))

        # Remove artifacts connected to image border
        #im_thr = clear_border(im_thr)
        Image.fromarray(im_thr).show()
        
        # 3. Find contour from entire image
        # Selection des blocs dont les dimensions correspondent a une lettre        
        crit_shape = [x * self.radius for x in self.crit_shape]        
        
        # OpenCV Solution
        # Contour extraction
        contours,  _ = cv2.findContours(im_thr,  mode=cv2.RETR_LIST,  method=cv2.CHAIN_APPROX_SIMPLE)
        # Selection des zones de contours correspondant a une lettre
        for ncnt,  cnt in enumerate(contours):
            # bounding box: x,  y,  width,  height
            # xB,  yB,  widthB,  heightB = cv2.boundingRect(contours)
            if ncnt == 0:
                bounding_box = cv2.boundingRect(cnt)
            else:
                bounding_box = np.vstack((bounding_box,  cv2.boundingRect(cnt)))
            # Extent is the ratio of contour area to bounding rectangle area.
            # extent = float(cv2.contourArea(cnt))/(widthB*heightB)
        
        # Skimage Solution
        #for region in regionprops(label_image):
            ## take regions with large enough areas
            #if region.area >= 100:
                ## draw rectangle around segmented coins
                #minr,  minc,  maxr,  maxc = region.bbox
                #rect = mpatches.Rectangle((minc,  minr), maxc - minc,  maxr - minr,
                                          #fill = False,  edgecolor='red', linewidth = 2)
                
        
        # Selection des boites sur des critère de forme
        indSel1 =  np.logical_and(np.logical_and(np.logical_and(crit_shape[0] < bounding_box[:, 1],
                                                                bounding_box[:, 1] < crit_shape[1]), # bornes pour position verticale en partant du haut
                                                 np.logical_and(crit_shape[2] < bounding_box[:, 2],
                                                                bounding_box[:, 2] < crit_shape[3])), # bornes pour  taille horizontale
                                                 np.logical_and(crit_shape[4] < bounding_box[:, 3],
                                                                bounding_box[:, 3] < crit_shape[5])).nonzero() # bornes pour taille verticale
        contoursSel    = [contours  [i] for i in indSel1[0]]
        boundingBoxSel = bounding_box[indSel1[0],:]
        # Selection des rectangles dont les marges autour de la mediane des rectangles
        indSel2 = np.logical_and(abs(boundingBoxSel[:, 1]-np.median(boundingBoxSel[:, 1])) < 10,
                                 abs(boundingBoxSel[:, 1]+boundingBoxSel[:, 3] -
                                     np.median(boundingBoxSel[:, 1]+boundingBoxSel[:, 3])) < 10).nonzero()
        contoursSel    = [contoursSel  [i] for i in indSel2[0]]   
        boundingBoxSel = boundingBoxSel[indSel2[0], :]

        # Estimation des marges verticales        
        self.marge_hor[0] = max(0,  np.median(boundingBoxSel[:, 1])-self.radius*0.03)
        self.marge_hor[1] = np.median(boundingBoxSel[:, 1]+boundingBoxSel[:, 3])+self.radius*0.03

        Image.fromarray(im[self.marge_hor[0]:self.marge_hor[1]:1, :]).show()

        # 4. Add some are filter to select the specific font-characters
        im_edge = rgb2gray(im)
        #im_edge = cv2.cvtColor(im,  cv2.COLOR_BGR2GRAY)
        
        # Fermeture horizontale
        kernel = np.ones((1,  5),np.uint8) # 1,  5 2,  2
        im_edge = cv2.morphologyEx(im_edge,  cv2.MORPH_CLOSE,  kernel)
        kernel = np.ones((1,  1),np.uint8) # 1,  5 2,  2
        im_edge = cv2.morphologyEx(im_edge,  cv2.MORPH_OPEN,  kernel)
        # Fermeture des contours
        im_edge = cv2.morphologyEx(im_edge[self.marge_hor[0]:self.marge_hor[1]:1, :], cv2.MORPH_CLOSE,  
                                   np.ones((5,  5), np.uint8))

        # Projection sur l'axe des abcsisses
        self.histo = np.sum(im_edge[self.marge_hor[0]:self.marge_hor[1]:1, :], axis=0)
        # Seuillage
        _,  self.histo_thr = cv2.threshold((self.histo*255.0/np.max(self.histo)).astype(np.uint8), 0,  255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Dilatation a l'horizontalel
        self.histo_thr = cv2.morphologyEx(self.histo_thr,  cv2.MORPH_CLOSE,  np.ones(int(self.radius*0.2), np.uint8))
        self.histo_thr = [k[0] for k in self.histo_thr]
        # Extraction des mots detectes
        frontmontant = ((np.concatenate((self.histo_thr,  np.zeros(1)),axis=0) -
                         np.concatenate((np.zeros(1), self.histo_thr), axis=0)) > 0).nonzero()
        frontmontant = [max(0,  k-1) for k in frontmontant[0]]
        frontdescend = ((np.concatenate((self.histo_thr,  np.zeros(1)),axis=0) -
                         np.concatenate((np.zeros(1), self.histo_thr), axis=0)) < 0).nonzero()
        frontdescend = [max(0,  k-0) for k in frontdescend[0]]

        self.marge_vert = np.array([frontmontant,  frontdescend])
        self.marge_vert = self.marge_vert[:, self.marge_vert[1, :]-self.marge_vert[0, :]>100]
        
        # Compiling results from vertical and horizontal
        self.boundingBox = np.array([[self.marge_hor[0],
                                      self.marge_vert[0, k],
                                      self.marge_hor[1],
                                      self.marge_vert[1, k]]
                                     for k in range(0,  np.shape(self.marge_vert)[1])]).astype('int32')

    def get_words(self):
        """
        show words (images) extracted from extract_words
        """
        words = []
        for k_n,  k in enumerate(self.boundingBox):
            words.append(self.stripe[k[0]:k[2]:1,  k[1]:k[3]:1])
        
        return words

    def show_words(self):
        """
        show words extracted from extract_words
        """
        for k in self.boundingBox:
            cv2.imshow('Detected Avers and Revers', self.stripe[k[0]:k[2]:1,  k[1]:k[3]:1].astype(np.uint8))
            cv2.waitKey(0)

    def save_words(self,  pathfile, mode):
        """
        save image with extracted words for read_words
        """
        # Find words
        self.extract_words()
        
        # Create empty image
        spacing = 40
        total_height = 0
        max_width = 0
        for k in self.boundingBox:
            total_height += spacing+(k[3]-k[1]+1)
            max_width = max(max_width,  spacing+(k[2]-k[0]+1))
        
        new_im = Image.new('BRG', (total_height,  max_width), "white")
        
        # Agregate words
        x_offset = int(spacing/2.0)
        for k in self.boundingBox:
            new_im.paste(Image.fromarray(self.stripe[k[0]:k[2]:1,  k[1]:k[3]:1, :].astype(np.uint8)),
                         (x_offset,  int(spacing/2)))
            x_offset += k[3]-k[1]+spacing
        
        # Save the final gib image
        new_im.save(pathfile, 'jpeg')

        return new_im

    """
    FEATURE MATCHING
    """
    def register_multireference(self, pathTemplates, matchType):
        """
        Template matching results w.r.t. reference images
        """
        # Template/image matching error
        error = []
        for _path in pathTemplates: # liste des template a essayer
            # Initiate template data
            tmplt = Template.Template(_path)
            if tmplt.side_type == self.side_type:
                # Get template image
                tmpltGray = cv2.cvtColor(tmplt.image,  cv2.COLOR_BGR2GRAY)
                # Append registration error
                error.append(self.match_features(self,  tmpltGray,  matchType),
                             matchType,  tmplt.relative_coord,  tmplt.nradius)
            else:
                # if this template does not belong to this side
                error.append(0.0)


    def match_features(self, ref_img, matchType, ref_relative_coord, ref_radius):
        """
        Template detection for specific regions of interest
        """
        # Margin aroung image in percentage
        crop_margin = 0.05
        # Query (this instance) image is cropped in the ROI defined in ref_relative_coord
        img = cv2.cvtColor(self.im_BRG.copy(), cv2.COLOR_BGR2GRAY)[
                    int(self.xc+self.radius*ref_relative_coord[0]-self.radius*crop_margin):
                    int(self.xc+self.radius*ref_relative_coord[2]+self.radius*crop_margin):1,
                    int(self.yc+self.radius*ref_relative_coord[1]-self.radius*crop_margin):
                    int(self.yc+self.radius*ref_relative_coord[3]+self.radius*crop_margin):1]

        # Features matching
        if matchType == 'ORB':
            try:                
                # Compute descriptors with skimage            
                descriptor_extractor = ORB(n_keypoints=self.Ndescr)
                
                descriptor_extractor.detect_and_extract(img)
                img_kpt = descriptor_extractor.keypoints
                img_dsc = descriptor_extractor.descriptors
                
                descriptor_extractor.detect_and_extract(ref_img)
                ref_kpt = descriptor_extractor.keypoints
                ref_dsc = descriptor_extractor.descriptors
            except:
                # catch *all* exceptions
                img_dsc = [0]
                ref_dsc = [0]
            
            # Match descriptors
            if len(img_dsc) > 5 and len(ref_dsc) > 5:
                matches = match_descriptors(img_dsc,  ref_dsc,  cross_check=True,  max_distance=self.radius*0.1)
                
                """
                # Lowe's test ratio
                from scipy.spatial.distance import cdist
                distances = np.zeros((matches.shape[0]))
                for _k in range(0,  matches.shape[0]):
                    distances[_k] = cdist(ref_dsc[(_k):(_k+1):1, ], img_dsc[(_k):(_k+1):1, ], metric='hamming', p = 2)
                """
                
                img_kpt = img_kpt[matches[:, 0], ]
                img_dsc = img_dsc[matches[:, 0], ]
                            
                ref_kpt = ref_kpt[matches[:, 1], ]
                ref_dsc = ref_dsc[matches[:, 1], ]
                            
                img_pts = np.float32(img_kpt).reshape(-1,  1,  2)
                ref_pts = np.float32(ref_kpt).reshape(-1,  1,  2)
            
                # Find robust homography
                tform,  mask = cv2.findHomography(img_pts,  ref_pts,  cv2.RANSAC,  5.0)
                # Extract the rotation sclaing part and get principal components
                tform_w,  tform_v = np.linalg.eig(tform[0:2:1,  0:2:1])
                
                # Ratio of mismatches
                tform_matchesratio = 1/float(sum(mask))
                # Aspect ratio for the homography
                tform_ratio = max(abs(tform_w))/min(abs(tform_w))-1.0
                # Homography angle in degree
                tform_angle = abs(np.arccos(max(tform_v[0,  0], -tform_v[1,  1]))*180.0/np.pi)
                error = [tform_matchesratio,  tform_ratio,  tform_angle]
    
                # Affichage
                h,  w = ref_img.shape
                pts = np.float32([[0,  0], [0,  h-1], [w-1,  h-1], [w-1,  0]]).reshape(-1,  1,  2)
                dst = cv2.perspectiveTransform(pts,  tform)
                titre = 'tform_ratio=' + str(tform_ratio) + \
                        '; tform_matchesratio = ' + str(int(1/tform_matchesratio)) + \
                        '; tform_angle = ' + str(tform_angle)
                imgHomog = cv2.polylines(img, [np.int32(dst)], True,  255,  3,  cv2.LINE_AA)
                M1 = np.zeros((max(imgHomog.shape[0], ref_img.shape[0]), imgHomog.shape[1]+10+ref_img.shape[1])).astype(np.uint8)
                M1[0:imgHomog.shape[0]:1,  0                  :imgHomog.shape[1]                :1] = imgHomog
                M1[0:ref_img.shape[0]    :1,  imgHomog.shape[1]+10:imgHomog.shape[1]+10+ref_img.shape[1]:1] = ref_img
            else:
                error = [10^4,  10^4,  10^4]
                M1 = 0
                titre  = "No matches"
            
        elif matchType == 'Correlation1' or matchType == 'Correlation2':
        
            # Template and query image transformed to has same scale
            scale = [float(self.radius)/float(ref_radius)*x for x in np.arange(0.95,  1.05,  0.02)]
            errorMatching = []
            for s_n,  s in enumerate(scale):  # for several scales we compare the two images
                # Scaled template
                ref_imgscaled = cv2.resize(ref_img,  dsize=(0,  0), fx=s,  fy=s ,interpolation = cv2.INTER_LINEAR)
                # Pick the right correlation method (utput is the same size as the image)
                try:
                    if matchType == 'Correlation1':
                        res = cv2.matchTemplate( img,  ref_imgscaled,  cv2.TM_CCORR_NORMED) # cv2.TM_CCOEFF_NORMED cv2.TM_CCORR_NORMED cv2.TM_SQDIFF_NORMED
                    elif matchType == 'Correlation2':
                        res = match_template(img,  ref_imgscaled,  pad_input=False)
                except:
                    # catch *all* exceptions
                    res = 0.0
                    e = sys.exc_info()[0]
                    print( "Error: %s" % e)
                    
                # Update error depending on error at previous scales
                # Error is defined as opposite of correlation
                errorMatching.append(1-np.max(res)) 
                if s_n == 0:
                    resmin = 1-res
                    s_nmin = 0
                else:
                    # current scale fits better,  then update the best restult
                    if errorMatching[s_n] < np.min(resmin):
                        resmin = 1-res
                        s_nmin = s_n
                      
            error = errorMatching[s_nmin]
            # Show the results            
            min_val,  max_val,  min_loc,  max_loc = cv2.minMaxLoc(resmin)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED,  take minimum
            top_left = min_loc
            w,  h = np.shape(cv2.resize(ref_img,  dsize=(0,  0), fx=scale[s_nmin], fy=scale[s_nmin],
                                        interpolation=cv2.INTER_AREA))
            bottom_right = (top_left[0] + h,  top_left[1] + w)
            cv2.rectangle(img,  top_left,  bottom_right,  255,  2)
            titre = 'Best error is '+str(errorMatching[s_nmin])+\
                    ' and best scale is '+str(scale[s_nmin]/(float(self.radius)/float(ref_radius)))
            M1 = np.zeros((max(ref_img.shape[0], img.shape[0]), ref_img.shape[1]+10+img.shape[1])).astype(np.uint8)
            M1[0:ref_img.shape[0]:1,  0                  :ref_img.shape[1]                :1] = ref_img
            M1[0:img.shape[0]    :1,  ref_img.shape[1]+10:ref_img.shape[1]+10+img.shape[1]:1] = img
                                
        else: # if this template does not belong to this side
            error = 10^4
            M1 = 0
            titre = "None"
                    
        # Output
        res = [error,  M1,  titre]
        return res

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
        
        s = (maxc-minc) / (maxc+0.00000000000001)
        
        rc = (maxc-r) / (maxc-minc+0.00000000000001)
        gc = (maxc-g) / (maxc-minc+0.00000000000001)
        bc = (maxc-b) / (maxc-minc+0.00000000000001)
        
        v = maxc
        
        h = (minc == maxc)*np.zeros(np.shape(r)) + \
            (minc != maxc)*((r == maxc)*(bc-gc) +
                            (r != maxc)*(g == maxc)*(2.0+rc-bc) +
                            (r != maxc)*(g != maxc)*(b == maxc)*(4.0+gc-rc))
        h = (h/6.0) % 1.0   
        
        h = np.asarray(h)
        s = np.asarray(s)
        v = np.asarray(v)
        
        return h,  s,  v

    @staticmethod
    def hsv2rgb(h,  s,  v):
        i = (h*6.0).astype(np.uint8)
        f = (h*6.0) - i
        p = v*(1.0 - s)
        q = v*(1.0 - s*f)
        t = v*(1.0 - s*(1.0-f))
        i = i % 6.0
        
        r = (s == 0.0)*v+(s != 0.0)*((i == 0)*v+(i == 1)*q+(i == 2)*p+(i == 3)*p+(i == 4)*t+(i == 5)*v)
        g = (s == 0.0)*v+(s != 0.0)*((i == 0)*t+(i == 1)*v+(i == 2)*v+(i == 3)*q+(i == 4)*p+(i == 5)*p)
        b = (s == 0.0)*v+(s != 0.0)*((i == 0)*p+(i == 1)*p+(i == 2)*t+(i == 3)*v+(i == 4)*v+(i == 5)*q)
                
        return r,  g,  b

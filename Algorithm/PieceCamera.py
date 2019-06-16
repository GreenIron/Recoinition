# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:51:36 2016

@author: victor
 """

from PIL import Image,  ImageOps
import cv2
import numpy as np
from skimage.morphology import disk,  reconstruction
from skimage import color
from skimage.transform import downscale_local_mean
from skimage.filters import gaussian
from skimage.color import rgb2grey

from Algorithm import Preprocessing


class PieceCamera:
    def __init__(self):
        self.avers = Ravers()
        self.revers = Ravers()
        
        self.avers_BGR = None
        self.revers_BGR = None
        self.numismatic_image = None
        
        # Height of extracted image
        self.height = 1000.0

    """
    IMAGE DISPLAY
    """
    def show_coin_with_circles(self):
        """
        Show avers and revers in the picture
        """
        if not self.avers.im_BRG is None:
            coin_with_circle = self.avers.im_BRG
            cv2.circle(coin_with_circle, (int(self.avers.xc), int(self.avers.yc)),
                       int(self.avers.radius), (0,  255,  0), 2)
            cv2.circle(coin_with_circle, (int(self.avers.xc), int(self.avers.yc)), 2, (0,  0,  255), 3)
            Image.fromarray(coin_with_circle).show()
        
        if not self.revers.im_BRG is None:             
            coin_with_circle = self.revers.im_BRG
            cv2.circle(coin_with_circle, (int(self.revers.xc), int(self.revers.yc)),
                       int(self.revers.radius), (0,  255,  0), 2)
            cv2.circle(coin_with_circle, (int(self.revers.xc), int(self.revers.yc)), 2, (0,  0,  255), 3)
            Image.fromarray(coin_with_circle).show()
            
    def rmvbckgnd_localmax(self,  img):
        # Get the HSV image    
        marge = np.int((0.05*min(img.shape[0], img.shape[1])))
        kernelsize = np.int((0.2*min(img.shape[0], img.shape[1])))
        __hsv = color.rgb2hsv(img)
        
        # Unknown threshold : use otsu threshoding on S and V
        otsu_threshold_s,  img_thred_s = cv2.threshold(np.asarray(256*__hsv[:, :, 1]).astype(np.uint8),
                                                        0,  1024,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        otsu_threshold_v,  img_thred_v = cv2.threshold(np.asarray(256*__hsv[:, :, 2]).astype(np.uint8),
                                                       0,  1024,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_thred = img_thred_v
        
        # Rate of white at the center of the image
        img_thred_center = img_thred[np.int(0.45*img_thred.shape[0]):np.int(0.55*img_thred.shape[0]):1,
                                     np.int(0.45*img_thred.shape[1]):np.int(0.55*img_thred.shape[1]):1]
        rateofwhite = np.sum(np.sum(img_thred_center,  axis=0), axis=0)\
                             / 255.0/(img_thred_center.shape[0]*img_thred_center.shape[1])
        
        # Rate of black on the edges of the image
        img_thred_center_1 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = np.sum(np.sum(img_thred_center_1,  axis=0), axis=0)
        den = img_thred_center_1.shape[0]*img_thred_center_1.shape[1]
        
        img_thred_center_2 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_2,  axis=0), axis=0)
        den = den+ img_thred_center_2.shape[0]*img_thred_center_2.shape[1]
        
        img_thred_center_3 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_3,  axis=0), axis=0)
        den = den + img_thred_center_3.shape[0]*img_thred_center_3.shape[1]
        
        img_thred_center_4 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_4,  axis=0), axis=0)
        den = den + img_thred_center_4.shape[0]*img_thred_center_4.shape[1]
        
        rateofblack = 1.0 - num/den/255
        
        # Init for morphological reconstruction
        seed = np.copy(img)
        mask = img
        
        if rateofblack > 0.5:
            seed[1:-1,  1:-1] = img.min()
            reconstructed = reconstruction(seed,  mask,  method='dilation')
        else:
            seed[1:-1,  1:-1] = img.max()
            reconstructed = reconstruction(seed,  mask,  method='erosion')
            
        reconstructed = (img-reconstructed).astype(np.uint8)
        return reconstructed

    def extract_coinHough(self,  cam_im,  ravers):
        """
        EXTRACT COIN
        """
        # STEP 0: INITIALIZATION AND PARAMETERS
        # Image init
        cam_imasarray = np.asarray(cam_im)
        cam_imasarray.setflags(write=1)
        circle_coordinates = None
        size_gaus = int(max(15,  2.0*round(int(np.shape(cam_im)[0]*0.03)/2.0)+1.0))

        # Image is resized to achieve 200 pixels each side
        scalefactors = max(1,  int(np.shape(cam_imasarray)[0]/200.0))

        # STEP 1: DETOURAGE PAR COULEUR. Detourage autour de la pièce
        img_filtered = cv2.medianBlur(cam_imasarray,  size_gaus)

        # Contour extraction
        has_cverge = False
        # number of allowed iterations to find valid boundaries
        max_iter = 5
        cnt_ter = 0
        while (not has_cverge) and (cnt_ter < max_iter):
            cnt_ter = cnt_ter + 1
            #  Find boundaried around the coin and croponce
            __,  bornes0EQ,  bornes1EQ,  has_cverge = self.rmvbvckgnd_otsu(img_filtered,  kernelscale = 0.05)
            img_filtered  = img_filtered [bornes1EQ[0]:bornes1EQ[1]:1,  bornes0EQ[0]:bornes0EQ[1]:1, :]
            cam_imasarray = cam_imasarray[bornes1EQ[0]:bornes1EQ[1]:1,  bornes0EQ[0]:bornes0EQ[1]:1, :]

        # Same with different blurring
        img_filtered = cv2.medianBlur(cam_imasarray,  2*int(size_gaus/4)+1)
        has_cverge = False
        cnt_ter = 0
        while not(has_cverge) and (cnt_ter<max_iter):
            cnt_ter = cnt_ter + 1
            #  Find boundaries around the coin and crop
            __,  bornes0EQ,  bornes1EQ,  has_cverge = self.rmvbvckgnd_otsu(img_filtered,  kernelscale = 0.05)
            img_filtered  = img_filtered [bornes1EQ[0]:bornes1EQ[1]:1,  bornes0EQ[0]:bornes0EQ[1]:1, :]
            cam_imasarray = cam_imasarray[bornes1EQ[0]:bornes1EQ[1]:1,  bornes0EQ[0]:bornes0EQ[1]:1, :]

        # STEP 2: HOUGH PROCESSING. Fitting the coin inside a circle.
        # Preprocessing to improve Hough performance: resizing,  median blurring
        downscale_local_mean(cam_imasarray,  (scalefactors,  scalefactors,  scalefactors))
        img_filtered = gaussian(rgb2grey(cam_imasarray), size_gaus)
        # Preprocessing : contrast improvement
        img_filtered = Preprocessing.Preprocessing.contrast_stretching(img_filtered)
        otsu_threshold,  _ = cv2.threshold(img_filtered,
                                           np.min(np.min(img_filtered)),
                                           np.max(np.max(img_filtered)),
                                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Parameters for Hough
        __l = min(img_filtered.shape[0], img_filtered.shape[1])
        # Margins to extract revers and avers
        marge_ravers = np.int(0.05*__l)
        # Minimum distance between the centers of the detected circles
        min_dist_center = max(0.5*__l,  1)
        # Margin range in % to look for the right diameter
        range_margin = 0.00
        param2scale = 1.0
        size_gaus = 2*np.int(0.05*__l/2.0)+1

        try:
            # Try to find circles with increasing radius range
            while circle_coordinates is None:

                min_radius = max(20,  int((0.95-range_margin)/4.0*float(__l)))
                max_radius = int(0.55*float(__l))

                # Hough Transform to detect coins. This method is based on openCV rather than sci-image,
                # since openCV demonstrate superior circle detection performances
                circle_coordinates = cv2.HoughCircles(img_filtered,  cv2.HOUGH_GRADIENT,
                                                      dp=1, minDist=min_dist_center,
                                                      param1=int(param2scale*otsu_threshold),
                                                      param2=int(param2scale*otsu_threshold/2.0),
                                                      minRadius=min_radius,  maxRadius=max_radius)

                # Get circles only within the original image frame. We allow extra margin
                if circle_coordinates is not None:
                    index_good_circles = np.logical_and(
                        np.logical_and(1 < (circle_coordinates[0, :, 0]-circle_coordinates[0, :, 2]),
                                       (circle_coordinates[0, :, 0]+circle_coordinates[0, :, 2]) < img_filtered.shape[1]+1),
                        np.logical_and(1 < (circle_coordinates[0, :, 1]-circle_coordinates[0, :, 2]),
                                       (circle_coordinates[0, :, 1]+circle_coordinates[0, :, 2]) < img_filtered.shape[0]+1)
                                                      )
                    circle_coordinates = circle_coordinates[:, index_good_circles, :]

                    # If list is empty,  then no circles were found
                    if not(circle_coordinates.any()):
                        circle_coordinates = None

                # Expanding the radius range
                range_margin = range_margin+0.01
                min_dist_center = max(min_dist_center/1.0,  5.0)
                param2scale = param2scale/1.05

        except: # Watershed... if Hough transform failed
            pass

        # STEP 3: EXTRACTION OF THE CIRCLE AND ALLOCATION THE IMAGE.
        # After an acceptable candidate has been found,  we update the coordinates.
        # Update avers and revers attributes,  second argument
        # circle_coordinates[0, :,2].argmax(0) is the circle with largest diameter,  0 is the best circle
        circle_coordinates = np.int16(1/scalefactors
                                      * np.around(circle_coordinates[0,  circle_coordinates[0, :, 2].argmax(0), :]))

        # Circle boundaries in the original picture and store the detected image
        circle_coordinates[2] = circle_coordinates[2]*1.02
        radius = circle_coordinates[2]
        taille = 2*(marge_ravers+circle_coordinates[2])
        getattr(self,  ravers).im_BRG = 0*np.ones((taille,  taille,  3)).astype(np.uint8)

        # Copy image
        bornes = [circle_coordinates[0]-circle_coordinates[2]-marge_ravers,
                  circle_coordinates[0]+circle_coordinates[2]+marge_ravers,
                  circle_coordinates[1]-circle_coordinates[2]-marge_ravers,
                  circle_coordinates[1]+circle_coordinates[2]+marge_ravers]

        boundaries = [int(max(0,  bornes[0])), int(min(cam_imasarray.shape[1], bornes[1])),
                      int(max(0,  bornes[2])), int(min(cam_imasarray.shape[0], bornes[3]))]

        # Update image
        getattr(self,  ravers).im_BRG[
                        max(0, -0-bornes[2]):(taille-max(0,  bornes[3]-0-cam_imasarray.shape[0])):1,
                        max(0, -0-bornes[0]):(taille-max(0,  bornes[1]-0-cam_imasarray.shape[1])):1,
                        ::] = cam_imasarray[ boundaries[2]:boundaries[3]:1,
                                      boundaries[0]:boundaries[1]:1, ::]

        # Set background to white
        im_brg_blurred = cv2.medianBlur(getattr(self,  ravers).im_BRG,  size_gaus)

        # Remove black background
        getattr(self,  ravers).im_BRG = self.black2white(getattr(self,  ravers).im_BRG,  im_brg_blurred,
                                                         satL=8,  valU=15)

        # Update attributes
        getattr(self,  ravers).xc = float(marge_ravers+circle_coordinates[2])
        getattr(self,  ravers).yc = float(marge_ravers+circle_coordinates[2])
        getattr(self,  ravers).radius = float(radius)

        # Set to white whats not in the circle
        x_coords,  y_coords = np.meshgrid(range(0,  int(getattr(self,  ravers).im_BRG.shape[0]), 1),
                                          range(0,  int(getattr(self,  ravers).im_BRG.shape[0]), 1))
        # Transposition
        x_coords = x_coords.T
        y_coords = y_coords.T

        for color in 0,  1,  2:
            g_ind = (((x_coords-getattr(self,  ravers).xc)**2+(y_coords-getattr(self,  ravers).yc)**2)>radius**2)
            x_coords = x_coords[g_ind]
            y_coords = y_coords[g_ind]
            # Transformation to linear indices
            index_ = np.ravel_multi_index(np.concatenate(([np.ravel(x_coords,  order='C')],
                                                          [np.ravel(y_coords,  order='C')],
                                                          [np.ravel(y_coords,  order='C')*0+color])),
                                          dims=getattr(self,  ravers).im_BRG.shape,  order='C')
            np.ravel(getattr(self,  ravers).im_BRG,  order='C')[index_] = 255

    def generate_numismaticimage(self):
        """
        Generate an image that follows numismatic picture format
        """
        # Get avers and revers and reshape then to match width
        r_avers = np.asarray(
            Image.fromarray(self.avers.im_BRG).resize(
                (int(self.height/self.avers.im_BRG.shape[0]*self. avers.im_BRG.shape[1]), int(self.height)),
                Image.BICUBIC))
        r_revers = np.asarray(
            Image.fromarray(self.revers.im_BRG).resize(
                (int(self.height/self.revers.im_BRG.shape[0]*self.revers.im_BRG.shape[1]), int(self.height)),
                Image.BICUBIC))
        # Make the numismatic image width avers and revers side by side
        self.numismatic_image = np.concatenate((r_avers,
                                                np.asarray(Image.new("RGB", (int(self.height/10.0), int(self.height)),
                                                          "white")), r_revers), axis=1).astype(np.uint8)

    """
    COLOR FRAME CONVERSION 
    """
    @staticmethod
    def black2white(image_orig, image_blurred, satL, valU):
        """
        Thresholding : Convert black background to white
        """
        __hsv = color.rgb2hsv(image_blurred)

        ind_cont = np.logical_or(np.logical_and((satL/100.0) < __hsv[:, :, 1], __hsv[:, :, 2] < (valU/100.0)),
                                 __hsv[:, :, 2] < (8.0/100.0))

        image_orig_e = image_orig
        image_orig_e[ind_cont,  0] = 255.0
        image_orig_e[ind_cont,  1] = 255.0
        image_orig_e[ind_cont,  2] = 255.0

        return image_orig_e

    @staticmethod
    def rmvbvckgnd_otsu(img,  kernelscale):
        """
        Select elements in the inout HSV boundaries and define the image boundaries around them
        """

        # Get the HSV image
        marge = np.int(0.04*min(img.shape[0], img.shape[1]))
        kernel_size = np.int(max(20, kernelscale * min(img.shape[0], img.shape[1])))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        __hsv = color.rgb2hsv(img)

        # Unknwon threshold : use otsu threshoding on S and V
        Otsu_threshold_v,  img_thred_v = cv2.threshold(np.asarray(255*__hsv[:, :, 2]).astype(np.uint8),
                                                       0,  255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_thred = img_thred_v

        # It is assumed that the coin is located at the center of the picture
        img_thred_center = img_thred[np.int(0.45*img_thred.shape[0]):np.int(0.55*img_thred.shape[0]):1,
                                     np.int(0.45*img_thred.shape[1]):np.int(0.55*img_thred.shape[1]):1]

        # Rate of black on the edges of the image
        img_thred_center_1 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = np.sum(np.sum(img_thred_center_1,  axis=0), axis=0)
        den = img_thred_center_1.shape[0]*img_thred_center_1.shape[1]

        img_thred_center_2 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_2,  axis=0), axis=0)
        den = den + img_thred_center_2.shape[0]*img_thred_center_2.shape[1]

        img_thred_center_3 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_3,  axis=0), axis=0)
        den = den + img_thred_center_3.shape[0]*img_thred_center_3.shape[1]

        img_thred_center_4 = img_thred[np.int(0.95*img_thred.shape[0]):np.int(1.00*img_thred.shape[0]):1, ::]
        num = num + np.sum(np.sum(img_thred_center_4,  axis=0), axis=0)
        den = den + img_thred_center_4.shape[0]*img_thred_center_4.shape[1]

        rate_of_black = 1.0 - num / den / 255

        if rate_of_black > 0.5:
            ind_cont = img_thred_v < __hsv[:, :, 2]
        else:
            ind_cont = img_thred_v > __hsv[:, :, 2]

        # Closing and then opening to remove artifacts and holes
        masque = cv2.morphologyEx((255.0*ind_cont).astype(np.uint8), cv2.MORPH_CLOSE,  kernel)
        ind_cont = masque > 0
        masque = cv2.morphologyEx((255.0*ind_cont).astype(np.uint8), cv2.MORPH_OPEN,   kernel)
        ind_cont = masque > 0

        # Find boundaries (non 1) of the mask and crop around it
        # Projection on one of the axis
        indice0 = [i for (i,  val) in enumerate(np.prod(ind_cont,  axis=0)) if val == False]
        if np.size(indice0) < 1:
            indice0 = [0,  ind_cont.shape[1]]
        bornes0 = [round(max(indice0[0]-marge,  0)), round(min(indice0[-1]+marge,  ind_cont.shape[1]))]
        # Projection on one of the axis
        indice1 = [i for (i,  val) in enumerate(np.prod(ind_cont,  axis=1)) if val == False]
        if np.size(indice1) < 1:
            indice1 = [0,  ind_cont.shape[0]]
        bornes1 = [round(max(indice1[0]-marge,  0)), round(min(indice1[-1]+marge,  ind_cont.shape[0]))]
        
        # Check if boundaries moved or no
        has_cverge = (bornes0[0] < 0.05*ind_cont.shape[1]) and \
                     (0.95*ind_cont.shape[1] < bornes0[1]) and \
                     (bornes1[0] < 0.05*ind_cont.shape[0]) and \
                     (0.95*ind_cont.shape[0] < bornes1[1])
        
        return ind_cont,  bornes0,  bornes1,  has_cverge

class Ravers:
    def __init__(self):
        self.im_BRG = None
        self.xc = None
        self.yc = None
        self.radius = None

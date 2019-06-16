# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 21:29:08 2016

@author: victor
"""
import os
import re
import time
from PIL import Image, ImageOps
import cv2
import csv
import numpy as np
from xlrd import open_workbook
import unicodedata
from io import StringIO

from Algorithm import AversRevers


class Piece:
    def __init__(self,  ImagePiecePath):
        """
        Loading the image
        """
        if isinstance(ImagePiecePath,  str):
            # When it's a local file, Color image loaded by OpenCV is in BGR mode
            self.coinBGR = cv2.imread(ImagePiecePath,  cv2.IMREAD_COLOR)
            self.coinGray = cv2.imread(ImagePiecePath,  cv2.IMREAD_GRAYSCALE)
            self.picture_path = ImagePiecePath
        elif isinstance(ImagePiecePath,  Image.Image):
            # When it's passed as a PIL image
            imagelocal = ImagePiecePath
            self.coinBGR = np.asarray(imagelocal)
            self.coinGray = np.asarray(ImageOps.grayscale(imagelocal))
            self.picture_path = ImagePiecePath.name
        else:
            # When it's passed as a string
            imagelocal = Image.open(StringIO(ImagePiecePath.read()))
            self.coinBGR = np.asarray(imagelocal)
            self.coinGray = np.asarray(ImageOps.grayscale(imagelocal))
            self.picture_path = ImagePiecePath.name

        # o is for original
        self.coinBGRo = self.coinBGR.copy()
        
        # Avers et revers de la piece
        self.avers = AversRevers.AversRevers("avers")
        self.revers = AversRevers.AversRevers("revers")
        # Margin to extract the two sides (revers and avers) in # of pixels
        self.marge_ravers = 10
        self.param_extract = [["avers", 0, int(self.coinGray.shape[1]*0.6)],
                              ["revers", int(self.coinGray.shape[1]*0.4), self.coinGray.shape[1]]]

        # Numismatic identification
        # Numismatic
        self.numismatic = []
        # Type of text extraction
        self.text_extraction = 'Tesseract', 'Google', 'Tensorflow'
        # List with different ROI in the coin. Unit is a percentage of diameter wrt to the coin center.
        # Each ROI is described as follow: x_upperleft,  y_upperleft,  x_lowerdroite,  y_lowerdroite
        self.decoupageRavers = [[[-0.6, -0.6, +0.4, -0.2], # profil gauche - avers
                               [-0.6, +0.2, +0.4, +0.6], # profil droite - avers
                               [+0.6, -0.4, +0.9, +0.4]], # bas de la piece - avers
                               [[-0.7, -0.4, +0.0, +0.4], # partie haut - revers
                               [-0.0, -0.4, +0.6, +0.4], # partie moyenne basse - revers
                               [+0.4, -0.7, +0.7, -0.45], # partie bas gauche - revers
                               [+0.4, +0.45, +0.7, +0.7], # partie bas gauche - revers
                               [+0.6, -0.6, +0.8, -0.3], # partie bas droite - revers
                               [+0.6, +0.3, +0.8, +0.6],
                               [+0.7, -0.2, +0.93, +0.2]]]
        # Threshold for a Correlation and Feature for the error. Theslower,  the more discriminant
        self.threshold_corr = 0.5
        # inv of mismatches ratio,  aspect ratio(max val pro/smal val p)-1,  angle in degree. error <threshold_feature
        self.threshold_feature = 1.0/3,  0.2,  30

    """
    IMAGE DISPLAY
    """
    def show_coin_with_circles(self):
        """
        Show avers and revers in the picture
        """
        if not self.avers.im_BRG is None and not self.avers.im_BRG is None:

            coin_with_circle = self.avers.im_BRG
            cv2.circle(coin_with_circle, (int(self.avers.xc), int(self.avers.yc)), int(self.avers.radius), (0,  255,  0), 2)
            cv2.circle(coin_with_circle, (int(self.avers.xc), int(self.avers.yc)), 2, (0,  255,  0), 3)
            Image.fromarray(coin_with_circle[:, :, [2,  1,  0]]).show()
            
            coin_with_circle = self.revers.im_BRG
            cv2.circle(coin_with_circle, (int(self.revers.xc), int(self.revers.yc)), int(self.revers.radius), (0,  255,  0), 2)
            cv2.circle(coin_with_circle, (int(self.revers.xc), int(self.revers.yc)), 2, (0,  255,  0), 3)
            time.sleep(0.1)
            Image.fromarray(coin_with_circle[:, :, [2,  1,  0]]).show()

    def reset_image(self,  method_type):
        """
        Reinitialize image
        """
        self.coinBGR  = self.coinBGRo
        self.coinGray = np.dot(self.coinBGRo,  [0.299,  0.587,  0.114])

    def show_image(self):
        """
        Accessor
        """
        Image.fromarray(np.vstack([self.coinBGRo.astype(np.uint8), self.coinBGR.astype(np.uint8)])).show()

    """
    IMAGE ENHANCEMENT
    """
    # Different Types of image enhancement
    def enhance_image(self,  method_type):
        if method_type[0] == "guidedFilter":  # Edge preserving filtering : guidedFilter Kaiming He et. al., “Guided Image Filtering, ” ECCV 2010,  pp. 1 - 14.
            if len(method_type) == 3:
                self.coinBGR = cv2.guidedFilter(self.coinBGR,  method_type[1], method_type[2])
            else:
                radius = max(5,  0.3*int(len(self.coinBGR)))
                eps = 10  # eps**2 is similar to sigma_color in bilateralFilter
                self.coinBGR = cv2.guidedFilter(self.coinBGR,  radius,  eps)
        elif method_type[0] == "bilateralFilter":  # Edge preserving filtering : bilateralFilter
            # http://docs.opencv.org/3.0-beta/modules/imgproc/doc/filtering.html#void%20bilateralFilter%28InputArray%20src, %20OutputArray%20dst, %20int%20d, %20double%20sigmaColor, %20double%20sigmaSpace, %20int%20borderType%29
            # http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MANDUCHI1/Bilateral_Filtering.html            
            if len(method_type) == 4:
                self.coinBGR = cv2.guidedFilter(self.coinBGR,  method_type[1], method_type[2], method_type[3])
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
                # the pixel neighborhood (see sigma_space ) will be mixed together,  resulting in larger areas of
                # semi-equal color
                sigma_color = 50
                # Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels
                # will influence each other as long as their colors are close enough (see sigma_color ). When d>0 ,
                # it specifies the neighborhood size regardless of sigma_space . Otherwise,  d is proportional to
                # sigma_space .
                sigma_space = 0

                self.coinBGR = cv2.bilateralFilter(self.coinBGR,  d,  sigma_color,  sigma_space)
                
        elif method_type[0] == "fastNlMeansDenoisingColored": # fastNlMeansDenoisingColored
            self.coinBGR = cv2.fastNlMeansDenoisingColored(self.coinBGR,  templateWindowSize=7,  searchWindowSize=21) # http://docs.opencv.org/2.4/modules/photo/doc/denoising.html#fastnlmeansdenoisingcolored
            # Local threshold http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#adaptive-thresholding
            self.coinGray = cv2.adaptiveThreshold(self.coinGray,  255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  cv2.THRESH_BINARY,  31,  10)
            self.coinBGR = cv2.cvtColor(self.coinGray,  cv2.COLOR_GRAY2RGB)

    # Convert dark or white background to black using in the HSV domain
    def hue_enhance(self):
        self.coinBGR.flags.writeable = True
        # Equalize Hue part
        __h,  __s,  __v = self.rgb2hsv(self.coinBGR[:, :, 0].astype('double')/255.0,
                                       self.coinBGR[:, :, 1].astype('double')/255.0,
                                       self.coinBGR[:, :, 2].astype('double')/255.0)
        __h[__h > 0.1]  = np.median(__h[__h > 0.1])

        __r,  __g,  __b = self.hsv2rgb(__h,  __s,  __v)
        __r = (255.0*__r).astype(np.uint8)
        __g = (255.0*__g).astype(np.uint8)
        __b = (255.0*__b).astype(np.uint8)
        self.coinBGR = (np.concatenate((__r[..., np.newaxis], __g[..., np.newaxis], __b[..., np.newaxis]),
                                       axis=len(np.shape(__r)))).astype(np.uint8)
        # Filtering

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
        
        s  = (maxc-minc) / maxc
        
        rc = (maxc-r) / (maxc-minc+0.00000000000001)
        gc = (maxc-g) / (maxc-minc+0.00000000000001)
        bc = (maxc-b) / (maxc-minc+0.00000000000001)
        
        v = maxc
        
        h = (minc == maxc)*np.zeros(np.shape(r)) + (minc != maxc)*((r == maxc)*(bc-gc) +
            (r != maxc)*(g == maxc)*(2.0+rc-bc) + (r != maxc)*(g != maxc)*(b == maxc)*(4.0+gc-rc))
        h = (h/6.0) % 1.0   
        
        h = np.asarray(h)
        s = np.asarray(s)
        v = np.asarray(v)
        
        return h,  s,  v

    @staticmethod
    def hsv2rgb(h,  s,  v):
        """
        assumes int truncates!
        """
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
        
    """
    EXTRACT COIN CONTOUR
    """
    def generate_aversrever(self):
        """
        Creates and extract both sides (revers and avers) assuming that the image follows coin dealing standards (both
        sides on the same picture side by side)
        """
        # For both sides
        for ravers in self.param_extract:

            """
            Hough Circle parameter
            """
            circle_coordinates = None
            # Minimum distance between the centers of the detected circles
            min_dist_center = max(0.05*self.coinGray.shape[0], 1)
            # margin range in % to look for the right diameter
            range_margin = 0.00
            param2scale = 1.1
            resizedheight = 100
            sizeGaus = int(max(15, 2.0*round(int(self.coinGray.shape[0]*0.03)/2.0)+1.0))
            
            # Preprocessing : blurring
            img_resize = self.coinGray[::, ravers[1]:ravers[2]:1]
            img_resize = cv2.GaussianBlur(img_resize, (sizeGaus,  sizeGaus), 2)
            img_resize = cv2.medianBlur(img_resize,  sizeGaus)
            #img_resize = roberts( img_resize )
            img_resize = (img_resize*255.0/np.max(img_resize)).astype(np.uint8)
                        
            # Preprocessing : downsizing image
            #img_resize = downscale_local_mean(img_resize, (np.floor(np.shape(img_resize)[0]/resizedheight),np.floor(np.shape(img_resize)[0]/resizedheight)))
            Image.fromarray(img_resize).show()

            # Otsu's thresholding
            Otsu_threshold,  _ = cv2.threshold(img_resize,  0,  255,  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
            # Try to find the first acceptable circles by realeasing iterativelly the constraints
            while circle_coordinates is None:
                                
                # Hough Transform to detect coins
                min_radius = int((0.95-range_margin)/3.0*float(self.coinGray.shape[0]))
                max_radius = int((1.03 + 0) / 2.0 * float(self.coinGray.shape[0]))

                circle_coordinates = cv2.HoughCircles(img_resize, cv2.HOUGH_GRADIENT,
                                                      dp=1, minDist=min_dist_center, param1=int(Otsu_threshold),
                                                      param2=int(Otsu_threshold/param2scale),
                                                      minRadius=min_radius, maxRadius=max_radius)

                if circle_coordinates is not None:
                    # Get circles only within the original image. We allow extra margin (5 pixels)
                    index_good_circles = np.logical_and(
                        np.logical_and(1 < (circle_coordinates[0, :, 0]-circle_coordinates[0, :, 2]),
                                       (circle_coordinates[0, :, 0]+circle_coordinates[0, :, 2]) < img_resize.shape[1]+1),
                        np.logical_and(1 < (circle_coordinates[0, :, 1]-circle_coordinates[0, :, 2]),
                                       (circle_coordinates[0, :, 1]+circle_coordinates[0, :, 2]) < img_resize.shape[0]+1)
                                                      )

                    circle_coordinates = circle_coordinates[:, index_good_circles, :]
                                                
                    if not(circle_coordinates.any()):
                        # if list is empty
                        circle_coordinates = None

                # Expanding the radius range  
                range_margin = range_margin+0.01
                min_dist_center = max(min_dist_center/2.0,  1.0)
                param2scale = param2scale*1.1
            
            # Update avers and revers attributes,  second argument
            # circle_coordinates[0, :,2].argmax(0) is the circle with largest diameter
            # 0 is the best circle
            # Coordinate wihtin the original picture
            getattr(self,  ravers[0]).circleCoordinates = np.uint16(
                np.around(circle_coordinates[0,  circle_coordinates[0, :, 2].argmax(0), :]
                          + np.array([ravers[1], 0,  0]))
                                                                   )
            # Circle boundaries in the original picture            
            getattr(self,  ravers[0]).radius = getattr(self,  ravers[0]).circleCoordinates[2]
            taille = 2*(self.marge_ravers+getattr(self,  ravers[0]).circleCoordinates[2])
            getattr(self,  ravers[0]).im_BRG = 255*np.ones((taille,  taille,  3)).astype(np.uint8)
            
            # Copy image
            bornes = [getattr(self, ravers[0]).circleCoordinates[0]-getattr(self,  ravers[0]).circleCoordinates[2]
                      - self.marge_ravers,
                      getattr(self, ravers[0]).circleCoordinates[0]+getattr(self,  ravers[0]).circleCoordinates[2]
                      + self.marge_ravers,
                      getattr(self, ravers[0]).circleCoordinates[1]-getattr(self,  ravers[0]).circleCoordinates[2]
                      - self.marge_ravers,
                      getattr(self, ravers[0]).circleCoordinates[1]+getattr(self,  ravers[0]).circleCoordinates[2]
                      + self.marge_ravers]
             
            getattr(self,  ravers[0]).boundaries = [int(max(0,                      bornes[0])),
                                                    int(min(self.coinGray.shape[1], bornes[1])),
                                                    int(max(0,                      bornes[2])),
                                                    int(min(self.coinGray.shape[0], bornes[3]))]
            
            getattr(self,  ravers[0]).im_BRG[
                            max(0, -0-bornes[2]):(taille-max(0,  bornes[3]-0-self.coinGray.shape[0])):1,
                            max(0, -0-bornes[0]):(taille-max(0,  bornes[1]-0-self.coinGray.shape[1])):1,
                            ::] = self.coinBGR[
                                    getattr(self,  ravers[0]).boundaries[2]:getattr(self,  ravers[0]).boundaries[3]:1,
                                    getattr(self,  ravers[0]).boundaries[0]:getattr(self,  ravers[0]).boundaries[1]:1,
                                    ::]

            # Update attributes
            getattr(self,  ravers[0]).xc = float(self.marge_ravers+getattr(self,  ravers[0]).circleCoordinates[2])
            getattr(self,  ravers[0]).yc = float(self.marge_ravers+getattr(self,  ravers[0]).circleCoordinates[2])
            getattr(self,  ravers[0]).radius = float(getattr(self,  ravers[0]).radius)
                
    def update_numismatic(self, excelpath, posRef, posAnn, posLet):
        """
        # Update coin properties based on a excel file
        """
        book = open_workbook(excelpath,  on_demand=True)
        head,  tail = os.path.split(os.path.splitext(self.picture_path)[0])
        # words[0] == Reference 2 in excel and words[1] == annee == col 3 in excel and 
        # words[2] == lettre atelier == col 4 in excel 
        words = [item.replace(" ", "") for item in tail.split("_")]
        # Initialize numismatic information
        self.numismatic = []       
        # We look for the info in the excel table
        for name in book.sheet_names():
            # Get current sheet
            sheet = book.sheet_by_name(name)
            # Get column valu with no string
            list_annee  = [re.sub('\s', '', str(item).replace(".0", "")) for item in sheet.col_values(3)]
            list_lettre = [re.sub('\s', '', unicodedata.normalize('NFKD', str(item)).encode('ascii', 'ignore'))
                          for item in sheet.col_values(4)]
            # Try to see if coin belongs to this sheet
            # 0
            contain_reference = name.replace(" ", "") == words[posRef]
            # 1
            contain_annee  = str(words[posAnn]) in list_annee
            if posLet < 0:
                contain_lettre = True
            else:
                # 2
                contain_lettre = words[posLet] in list_lettre
            if contain_reference and contain_annee and contain_lettre:
                # On cherche la meme annee (colonne 3)
                indSel1 = [i for i,  x in enumerate(list_annee) if x == words[posAnn]]
                # On cherche le meme numero d'atelier (colonne 4)
                if posLet < 0:
                    indSel2 = [i for i,  x in enumerate(list_lettre) if True]
                else:
                    indSel2 = [i for i,  x in enumerate(list_lettre) if x == words[posLet]]
                # print list(set(indSel1) & set(indSel2)) 
                for kind,  k in enumerate(list(set(indSel1) & set(indSel2))):
                    # We store as a list ...
                    self.numismatic.append([sheet.name,  sheet.row_values(k)])
                    # ... and as separated attributes
                    self.Pays 		   = sheet.row_values(k)[0]
                    self.Regnant       = sheet.row_values(k)[1]
                    self.Valeur		   = sheet.row_values(k)[2]
                    self.Date		   = sheet.row_values(k)[3]
                    self.Lettre		   = sheet.row_values(k)[4]
                    self.Atelier	   = sheet.row_values(k)[5]
                    self.LegendeAvers  = sheet.row_values(k)[6]
                    self.LegendeRevers = sheet.row_values(k)[7]
                    self.LegendeRevers2= sheet.row_values(k)[8]
                    self.Metal		   = sheet.row_values(k)[9]
                    self.Poids		   = sheet.row_values(k)[10]
                    self.Diametre	   = sheet.row_values(k)[11]
                    self.Reference1	   = sheet.row_values(k)[12]
                    self.Reference2	   = sheet.row_values(k)[13]
                    self.Reference3	   = sheet.row_values(k)[14]

    """
    FEATURE MATCHING
    """
    def generate_feature(self, featurePath):
        """
        Extract features/ROI (region of interest) from the two sides (avers and revers)
        """
        ravers = "avers", "revers"
        head,  tail = os.path.split(os.path.splitext(self.picture_path)[0])
        # Each side
        for k in [0,  1]:
            # Each feature/ROI for template matching
            for n_coord,  coord in enumerate(self.decoupageRavers[k]):
                chemin = os.path.join(featurePath,  'Feature', tail+"_"+ravers[k]+"_Feature"+str(n_coord))
                # Crop and save
                image = getattr(self,  ravers[k]).im_BRG[int(getattr(self,  ravers[k]).xc +
                                        getattr(self,  ravers[k]).radius*coord[0]):
                                     int(getattr(self,  ravers[k]).xc+getattr(self,  ravers[k]).radius*coord[2]):1,
                                     int(getattr(self,  ravers[k]).yc +
                                        getattr(self,  ravers[k]).radius*coord[1]):
                                     int(getattr(self,  ravers[k]).yc+getattr(self,  ravers[k]).radius*coord[3]):1, ::]
                cv2.imwrite(chemin+".png", image)
                # Meta data written in an external file
                with open(chemin+".csv", 'wb') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([["path", self.picture_path],
                           ["numismatic", self.numismatic],
                           ["side_type", ravers[k]],
                           ["relative_coord", str(coord)],
                           ["radius", str(getattr(self,  ravers[k]).radius)],
                           ["center", str([getattr(self,  ravers[k]).xc,  getattr(self,  ravers[k]).yc])]])
                       
    def register_coin(self,  reference, matchType):
        """
        Match feature for avers and revers
        """
        # Initiate returns        
        matches, error_avers, error_revers, M1_avers, M1_revers, titre_avers, titre_revers = [], [], [], [], [], [], []
        # Arithmetic mean
        global_error = 0
        list_error = []     

        if isinstance(reference,  list):
            # If reference is a patch with a text file list basestring
            error_avers  = self.avers.match_features(reference,  matchType)
            error_revers = self.revers.match_features(reference,  matchType)
            return matches,  error_avers,  error_revers
        
        elif isinstance(reference,  Coin):
            # Extract ROI on the two sides as two sides with meta data
            ravers = "avers", "revers"
            # Instance piece for reference
            ref_Piece = Piece(reference.uploadfile.path)
            ref_Piece.generate_aversrever()
            
            for k_1 in [0,  1]:
                # For each side
                ref_side = getattr(ref_Piece,  ravers[k_1])
                for n_coord,  coord in enumerate(self.decoupageRavers[k_1]): # les templates
                    # Generate cropped reference image
                    ref_crop = cv2.cvtColor(ref_side.im_BRG[int(ref_side.xc+ref_side.radius*coord[0]):
                                                            int(ref_side.xc+ref_side.radius*coord[2]):1,
                                                            int(ref_side.yc+ref_side.radius*coord[1]):
                                                            int(ref_side.yc+ref_side.radius*coord[3]):1, ::],
                                            cv2.COLOR_BGR2GRAY)
                    # Generate query
                    res = getattr(self,  ravers[k_1]).match_features(ref_crop,  matchType,  coord,  ref_side.radius)
                    error, M1, titre = res[0], res[1], res[2]
                    
                    # Update errors
                    if k_1 == 0:
                        error_avers.append(error)
                        M1_avers.append(M1)
                        titre_avers.append(titre)
                    elif k_1 == 1:
                        error_revers.append(error)
                        M1_revers.append(M1)
                        titre_revers.append(titre)

                    # Compute match
                    if matchType == 'ORB':
                        matches.append(all(error[k] < self.threshold_feature[k] for k in range(0,  len(error))))
                        # Arithmetic average for the score
                        error_local = sum([error[k]/self.threshold_feature[k] for k in range(0,  len(error))])
                        global_error = global_error + error_local
                        list_error.append(error_local)
                    elif matchType == 'Correlation1' or matchType == 'Correlation2':
                        matches.append(float(error) < float(self.threshold_corr))
                        # Arithmetic average for the score
                        error_local = float(error)/float(self.threshold_corr)
                        global_error = global_error + error_local
                        list_error.append(error_local)

            # Keep the best Nbestm pattern matches            
            Nbestm = 4
            list_error.sort()
            if matchType == 'ORB':
                len_error = float(len(error))
            elif matchType == 'Correlation1' or matchType == 'Correlation2':
                len_error = 1.0

            # Final error/score is the arithmetic mean
            global_error = float(sum(list_error[0:Nbestm:1]))/(len_error*float(Nbestm))

            return global_error

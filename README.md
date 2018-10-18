# Recoinition, a tool for coin recognition

# 1- Overview
The objective here is to identify a coin based on a phone-camera picture.

There are a few difficulties that make this challenge a little bit more complicated than bill or form identificaiton:
* letters and numbers are engraved and not printed, meaning that character contours and color are more complex and less reproductible
* metal is more ductible, so marks and shape alteration are more common
* the amount of real pictures available on the internet is scarcer
* character orientation can vary quite significantly
* experts expectation on the level on required level of identification. Check out the American Numismatic Association Coin Grading Scale to give you an idea.

On the good side, the coins are well more structured and studied than random images. For example a list of expected fields are supposed to appear on every single coin.

TODO:
* improve network to get better performance on OCR


# 2- Algorithm
* Image Preprocessing
  * Median-filter like (e.g. anisotropic gradient, guided-filter, bilateral-filter)
image here
  
  * Colors are altered in a HSL (hue, saturation, lightness) color space. Coin color can be more easily discriminated from background color in HSL rather than in RGB.
image here
 
  * Coin contour extraction a circle Hough transform
image here
 
  * Coin to stripe transform. Since some characters are engraved following the coin circumference, the coin is transformed into a stripe to make these characters follow a horizontal line. This corresponds to the diffeomorphism in cartesian frame (x=r \times \cos \theta, y=\r \times \sin \theta)->(x=r, x=\theta).
image here
  
* Feature generation
 * Image feature matching using OpenCV's ORB algorithm (Oriented FAST and Rotated BRIEF)
 * Image feature registration based on a library of features. This produces a first vector of features with assiciated probabilities 
  
 * OCR: a neural network is used to read the different characters on the coin (original and stripe version). This produces a list of texts that have been identified in different areas of the coin
 * Feature registration 
 * Classification
   * OCR results and features are 

SVM


# 3- Results




# 4- References




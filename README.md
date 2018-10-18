# Recoinition, a tool for coin recognition

# 1- Overview
The objective here is to identify a coin based on a phone-camera picture.

There are a few difficulties that make this challenge a little bit more complicated than bill or form identificaiton:
* letters and numbers are engraved and not printed, meaning that character contours and color are more complex and less reproductible
* metal is more ductible, so marks and shape alteration are more common
* the amount of real pictures available on the internet is scarcer
* character orientation can vary quite significantly
* experts expectation on the level on required level of identification. Check out the American Numismatic Association Coin Grading Scale to give you an idea.

On the good side, the coins are well more structured than random images.

TODO:
* improve network to get better performance on OCR


# 2- Algorithm
* Image Preprocessing
  * Median-filter like (e.g. anisotropic gradient, guided-filter, bilateral-filter)
image here
  
  * Colors are altered in a HSL (hue, saturation, lightness) color space. Coin color can be more easily discriminated from background color in HSL rather than in RGB.
image here
 
  * coin contour extraction a circle Hough transform
image here
 
  * coin to stripe transform. Since some characters are engraved following the coin circumference, the coin is transformed into a stripe to make these characters follow a horizontal line
image here
  
* Processing: the idea here is to extract 
 * OCR: a neural network is used . D
 
 * Feature registration
 * Quality assessment
* Classification

SVM


# 3- Results




# 4- References




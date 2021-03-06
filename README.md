# Recoinition, a tool for coin recognition

# 1- Overview
The objective here is to identify a coin based on a phone-camera picture. There are a few difficulties that make this challenge a little bit more complicated than bill or form identification:
* letters and numbers are engraved and not printed, meaning that character contours and color are more complex and less reproductible
* metal is more ductible, so marks and shape alteration are more common
* the amount of real pictures available on the internet is scarce
* character orientation can vary quite significantly
* experts expectation on the level on required level of identification. Check out the American Numismatic Association Coin Grading Scale to give you an idea.

On the good side, the coins are well more structured and studied than random images. For example a list of expected fields are supposed to appear on every single coin.

TODO:
* upload feature matching
* upload letter extractor
* upload coin letter generator to train the OCR engine
* upload OCR engine demonstrator

# 2- Algorithm
* Assuming the raw image follows coin standards

![base coin image](https://github.com/GreenIron/Recoinition/blob/master/Examples/1) 
* Image Preprocessing
  * Median-filter like (e.g. anisotropic gradient, guided-filter, bilateral-filter)
  
![coin avers](https://github.com/GreenIron/Recoinition/blob/master/Examples/5)
  * Colors are altered in a HSL (hue, saturation, lightness) color space. Coin color can be more easily discriminated from background color in HSL rather than in RGB.
  * Coin contour extraction a circle Hough transform
  
 ![contour revers](https://github.com/GreenIron/Recoinition/blob/master/Examples/2)
![contour revers](https://github.com/GreenIron/Recoinition/blob/master/Examples/3)
  * Coin to stripe transform. Since some characters are engraved following the coin circumference, the coin is transformed into a stripe to make these characters follow a horizontal line. This corresponds to the diffeomorphism in cartesian frame (x=r*cos theta, y=r*sin theta)->(x=r, x=theta).
  
![contour revers](https://github.com/GreenIron/Recoinition/blob/master/Examples/6)
![contour revers](https://github.com/GreenIron/Recoinition/blob/master/Examples/7)
* Feature generation
 * "Classic" feature extraction
   * Image feature matching using OpenCV's ORB algorithm (Oriented FAST and Rotated BRIEF)
   * Image feature registration based on a library of features scraped from the internet. This produces a first vector of features with associated probabilities
   => At this point we have a first vector of features with matching scores
* Text recognition
   * Letter and words detection
    OCR: a neural network (CNN based on tensorflow) is used to read the different characters on the coin (original and stripe version). The training set is generated from a open-source fonts and using random rotations, alterations, speckle... Once the OCR is trained and applied, it produces a list of texts that have been identified in different areas of the coin. Please note that I've tried to use existing OCR solutions, with very poor results. One of the problems might be that the letters are a unique depth/shadow pattern because they are 3D printed)
 => we end up with a second vector of words/features
* Classification: previously extracted features are compared with the existing library of coins (extracted from the web via web-scraping) 

# 4- References
* Open CV
* scki-image 
* guided and bilateral filter
* tensorflow
* image registration

